import numpy as np
import pathlib
import matplotlib.pyplot as plt
import sys
import time
from scipy import signal

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src.factory import CoilFactory
from src.array_manager import ArrayActiveShielding
from src import physics

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Simulation Settings
FS = 500.0          # Sampling Rate (Hz). 500Hz covers up to 250Hz Nyquist
DURATION = 10.0     # Seconds
T_AXIS = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
N_SAMPLES = len(T_AXIS)

# Physical Array Settings
# Using the optimized parameters found previously
ARRAY_REG_LAMBDA = 2.3e-16 # The 'Sweet Spot' from L-Curve
SHIELD_DIMS = (2.4, 1.9, 1.65) # Real MSR dimensions

# ROI Settings
ROI_RADIUS = 0.30 # 30cm radius sphere
GRID_DENSITY = 7  # Coarse grid for speed (7x7x7 ~ 343 points)

# Noise Source Settings
BG_MEAN_NT = 150.0 # DC Baseline approx
LINE_NOISE_NT = 5.0 # 50Hz amplitude
DRIFT_NOISE_NT = 20.0 # 1/f drift amplitude

# ---------------------------------------------------------
# Helper: Colored Noise Generator
# ---------------------------------------------------------
def generate_colored_noise(n_samples, alpha=1.0):
    """
    Generate 1/f^alpha noise.
    alpha=1.0 -> Pink Noise (1/f)
    alpha=2.0 -> Brown Noise (1/f^2)
    """
    # White noise in frequency domain
    white = np.random.randn(n_samples)
    X_white = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(n_samples)
    
    # Scale by 1/f^(alpha/2)
    # Avoid division by zero at f=0
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / (frequencies[1:] ** (alpha / 2.0))
    
    X_colored = X_white * scale
    
    # Normalize to unit variance
    colored = np.fft.irfft(X_colored, n=n_samples)
    colored /= np.std(colored)
    return colored

# ---------------------------------------------------------
# Main Dynamic Simulation
# ---------------------------------------------------------
def run_dynamic_psd_sim():
    print("========================================================")
    print("   Project Nuke: Dynamic PSD Analysis (4D Simulation)   ")
    print("========================================================")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / f"psd_sim_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Array & ROI
    # --------------------
    print(f"[Init] Setting up Array (Lambda={ARRAY_REG_LAMBDA:.1e})...")
    config = {
        'L': 0.6, 'a': 0.7, 'modes': (4, 4), 'reg_lambda': 1e-14, 'num_turns': 10, 
        'grid_res': 60, 'shield_dims': SHIELD_DIMS
    }
    factory = CoilFactory(config)
    
    # 3x3 Layout
    s = 1.25
    layout = np.array([[x, y, 0.0] for x in [-s,0,s] for y in [-s,0,s]])
    manager = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=SHIELD_DIMS)
    
    # ROI: 3D Grid inside Sphere
    gx = np.linspace(-ROI_RADIUS, ROI_RADIUS, GRID_DENSITY)
    GX, GY, GZ = np.meshgrid(gx, gx, gx, indexing='ij')
    pts_all = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    # Filter to sphere
    dist = np.linalg.norm(pts_all, axis=1)
    mask = dist <= ROI_RADIUS
    roi_points = pts_all[mask]
    n_roi = len(roi_points)
    print(f"[Init] ROI defined: {n_roi} points within R={ROI_RADIUS}m sphere.")
    
    # 2. Compute S Matrix (Once)
    # --------------------------
    print("[Init] Computing Response Matrix S (Static)...")
    # This matrix connects Currents (Input) -> Field at ROI (Output)
    # Since geometry is static, S is static.
    S_full = manager.compute_response_matrix(roi_points)
    
    # Pre-compute Inverse for real-time speed
    # x = (S'S + lambda*I)^-1 * S' * (-b)
    # Let K = (S'S + lambda*I)^-1 * S'
    # Then x = -K * b
    print("[Init] Pre-computing Inverse Solver Matrix K...")
    STS = S_full.T @ S_full
    num_ch = STS.shape[0]
    inv_lhs = np.linalg.inv(STS + ARRAY_REG_LAMBDA * np.eye(num_ch))
    K_matrix = inv_lhs @ S_full.T
    
    # 3. Generate 4D Background Noise (Space + Time)
    # ----------------------------------------------
    print(f"\n[Sim] Synthesizing 4D Noise Environment ({DURATION}s @ {FS}Hz)...")
    
    # Create "Phantom Sources"
    # We use 3 distant dipoles to create gradients
    source_positions = [
        np.array([5, 0, 0]),
        np.array([0, 5, 2]),
        np.array([-4, -4, -3])
    ]
    
    # Generate Time Series for each dipole's moment (Bx, By, Bz)
    # Total 3 dipoles * 3 components = 9 time series
    dipole_moments_t = np.zeros((N_SAMPLES, len(source_positions), 3))
    
    np.random.seed(42)
    
    for i in range(len(source_positions)):
        for j in range(3): # x, y, z components
            # Base: Pink Noise (Drift)
            drift = generate_colored_noise(N_SAMPLES, alpha=1.2) * DRIFT_NOISE_NT
            
            # Line: 50Hz
            line = LINE_NOISE_NT * np.sin(2 * np.pi * 50.0 * T_AXIS + np.random.rand()*2*np.pi)
            
            # DC Offset
            # Distribute mean field among sources randomly
            dc = (BG_MEAN_NT / len(source_positions)) * np.random.choice([0.5, 1.0, 1.5])
            
            series = (dc + drift + line)
            dipole_moments_t[:, i, j] = series

    # Calculate Field B(t) at all ROI points
    # Shape: (N_Samples, N_ROI_Points, 3)
    # This is the "Ground Truth" interference we want to cancel
    print("  -> calculating B_bg(t) for all frames...")
    
    B_bg_t_flat = np.zeros((N_SAMPLES, n_roi * 3))
    
    # Helper to calc field from multiple dipoles
    def get_B_from_dipoles(pts, positions, moments):
        B_total = np.zeros_like(pts)
        mu0_4pi = 1e-7
        for i, pos in enumerate(positions):
            R = pts - pos
            r2 = np.sum(R**2, axis=1)[:, np.newaxis]
            r5 = np.power(r2, 2.5)
            m_dot_r = np.sum(moments[i] * R, axis=1)[:, np.newaxis]
            r3 = np.power(np.sum(R**2, axis=1), 1.5)[:, np.newaxis]
            B_i = (3 * R * m_dot_r) / r5 - (moments[i] / r3)
            B_total += B_i
        return mu0_4pi * B_total

    # Loop frames
    for t in range(N_SAMPLES):
        moments_now = dipole_moments_t[t] 
        B_vec = get_B_from_dipoles(roi_points, source_positions, moments_now)
        B_bg_t_flat[t] = B_vec.flatten()

    # --- NORMALIZATION STEP ---
    # Ensure the background field is actually in the nT range
    current_rms = np.sqrt(np.mean(B_bg_t_flat**2))
    target_rms = BG_MEAN_NT * 1e-9
    scaling_factor = target_rms / current_rms
    B_bg_t_flat *= scaling_factor
    print(f"  -> Normalized background noise to {BG_MEAN_NT} nT RMS.")
    # --------------------------

    # 4. Run "Real-Time" Compensation
    # -------------------------------
    print(f"\n[Sim] Running Compensation Loop (Matrix Mul)...")
    # x(t) = -K * b(t)
    # B_res(t) = S * x(t) + b(t)
    #          = S * (-K * b(t)) + b(t)
    #          = (I - S*K) * b(t)
    
    # We can do this with one giant matrix multiplication
    # B_bg_t_flat: (T, 3M) 
    # K: (3K_coils, 3M)
    # S: (3M, 3K_coils)
    # This might be memory heavy if T or M is huge.
    # T=5000, M=300 -> 3M=900. 5000x900 is small. Safe.
    
    # 1. Calculate Currents
    # X_opt_t = (-K @ B_bg_t_flat.T).T
    # Actually K is (N_coil, N_meas).
    # X_opt_t shape: (T, N_coil)
    X_opt_t = - (K_matrix @ B_bg_t_flat.T).T
    
    # 2. Calculate Residual Field
    # B_coil_t = (S @ X_opt_t.T).T
    # B_res_t = B_coil_t + B_bg_t_flat
    
    # Shortcut: Residual Operator R = (I - S @ K)
    # But let's do it explicitly to get coil field
    B_coil_t_flat = (S_full @ X_opt_t.T).T
    B_res_t_flat = B_coil_t_flat + B_bg_t_flat
    
    # 5. Analysis & PSD
    # -----------------
    print("\n[Analysis] Computing Power Spectral Density (PSD)...")
    
    # We pick the "Center" point of the ROI for the main plot
    # Find index of point closest to origin
    center_idx = np.argmin(np.linalg.norm(roi_points, axis=1))
    
    # Extract Bx, By, Bz for center point
    # Structure of flat: [x1, y1, z1, x2, y2, z2...]
    # Index for center point k is: 3*k, 3*k+1, 3*k+2
    
    def get_field_at_point(flat_data, idx):
        bx = flat_data[:, 3*idx]
        by = flat_data[:, 3*idx + 1]
        bz = flat_data[:, 3*idx + 2]
        return bx, by, bz
    
    bg_x, bg_y, bg_z = get_field_at_point(B_bg_t_flat, center_idx)
    res_x, res_y, res_z = get_field_at_point(B_res_t_flat, center_idx)
    
    # Convert to nT
    bg_x *= 1e9; res_x *= 1e9
    
    # Calculate PSD using Welch's method
    freqs, psd_bg = signal.welch(bg_x, FS, nperseg=1024)
    _, psd_res = signal.welch(res_x, FS, nperseg=1024)
    
    # Calculate Attenuation (dB)
    # Avoid div by zero
    attenuation = 10 * np.log10(psd_bg / (psd_res + 1e-20))
    
    # 6. Plotting (Publication Quality)
    # ---------------------------------
    # Style settings for paper
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    # Plot 1: Time Domain (Snapshot)
    plt.figure(figsize=(12, 7))
    plt.subplot(2,1,1)
    plt.plot(T_AXIS, bg_x, 'r-', alpha=0.6, linewidth=1.0, label='Background (Unshielded)')
    plt.plot(T_AXIS, res_x, 'b-', linewidth=1.0, label='Residual (Active Shield)')
    plt.title(f"Time Domain Response (Static Target, Dynamic Noise)", fontsize=14)
    plt.ylabel("Magnetic Field (nT)")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2) # Zoom in first 2 seconds
    
    plt.subplot(2,1,2)
    plt.plot(T_AXIS, res_x, 'b-', linewidth=1.0, label='Residual Only (Zoomed)')
    plt.xlabel("Time (s)")
    plt.ylabel("Magnetic Field (nT)")
    plt.title("Residual Field Detail")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2)
    plt.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "1_time_domain.svg")
    plt.savefig(output_dir / "1_time_domain.pdf") # PDF for LaTeX
    plt.close()
    
    # Plot 2: PSD Comparison (Log-Log)
    plt.figure(figsize=(10, 8))
    
    # Subplot 1: Power Density
    ax1 = plt.subplot(2,1,1)
    ax1.semilogy(freqs, psd_bg, 'r', alpha=0.7, label='Background Noise', linewidth=1.5)
    ax1.semilogy(freqs, psd_res, 'b', alpha=0.9, label='Active Shielded', linewidth=1.5)
    ax1.set_title("Power Spectral Density (0.1 - 100 Hz)", fontsize=14)
    ax1.set_ylabel(r"Power Density ($nT^2/Hz$)")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_xlim(0.1, 100) # Brain frequency range
    
    # Highlight 50Hz
    ax1.axvline(50, color='k', linestyle='--', alpha=0.3)
    ax1.text(51, np.max(psd_bg)*0.1, "50Hz Line", fontsize=10, color='k')

    # Subplot 2: Attenuation dB
    ax2 = plt.subplot(2,1,2)
    ax2.plot(freqs, attenuation, 'g-', linewidth=1.5)
    ax2.set_title("Shielding Factor (Attenuation)", fontsize=14)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Attenuation (dB)")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.set_xlim(0.1, 100)
    ax2.set_ylim(0, 60) 
    
    # Highlight Mean Attenuation line
    mean_att = np.mean(attenuation[(freqs>0.1) & (freqs<100)])
    ax2.axhline(mean_att, color='g', linestyle='--', alpha=0.5, label=f'Mean: {mean_att:.1f} dB')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / "2_psd_analysis.svg")
    plt.savefig(output_dir / "2_psd_analysis.pdf") # PDF for LaTeX
    plt.close()

    # Calculate Statistics
    rms_bg = np.sqrt(np.mean(bg_x**2))
    rms_res = np.sqrt(np.mean(res_x**2))
    shield_factor_db = 20 * np.log10(rms_bg/rms_res)
    
    # Power Estimate (RMS Current)
    # X_opt_t: (T, N_coils)
    # Mean RMS current per channel
    rms_current_per_ch = np.sqrt(np.mean(X_opt_t**2, axis=0))
    total_rms_current_norm = np.linalg.norm(rms_current_per_ch)
    
    print(f"\n[Results]")
    print(f"  -> Time Domain RMS: {rms_bg:.2f} nT (Bg) -> {rms_res:.2f} nT (Res)")
    print(f"  -> Total Suppression: {shield_factor_db:.2f} dB")
    print(f"  -> System Current Load (RMS Norm): {total_rms_current_norm:.2f} A")
    print(f"  -> Saved plots to {output_dir}")

if __name__ == "__main__":
    run_dynamic_psd_sim()
