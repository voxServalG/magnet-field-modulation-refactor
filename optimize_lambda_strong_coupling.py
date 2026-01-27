import numpy as np
import pathlib
import matplotlib.pyplot as plt
import sys
import time
import pandas as pd
import seaborn as sns
from scipy.interpolate import UnivariateSpline

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src.factory import CoilFactory
from src.array_manager import ArrayActiveShielding
from src import physics

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def generate_fixed_background(target_points, seed=42):
    '''Generate a consistent background field for all trials'''
    np.random.seed(seed)
    num_dipoles = 5
    dipole_positions = []
    dipole_moments = []
    min_dist = 4.0
    max_dist = 6.0
    for _ in range(num_dipoles):
        r = np.random.uniform(min_dist, max_dist)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        pos = np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
        dipole_positions.append(pos)
        moment_mag = np.random.uniform(5e4, 2e5) 
        moment_vec = np.random.randn(3)
        moment_vec /= np.linalg.norm(moment_vec) 
        moment_vec *= moment_mag
        dipole_moments.append(moment_vec)
        
    def calculate_dipole_B(points, pos, moment):
        mu0_4pi = 1e-7
        R = points - pos
        r_norms = np.linalg.norm(R, axis=1)[:, np.newaxis] 
        dot_product = np.sum(moment * R, axis=1)[:, np.newaxis] 
        B = (3 * R * dot_product / r_norms**2 - moment) / r_norms**3
        return mu0_4pi * B

    B_vec = np.zeros_like(target_points)
    for pos, mom in zip(dipole_positions, dipole_moments):
        B_vec += calculate_dipole_B(target_points, pos, mom)
    
    # Scale to ~100nT
    current_mean = np.mean(np.linalg.norm(B_vec, axis=1))
    B_vec *= (100e-9 / current_mean)
    return B_vec

def calculate_curvature(log_lambdas, log_residuals, log_solutions, smoothing=0.0):
    '''
    Calculate curvature of the L-Curve using splines. 
    
    Parametrized by x(lambda) = log_residual, y(lambda) = log_solution.
    Curvature k = (x'y" - x"y') / (x'^2 + y'^2)^(1.5)
    Derivatives are w.r.t log_lambda.
    '''
    # Fit splines
    # k=3 (cubic), s=smoothing factor
    # We sort by log_lambda just in case (though input should be sorted)
    order = np.argsort(log_lambdas)
    t = log_lambdas[order]
    x = log_residuals[order]
    y = log_solutions[order]
    
    # Splines for x(t) and y(t)
    x_spline = UnivariateSpline(t, x, k=3, s=smoothing)
    y_spline = UnivariateSpline(t, y, k=3, s=smoothing)
    
    # Derivatives
    x_d1 = x_spline.derivative(1)(t)
    x_d2 = x_spline.derivative(2)(t)
    y_d1 = y_spline.derivative(1)(t)
    y_d2 = y_spline.derivative(2)(t)
    
    # Curvature formula
    curvature = (x_d1 * y_d2 - x_d2 * y_d1) / np.power(x_d1**2 + y_d1**2, 1.5)
    
    return t, curvature

# ---------------------------------------------------------
# Main Scanning Logic
# ---------------------------------------------------------
def run_strong_coupling_scan():
    print("=======================================================================")
    print("   Project Nuke: Strong Coupling L-Curve Scan (Shielding Enabled)      ")
    print("=======================================================================")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / f"shielded_scan_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup System
    # Define Shield Dimensions (Strong Coupling Environment)
    SHIELD_DIMS = (2.4, 1.9, 1.65) # x, y, z half-dims in meters
    
    config = {
        'L': 0.6, 'a': 0.7, 'modes': (4, 4), 'reg_lambda': 1e-12, 'num_turns': 20, 'grid_res': 60,
        'shield_dims': SHIELD_DIMS
    }
    factory = CoilFactory(config)
    s = 1.25
    layout = np.array([[x, y, 0.0] for x in [-s,0,s] for y in [-s,0,s]])
    
    print(f"\n[Init] Initializing Array with Shielding: {SHIELD_DIMS}")
    # Enable Gradients + Shielding support
    manager = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=SHIELD_DIMS)
    
    # ROI (Region of Interest)
    limit = 1.5; Np = 15 # Coarse grid for speed, or finer? Np=15 is okay.
    gx = np.linspace(-limit, limit, Np)
    GX, GY, GZ = np.meshgrid(gx, gx, gx, indexing='ij')
    target_points = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    center_spot = np.array([0.0, 0.0, 0.0]) 
    dist_from_spot = np.linalg.norm(target_points - center_spot, axis=1)
    region_mask = dist_from_spot < 0.30 # 60cm cube fit (slightly larger than previous test)
    
    print("\n[Init] Computing S Matrix (with Shielding Reflections)...")
    print("       Note: This may take a moment due to mirror current calculations.")
    S = manager.compute_response_matrix(target_points)
    
    print("\n[Init] Generating Background Field...")
    B_bg_vec = generate_fixed_background(target_points)
    B_bg_flat = B_bg_vec.flatten()
    B_bg_roi = B_bg_vec[region_mask]
    rms_bg = np.sqrt(np.mean(np.linalg.norm(B_bg_roi, axis=1)**2)) * 1e9
    print(f"  -> Baseline RMS: {rms_bg:.2f} nT")
    
    # 2. Sweep Parameters
    # Full panorama scan to catch the true elbow
    lambdas = np.logspace(-17, 0, 101) # 1e-30 to 1.0, 61 points
    
    log_lambdas = []
    log_resid_norms = []
    log_sol_norms = []
    results = []
    legend_lines = []  # Store lines for txt output
    
    print(f"\n[Scan] Starting scan over {len(lambdas)} lambda values...")
    
    # Pre-calculate common terms for speed
    full_mask = np.repeat(region_mask, 3)
    S_active = S[full_mask, :]
    rhs = -B_bg_flat
    rhs_active = rhs[full_mask]
    STS = S_active.T @ S_active
    STb = S_active.T @ rhs_active
    
    for i, lam in enumerate(lambdas):
        # Solve efficiently reusing pre-calculated matrices
        num_ch = STS.shape[0]
        lhs = STS + lam * np.eye(num_ch)
        x_opt = np.linalg.solve(lhs, STb)
        
        # Calculate Norms
        # Residual Norm: ||Sx - b||
        residual_vec = S_active @ x_opt - rhs_active
        resid_norm = np.linalg.norm(residual_vec)
        
        # Solution Norm: ||x||
        sol_norm = np.linalg.norm(x_opt)
        
        # Log values for L-Curve
        log_lam = np.log10(lam)
        log_res = np.log10(resid_norm)
        log_sol = np.log10(sol_norm)
        
        log_lambdas.append(log_lam)
        log_resid_norms.append(log_res)
        log_sol_norms.append(log_sol)
        
        # Metrics for report
        rms_res = (resid_norm / np.sqrt(np.sum(region_mask))) * 1e9 # approx RMS in nT
        suppression_db = 20 * np.log10(rms_bg / rms_res)
        mean_current_weight = np.mean(np.abs(x_opt))
        
        # Prepare legend string
        lam_str = f"{lam:.1e}"
        legend_str = f"#{i}: Lambda={lam_str} | LogRes={log_res:.2f}, LogSol={log_sol:.2f} | dB={suppression_db:.1f}"
        legend_lines.append(legend_str)

        results.append({
            'Lambda': lam,
            'Log_Lambda': log_lam,
            'Log_Residual_Norm': log_res,
            'Log_Solution_Norm': log_sol,
            'RMS_Residual_nT': rms_res,
            'Suppression_dB': suppression_db,
            'Mean_Weight': mean_current_weight
        })
        
        if (i+1) % 5 == 0:
             print(f"  -> Progress {i+1}/{len(lambdas)}: L={lam:.1e}, dB={suppression_db:.1f}, |x|={sol_norm:.1f}")

    # Save Legend Data to TXT
    with open(output_dir / "legend_data.txt", "w") as f:
        f.write("Index | Lambda | Residual(Log) | Solution(Log) | Suppression(dB)\n")
        f.write("=================================================================\n")
        for line in legend_lines:
            f.write(line + "\n")
    print(f"  -> Legend data saved to {output_dir / 'legend_data.txt'}")

    # 3. Analyze Curvature (Automatic Optimal Detection)
    print("\n[Analysis] Calculating L-Curve Curvature...")
    
    t_arr = np.array(log_lambdas)
    x_arr = np.array(log_resid_norms)
    y_arr = np.array(log_sol_norms)
    
    try:
        _, kappa = calculate_curvature(t_arr, x_arr, y_arr, smoothing=0.05)
        
        # Find index of max curvature
        best_idx = np.argmax(kappa)
        best_lambda = lambdas[best_idx]
        best_log_lambda = log_lambdas[best_idx]
        
        print(f"  -> *** OPTIMAL LAMBDA FOUND: {best_lambda:.2e} (Log: {best_log_lambda:.2f}) ***")
    except Exception as e:
        print(f"  -> Curvature calculation failed: {e}")
        best_lambda = None
        kappa = np.zeros_like(t_arr)

    # 4. Save & Plot
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "shielded_scan_results.csv", index=False)
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: L-Curve with Detailed Annotation
    # Increase figure width to accommodate the side legend
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Draw the connecting line (faintly) to show the path
    ax.plot(df['Log_Residual_Norm'], df['Log_Solution_Norm'], 'k-', alpha=0.3, zorder=1)
    
    # 2. Plot points with color mapping (Lambda value)
    # Use a colormap to visualize lambda progression
    sc = ax.scatter(df['Log_Residual_Norm'], df['Log_Solution_Norm'], 
                   c=np.log10(df['Lambda']), cmap='viridis', s=50, zorder=2, edgecolors='k', linewidth=0.5)
    
    # Highlight Optimal Point
    if best_lambda is not None:
         best_row = df.iloc[best_idx]
         ax.scatter(best_row['Log_Residual_Norm'], best_row['Log_Solution_Norm'], 
                    color='red', s=200, marker='*', zorder=3, 
                    label=f"Optimal $\lambda={best_lambda:.1e}$")
         ax.legend(loc='upper right', frameon=True, fontsize=12)

    # Colorbar to show Lambda values
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'$\log_{10}(\lambda)$', fontsize=12)

    ax.set_xlabel(r'$\log_{10} ||\mathbf{Ax}-\mathbf{b}||_2$ (Residual)', fontsize=12)
    ax.set_ylabel(r'$\log_{10} ||\mathbf{x}||_2$ (Solution Norm)', fontsize=12)
    ax.set_title(f"L-Curve Analysis (Shielded Environment)", fontsize=14)
    
    # 4. Add Arrows indicating Low -> High values
    # X-Axis Arrow (Residual)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    # Simplified annotations
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "l_curve.svg")
    plt.savefig(output_dir / "l_curve.pdf") # PDF for LaTeX
    plt.close()
    
    # Plot 2: Curvature vs Lambda
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['Log_Lambda'], kappa, 'purple', label='Curvature', linewidth=2)
    ax1.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=12)
    ax1.set_ylabel('Curvature $\kappa$', color='purple', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='purple')
    
    if best_lambda is not None:
        ax1.axvline(best_log_lambda, color='r', linestyle='--', alpha=0.8, label='Max Curvature')
    
    ax2 = ax1.twinx()
    ax2.plot(df['Log_Lambda'], df['Suppression_dB'], 'g--', alpha=0.6, label='Suppression dB', linewidth=1.5)
    ax2.set_ylabel('Suppression (dB)', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title("L-Curve Curvature Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "curvature_analysis.svg")
    plt.savefig(output_dir / "curvature_analysis.pdf") # PDF for LaTeX
    plt.close()
    
    print(f"\n[Done] Results saved to {output_dir}")

if __name__ == "__main__":
    run_strong_coupling_scan()
