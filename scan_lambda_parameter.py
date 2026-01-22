import numpy as np
import pathlib
import matplotlib.pyplot as plt
import sys
import time
import pandas as pd
import seaborn as sns

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
    # Similar logic to run_sim but simplified
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

def calculate_power(system):
    rho = 1.68e-8 
    dia = 2.26e-3 # 4mm^2 wire
    area = np.pi * (dia/2)**2
    res_per_m = rho / area 
    
    total_watts = 0.0
    for top, bot, I_top, I_bot in system:
        # Top Loop
        d = np.diff(top, axis=0)
        len_top = np.sum(np.sqrt(np.sum(d**2, axis=1)))
        total_watts += (I_top**2) * (len_top * res_per_m)
        # Bot Loop
        d = np.diff(bot, axis=0)
        len_bot = np.sum(np.sqrt(np.sum(d**2, axis=1)))
        total_watts += (I_bot**2) * (len_bot * res_per_m)
    return total_watts

# ---------------------------------------------------------
# Main Scanning Logic
# ---------------------------------------------------------
def run_scan():
    print("========================================================")
    print("   Project Nuke: Lambda Parameter Scan (L-Curve)        ")
    print("========================================================")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / f"scan_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup System (Fixed)
    # Using the Independent Plate Control setup (12 ch/unit)
    SHIELD_DIMS = (2.4, 1.9, 1.65)
    config = {
        'L': 0.6, 'a': 0.7, 'modes': (4, 4), 'reg_lambda': 1e-14, 'num_turns': 20, 'grid_res': 60,
        'shield_dims': SHIELD_DIMS
    }
    factory = CoilFactory(config)
    s = 1.25
    layout = np.array([[x, y, 0.0] for x in [-s,0,s] for y in [-s,0,s]])
    
    # Important: Enable Gradients + Shielding support
    # Note: We keep DEMO_MODE_FREE_SPACE = True equivalent by setting shield_dims=None 
    # to scan the 'ideal' physics first, or we can scan the 'shielded' physics.
    # Let's assume we want to tune for the FREE SPACE case first to find the intrinsic trade-off.
    manager = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=None)
    
    # ROI
    limit = 1.5; Np = 15
    gx = np.linspace(-limit, limit, Np)
    GX, GY, GZ = np.meshgrid(gx, gx, gx, indexing='ij')
    target_points = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    center_spot = np.array([0.0, 0.0, 0.0]) 
    dist_from_spot = np.linalg.norm(target_points - center_spot, axis=1)
    region_mask = dist_from_spot < 0.25 # 400mm cube fit
    
    print("\n[Init] Computing S Matrix...")
    S = manager.compute_response_matrix(target_points)
    
    print("\n[Init] Generating Background Field...")
    B_bg_vec = generate_fixed_background(target_points)
    B_bg_flat = B_bg_vec.flatten()
    B_bg_roi = B_bg_vec[region_mask]
    rms_bg = np.sqrt(np.mean(np.linalg.norm(B_bg_roi, axis=1)**2)) * 1e9
    print(f"  -> Baseline RMS: {rms_bg:.2f} nT")
    
    # 2. Sweep Parameters
    # Logspace from 1e-10 down to 1e-30
    lambdas = np.logspace(-10, -30, 21)
    results = []
    
    print(f"\n[Scan] Starting scan over {len(lambdas)} lambda values...")
    
    for lam in lambdas:
        print(f"  -> Testing lambda = {lam:.2e} ...", end="")
        
        # Solve
        x_opt, _ = manager.solve_optimization(B_bg_flat, S, region_mask=region_mask, reg_lambda=lam)
        
        # Verify (Fast physics check)
        # Note: We calculate power directly from x_opt combined with geometry
        sys, _ = manager.get_final_system(x_opt)
        
        # Power
        power = calculate_power(sys)
        
        # Residual Field
        # We can use matrix multiplication S*x + B for fast checking
        # But S is full domain. We need to mask it.
        # B_total_flat = S @ x_opt + B_bg_flat
        # B_total_vec = B_total_flat.reshape(-1, 3)
        # B_total_roi = B_total_vec[region_mask]
        # rms_res = np.sqrt(np.mean(np.linalg.norm(B_total_roi, axis=1)**2)) * 1e9
        
        # Let's use the robust physics engine for consistency
        B_coil = physics.calculate_field_from_coils(sys, target_points[region_mask], use_shielding=False)
        B_res = B_coil + B_bg_roi
        rms_res = np.sqrt(np.mean(np.linalg.norm(B_res, axis=1)**2)) * 1e9
        
        suppression_db = 20 * np.log10(rms_bg / rms_res)
        mean_current = np.mean(np.abs(x_opt))
        
        # Calculate Max Current (Actual Amperes)
        # We need to peek into the final system to get real Amperes, not just weights
        all_currents = []
        for _, _, it, ib in sys:
            all_currents.append(abs(it))
            all_currents.append(abs(ib))
        max_current_amp = max(all_currents) if all_currents else 0.0
        mean_current_amp = np.mean(all_currents) if all_currents else 0.0
        
        results.append({
            'Lambda': lam,
            'Log_Lambda': np.log10(lam),
            'Suppression_dB': suppression_db,
            'RMS_Residual_nT': rms_res,
            'Power_W': power,
            'Mean_Current_Amp': mean_current_amp,
            'Max_Current_Amp': max_current_amp
        })
        print(f" dB={suppression_db:.2f}, Power={power:.4f}W, MaxI={max_current_amp:.2f}A")

    # 3. Save & Plot
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "scan_results.csv", index=False)
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: L-Curve (dB vs Power)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Log10(Lambda)')
    ax1.set_ylabel('Suppression (dB)', color=color)
    ax1.plot(df['Log_Lambda'], df['Suppression_dB'], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Power Consumption (W)', color=color)
    ax2.semilogy(df['Log_Lambda'], df['Power_W'], 'x--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title("Trade-off: Suppression vs Power")
    plt.savefig(output_dir / "lambda_scan_l_curve.svg")
    plt.close()
    
    # Plot 2: Current Analysis (Lambda vs Current)
    plt.figure(figsize=(10, 6))
    plt.semilogy(df['Log_Lambda'], df['Mean_Current_Amp'], 'o-', label='Mean Current (A)')
    plt.semilogy(df['Log_Lambda'], df['Max_Current_Amp'], 'x--', label='Max Current (A)')
    plt.xlabel('Log10(Lambda)')
    plt.ylabel('Current (Amperes)')
    plt.title("Current Scaling vs Regularization")
    plt.legend()
    plt.savefig(output_dir / "lambda_scan_current.svg")
    plt.close()
    
    # Plot 3: Pareto Frontier (Current vs dB)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Mean_Current_Amp'], df['Suppression_dB'], 'o-')
    plt.xlabel('Mean Current (A)')
    plt.ylabel('Suppression (dB)')
    plt.title("Pareto Frontier: How much dB can 1 Ampere buy?")
    plt.xscale('log') # Log scale for current makes the knee clearer
    plt.grid(True, which="both", ls="-")
    plt.savefig(output_dir / "pareto_current_vs_db.svg")
    plt.close()
    
    print(f"\n[Done] Results saved to {output_dir}")

if __name__ == "__main__":
    run_scan()
