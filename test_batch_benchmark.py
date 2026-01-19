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

def generate_random_background(target_points, seed):
    '''
    Generates a reproducible random background field using phantom dipoles.
    '''
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
        pos = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        dipole_positions.append(pos)
        
        moment_mag = np.random.uniform(5e4, 2e5) 
        moment_vec = np.random.randn(3)
        moment_vec /= np.linalg.norm(moment_vec) 
        moment_vec *= moment_mag
        dipole_moments.append(moment_vec)
        
    def calculate_dipole_B(points, pos, moment):
        mu0_4pi = 1e-7
        R = points - pos
        r_norms = np.linalg.norm(R, axis=1)
        r_norms = r_norms[:, np.newaxis] 
        dot_product = np.sum(moment * R, axis=1)[:, np.newaxis] 
        B = (3 * R * dot_product / r_norms**2 - moment) / r_norms**3
        return mu0_4pi * B

    B_vec = np.zeros_like(target_points)
    for pos, mom in zip(dipole_positions, dipole_moments):
        B_vec += calculate_dipole_B(target_points, pos, mom)
        
    # Scale to ~100nT mean
    current_mean = np.mean(np.linalg.norm(B_vec, axis=1))
    target_mean = 100e-9
    B_vec *= (target_mean / current_mean)
    
    return B_vec

def analyze_components(pts, B_vec):
    '''
    Analyzes field components and estimates 1st order gradients via linear regression.
    Returns a dict of RMS values and gradient strengths.
    '''
    # 1. Component RMS
    rms_x = np.sqrt(np.mean(B_vec[:, 0]**2)) * 1e9
    rms_y = np.sqrt(np.mean(B_vec[:, 1]**2)) * 1e9
    rms_z = np.sqrt(np.mean(B_vec[:, 2]**2)) * 1e9
    
    # 2. Gradient Extraction (Linear Regression: B = G * r + B0)
    # We solve min || r_ext * G_row - B_comp ||^2
    # r_ext = [x, y, z, 1]
    r_ext = np.hstack([pts, np.ones((len(pts), 1))])
    
    # G_matrix shape will be (4, 3) -> columns are [dB/dx, dB/dy, dB/dz, offset] for each B component
    G_matrix, _, _, _ = np.linalg.lstsq(r_ext, B_vec, rcond=None)
    
    return {
        'Bx': rms_x, 'By': rms_y, 'Bz': rms_z,
        'Gxx': np.abs(G_matrix[0, 0]) * 1e9,
        'Gyy': np.abs(G_matrix[1, 1]) * 1e9,
        'Gxy': np.abs(G_matrix[1, 0] + G_matrix[0, 1]) * 0.5 * 1e9,
        'Gxz': np.abs(G_matrix[2, 0]) * 1e9,
        'Gyz': np.abs(G_matrix[2, 1]) * 1e9
    }

def run_benchmark():
    print("========================================================")
    print("   Project Nuke: Batch Benchmark (Order 0 vs Order 1)   ")
    print("========================================================")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[System] Output directory: {output_dir}")
    
    # 1. Setup Managers (Pre-load both systems)
    config = {
        'L': 0.85, 'a': 0.7, 'modes': (4, 4), 'reg_lambda': 1e-14, 'num_turns': 22, 'grid_res': 60
    }
    factory = CoilFactory(config)
    
    s = 1.8
    offsets = [-s, 0, s]
    layout_list = [[x, y, 0.0] for x in offsets for y in offsets]
    layout = np.array(layout_list)
    
    print("\n[Init] building System Order 0 (B0 Only) ...")
    mgr_0 = ArrayActiveShielding(factory, layout, use_gradients=False)
    
    print("\n[Init] building System Order 1 (B0 + Gradients) ...")
    mgr_1 = ArrayActiveShielding(factory, layout, use_gradients=True)
    
    # Define ROI
    Np = 15 # Moderate density for batch speed
    limit = 2.0 # Slightly smaller ROI to ensure we are inside the sweet spot
    gx = np.linspace(-limit, limit, Np)
    gy = np.linspace(-limit, limit, Np)
    gz = np.linspace(-limit, limit, Np)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    target_points = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    # Target Mask (Sphere)
    center_spot = np.array([1.0, 1.0, 0.0])
    radius_spot = 0.6
    dist_from_spot = np.linalg.norm(target_points - center_spot, axis=1)
    region_mask = dist_from_spot < radius_spot
    points_in_sphere = target_points[region_mask]
    
    print(f"\n[Init] Computing S matrices for {len(target_points)} points...")
    S_0 = mgr_0.compute_response_matrix(target_points)
    S_1 = mgr_1.compute_response_matrix(target_points)
    
    # 2. Run Batch Trials
    NUM_TRIALS = 10
    results = []
    
    print("\n[Benchmark] Starting 10 Trials...")
    
    for i in range(NUM_TRIALS):
        seed = 1000 + i
        print(f"  -> Trial {i+1}/{NUM_TRIALS} (Seed {seed})...")
        
        # Generate Common Enemy
        B_bg_vec = generate_random_background(target_points, seed)
        B_bg_flat = B_bg_vec.flatten()
        
        # Baseline Analysis
        stats_bg = analyze_components(points_in_sphere, B_bg_vec[region_mask])
        rms_bg_total = np.sqrt(np.mean(np.linalg.norm(B_bg_vec[region_mask], axis=1)**2)) * 1e9
        
        # --- Solve Order 0 ---
        x0, _ = mgr_0.solve_optimization(B_bg_flat, S_0, region_mask=region_mask)
        sys0, _ = mgr_0.get_final_system(x0)
        B_res_0 = physics.calculate_field_from_coils(sys0, points_in_sphere, use_shielding=False, show_progress=False) + B_bg_vec[region_mask]
        stats_0 = analyze_components(points_in_sphere, B_res_0)
        rms_0_total = np.sqrt(np.mean(np.linalg.norm(B_res_0, axis=1)**2)) * 1e9
        
        row0 = {'Trial': i, 'Order': '0 (B0)', 'Total_dB': 20*np.log10(rms_bg_total/rms_0_total)}
        for k in stats_bg:
            # Handle dB calculation (avoid log of zero)
            ratio = stats_bg[k] / max(stats_0[k], 1e-6)
            row0[f'{k}_dB'] = 20 * np.log10(max(ratio, 1e-3))
        results.append(row0)
        
        # --- Solve Order 1 ---
        x1, _ = mgr_1.solve_optimization(B_bg_flat, S_1, region_mask=region_mask)
        sys1, _ = mgr_1.get_final_system(x1)
        B_res_1 = physics.calculate_field_from_coils(sys1, points_in_sphere, use_shielding=False, show_progress=False) + B_bg_vec[region_mask]
        stats_1 = analyze_components(points_in_sphere, B_res_1)
        rms_1_total = np.sqrt(np.mean(np.linalg.norm(B_res_1, axis=1)**2)) * 1e9
        
        row1 = {'Trial': i, 'Order': '1 (B0+Grad)', 'Total_dB': 20*np.log10(rms_bg_total/rms_1_total)}
        for k in stats_bg:
            ratio = stats_bg[k] / max(stats_1[k], 1e-6)
            row1[f'{k}_dB'] = 20 * np.log10(max(ratio, 1e-3))
        results.append(row1)
        
        print(f"     [Result] Total Gain: {row1['Total_dB'] - row0['Total_dB']:.2f} dB")

    # 3. Analyze & Visualize
    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Data] Saved raw results to {csv_path}")
    
    # Plotting: Multi-panel Suppression Report
    metrics = ['Total_dB', 'Bx_dB', 'By_dB', 'Bz_dB', 'Gxx_dB', 'Gyy_dB', 'Gxy_dB', 'Gxz_dB', 'Gyz_dB']
    sns.set_theme(style="whitegrid")
    
    # (a) Main Comparison (Total dB)
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x='Order', y='Total_dB', hue='Order', palette='Set1', jitter=True, size=10, legend=False)
    sns.boxplot(data=df, x='Order', y='Total_dB', hue='Order', palette='Set1', showmeans=True, boxprops={'alpha': 0.3})
    plt.title(f"Total Suppression Performance ({NUM_TRIALS} Trials)")
    plt.savefig(output_dir / "benchmark_total_comparison.svg")
    plt.close()

    # (b) Breakdown Comparison (All Components)
    # Melt the dataframe for facet plotting
    df_melted = df.melt(id_vars=['Trial', 'Order'], value_vars=metrics, var_name='Metric', value_name='dB')
    
    g = sns.catplot(data=df_melted, x='Order', y='dB', col='Metric', col_wrap=3, 
                    kind='strip', palette='Set1', hue='Order', jitter=True, alpha=0.6, sharey=False)
    # Map boxplots to each facet. Note: map_dataframe is safer.
    g.map_dataframe(sns.boxplot, x='Order', y='dB', hue='Order', palette='Set1', showmeans=True, boxprops={'alpha': 0.3})
    g.set_titles("{col_name}")
    plt.savefig(output_dir / "benchmark_metrics_breakdown.svg")
    plt.close()

    # Summary
    mean_0 = df[df['Order'] == '0 (B0)']['Total_dB'].mean()
    mean_1 = df[df['Order'] == '1 (B0+Grad)']['Total_dB'].mean()
    print("\n[Summary]")
    print(f"  Average Suppression Order 0: {mean_0:.2f} dB")
    print(f"  Average Suppression Order 1: {mean_1:.2f} dB")
    print(f"  Average Improvement:         {mean_1 - mean_0:.2f} dB")
    print(f"  Results saved to: {output_dir}")

if __name__ == "__main__":
    run_benchmark()
