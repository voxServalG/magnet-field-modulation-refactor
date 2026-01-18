import numpy as np
import pathlib
import matplotlib.pyplot as plt
import sys
import time

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src.factory import CoilFactory
from src.array_manager import ArrayActiveShielding
from src import physics, visuals

def run_array_demo():
    print("========================================================")
    print("   Project Nuke: 3x3 Array Selective Suppression   ")
    print("========================================================")
    
    # Setup Timestamped Output Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[System] Output directory created: {output_dir}")
    
    # 1. Setup Environment
    # --------------------
    config = {
        'L': 0.85,
        'a': 0.7,
        'modes': (4, 4),
        'reg_lambda': 1e-14,
        'num_turns': 22,
        'grid_res': 80 # Fast grid for demo
    }
    factory = CoilFactory(config)
    
    # Define Array Layout: 3x3 Grid
    # Unit Width = 1.7m. Spacing = 1.8m (Slight gap)
    s = 1.8
    offsets = [-s, 0, s]
    layout_list = []
    for x_off in offsets:
        for y_off in offsets:
            layout_list.append([x_off, y_off, 0.0])
    layout = np.array(layout_list)
    
    manager = ArrayActiveShielding(factory, layout)
    
    # 2. Define ROI (Large area to see the "mess" outside target)
    # -----------------------------------------------------------
    Np = 25 # High density for visualization
    # Covering almost the whole array area
    limit = 2.5
    gx = np.linspace(-limit, limit, Np)
    gy = np.linspace(-limit, limit, Np)
    gz = np.array([0.0])
    
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    target_points = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    # Define TARGET MASK (The "Spotlight")
    # We want to suppress field only in a circle at X=1.0, Y=1.0, Radius=0.6
    center_spot = np.array([1.0, 1.0])
    radius_spot = 0.6
    dist_from_spot = np.linalg.norm(target_points[:, :2] - center_spot, axis=1)
    region_mask = dist_from_spot < radius_spot
    
    # 3. Compute System Response
    # --------------------------
    S = manager.compute_response_matrix(target_points)
    
    # 4. Define Background Interference (Enemy)
    # -----------------------------------------
    print("\n[Sim] Generating Background Interference (Complex Gradient)...")
    B_bg_vectors = np.zeros_like(target_points)
    # Bx = 100nT + 40nT*x + 30nT*y
    B_bg_vectors[:, 0] = 100e-9 + 40e-9 * target_points[:, 0] + 30e-9 * target_points[:, 1]
    # By = 50nT*y
    B_bg_vectors[:, 1] = 50e-9 * target_points[:, 1]
    # Bz = 20nT
    B_bg_vectors[:, 2] = 20e-9
    
    B_bg_flat = B_bg_vectors.flatten()
    
    # 5. Solve Optimization (REGIONAL!)
    # ---------------------------------
    x_opt, _ = manager.solve_optimization(B_bg_flat, S, region_mask=region_mask)
    
    # 6. Final Physics Verification
    # -----------------------------
    print("\n[Sim] Verifying with Full Physics Simulation...")
    final_system, coil_colors = manager.get_final_system(x_opt)
    
    B_array = physics.calculate_field_from_coils(final_system, target_points, use_shielding=False, show_progress=True)
    B_total = B_array + B_bg_vectors
    
    # Stats
    B_bg_nt = np.linalg.norm(B_bg_vectors, axis=1) * 1e9
    B_res_nt = np.linalg.norm(B_total, axis=1) * 1e9
    
    rms_bg_global = np.sqrt(np.mean(B_bg_nt**2))
    rms_res_global = np.sqrt(np.mean(B_res_nt**2))
    
    # Local stats (Target Zone)
    rms_bg_local = np.sqrt(np.mean(B_bg_nt[region_mask]**2))
    rms_res_local = np.sqrt(np.mean(B_res_nt[region_mask]**2))
    suppression_local_db = 20 * np.log10(rms_bg_local / rms_res_local)
    
    print(f"\n[Results]")
    print(f"  -> Global ROI RMS [Before]: {rms_bg_global:.2f} nT")
    print(f"  -> Global ROI RMS [After]:  {rms_res_global:.2f} nT (Expect increase due to side effects)")
    print(f"  -> TARGET ZONE RMS [Before]: {rms_bg_local:.2f} nT")
    print(f"  -> TARGET ZONE RMS [After]:  {rms_res_local:.2f} nT")
    print(f"  -> TARGET Suppression:       {suppression_local_db:.2f} dB")
    
    # 7. Comprehensive Visualization (Focused on Target Zone)
    # --------------------------------------------------
    print("\n[Viz] Generating 4 Focused Images...")

    # (a) 3D Array Structure + Target Marker
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    for i, (top, bot, _, _) in enumerate(final_system):
        c = coil_colors[i]
        ax1.plot(top[:,0], top[:,1], top[:,2], color=c, alpha=0.2, linewidth=0.5)
        ax1.plot(bot[:,0], bot[:,1], bot[:,2], color=c, alpha=0.2, linewidth=0.5)
    
    # Add a visual indicator for the target ROI (Circle in 3D)
    theta = np.linspace(0, 2*np.pi, 100)
    cx = center_spot[0] + radius_spot * np.cos(theta)
    cy = center_spot[1] + radius_spot * np.sin(theta)
    cz = np.zeros_like(cx)
    ax1.plot(cx, cy, cz, color='lime', linewidth=3, label='Target ROI')
    ax1.set_title("3D Array Structure & Target Region")
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.legend()
    
    path1 = output_dir / "1_array_structure_with_target.png"
    plt.savefig(path1, dpi=200)
    plt.close(fig1)
    print(f"  -> Saved: {path1}")

    # (b) Background Field (Target Zone Only)
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    sc2 = ax2.tricontourf(target_points[region_mask,0], target_points[region_mask,1], 
                         B_bg_vectors[region_mask,0]*1e9, levels=20, cmap='plasma')
    ax2.set_title("Background Field (Bx) @ Target Zone [nT]")
    plt.colorbar(sc2, ax=ax2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    
    path2 = output_dir / "2_background_field_target.png"
    plt.savefig(path2, dpi=200)
    plt.close(fig2)
    print(f"  -> Saved: {path2}")

    # (c) Total Field (Target Zone Only)
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    sc3 = ax3.tricontourf(target_points[region_mask,0], target_points[region_mask,1], 
                         B_total[region_mask,0]*1e9, levels=20, cmap='plasma')
    ax3.set_title("Total Field (Bx) @ Target Zone [nT]")
    plt.colorbar(sc3, ax=ax3)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
    
    path3 = output_dir / "3_total_field_target.png"
    plt.savefig(path3, dpi=200)
    plt.close(fig3)
    print(f"  -> Saved: {path3}")

    # (d) Suppression Ratio or Comparison
    fig4 = plt.figure(figsize=(8, 6))
    ax4 = fig4.add_subplot(111)
    # Calculate local residual as percentage of background
    residual_pct = (np.abs(B_total[region_mask, 0]) / np.abs(B_bg_vectors[region_mask, 0])) * 100
    sc4 = ax4.tricontourf(target_points[region_mask, 0], target_points[region_mask, 1], 
                         residual_pct, levels=20, cmap='viridis')
    ax4.set_title("Residual Field Ratio (%) - Lower is Better")
    cbar4 = plt.colorbar(sc4, ax=ax4)
    cbar4.set_label('% of Original Field')
    ax4.set_aspect('equal')
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)')
    
    path4 = output_dir / "4_suppression_ratio.png"
    plt.savefig(path4, dpi=200)
    plt.close(fig4)
    print(f"  -> Saved: {path4}")
    
    print("\n[Done] Active Shielding Mission Accomplished.")

if __name__ == "__main__":
    run_array_demo()