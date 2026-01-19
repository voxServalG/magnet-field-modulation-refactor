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

class DualLogger:
    '''
    A helper class to redirect stdout to both the console and a file.
    '''
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure real-time logging

    def flush(self):
        # Needed for python 3 compatibility
        self.terminal.flush()
        self.log.flush()

def run_array_demo():
    # Setup Timestamped Output Directory FIRST
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- START LOGGING CAPTURE ---
    log_file = output_dir / "simulation_log.txt"
    sys.stdout = DualLogger(log_file)
    # -----------------------------

    print("========================================================")
    print("   Project Nuke: 3x3 Array Selective Suppression   ")
    print("========================================================")
    print(f"[System] Output directory created: {output_dir}")
    print(f"[System] Log file initiated: {log_file}")
    
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
    
    # 2. Define ROI (3D Volume for Spherical Suppression)
    # -----------------------------------------------------------
    Np = 25 # High density for visualization
    # Covering almost the whole array area
    limit = 2.5
    gx = np.linspace(-limit, limit, Np)
    gy = np.linspace(-limit, limit, Np)
    gz = np.linspace(-limit, limit, Np) # 3D Volume
    
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    target_points = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    # Define TARGET MASK (The "Sphere of Silence")
    # We want to suppress field in a SPHERE at X=1.0, Y=1.0, Z=0.0, Radius=0.6
    center_spot = np.array([1.0, 1.0, 0.0])
    radius_spot = 0.6
    dist_from_spot = np.linalg.norm(target_points - center_spot, axis=1)
    region_mask = dist_from_spot < radius_spot
    
    # 3. Compute System Response
    # --------------------------
    S = manager.compute_response_matrix(target_points)
    
    # 4. Define Background Interference (Randomized Dipole Sources)
    # ---------------------------------------------------------
    print("\n[Sim] Generating Randomized Background Field (Phantom Dipoles)...")
    
    # Random Seed for Reproducibility (Change this to get different fields)
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    print(f"  -> Random Seed: {seed}")

    # Create synthetic "Phantom Dipoles" outside the ROI to generate a smooth but random field
    num_dipoles = 5
    dipole_positions = []
    dipole_moments = []
    
    # ROI is roughly within [-2.5, 2.5], so place dipoles further out
    min_dist = 4.0
    max_dist = 6.0
    
    for _ in range(num_dipoles):
        # Random spherical position
        r = np.random.uniform(min_dist, max_dist)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        
        pos = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        dipole_positions.append(pos)
        
        # Random moment vector (Magnitude ~ 1e4 A*m2 gives ~nT at 5m)
        moment_mag = np.random.uniform(5e4, 2e5) 
        moment_vec = np.random.randn(3)
        moment_vec /= np.linalg.norm(moment_vec) # Normalize direction
        moment_vec *= moment_mag
        dipole_moments.append(moment_vec)
        
    def calculate_dipole_B(points, pos, moment):
        # B = (mu0 / 4pi) * (3(m.r)r - m) / r^5
        # r vector from dipole to point
        mu0_4pi = 1e-7
        R = points - pos
        r_norms = np.linalg.norm(R, axis=1)
        r_norms = r_norms[:, np.newaxis] # Shape (N, 1)
        
        dot_product = np.sum(moment * R, axis=1)[:, np.newaxis] # (m.r)
        
        B = (3 * R * dot_product / r_norms**2 - moment) / r_norms**3
        return mu0_4pi * B

    # Sum up fields
    B_bg_vectors = np.zeros_like(target_points)
    for pos, mom in zip(dipole_positions, dipole_moments):
        B_bg_vectors += calculate_dipole_B(target_points, pos, mom)
        
    # Scale to ensure it's a challenging but realistic test (mean ~100-200 nT)
    current_mean = np.mean(np.linalg.norm(B_bg_vectors, axis=1))
    target_mean = np.random.uniform(100e-9, 200e-9)
    scaling_factor = target_mean / current_mean
    B_bg_vectors *= scaling_factor
    
    print(f"  -> Generated field with mean intensity: {target_mean*1e9:.2f} nT")
    
    B_bg_flat = B_bg_vectors.flatten()
    
    # 5. Dynamic Response Experiment (Real-time Switching)
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("      Dynamic Response & Switching Test")
    print("="*50)
    print(f"[System] Response Matrix S (Shape {S.shape}) Pre-calculated.")
    print("[System] Background Field B_bg Pre-calculated.")
    
    # --- Phase 1: Initial Target (Alpha) ---
    print("\n>>> Phase 1: Target Alpha (Original)")
    center_spot_A = np.array([1.0, 1.0, 0.0])
    radius_spot_A = 0.6
    
    # Fast Mask Generation
    t_p1_start = time.perf_counter()
    dist_A = np.linalg.norm(target_points - center_spot_A, axis=1)
    mask_A = dist_A < radius_spot_A
    
    # Solve
    x_opt_A, _ = manager.solve_optimization(B_bg_flat, S, region_mask=mask_A)
    t_p1_end = time.perf_counter()
    
    duration_A_ms = (t_p1_end - t_p1_start) * 1000.0
    print(f"  -> Target Location: {center_spot_A}")
    print(f"  -> Optimization Time: {duration_A_ms:.4f} ms")
    
    # --- Phase 2: Target Switch (Beta) ---
    print("\n>>> Phase 2: Switching to Target Beta (New Location)")
    center_spot_B = np.array([-1.0, -0.5, 0.5]) 
    radius_spot_B = 0.6 
    
    # Timer starts NOW (Simulating "Enemy moved, System reacting")
    t_p2_start = time.perf_counter()
    
    # 1. Update Logic (Recalculate Mask)
    dist_B = np.linalg.norm(target_points - center_spot_B, axis=1)
    mask_B = dist_B < radius_spot_B
    
    # 2. Solve (Reuse S matrix!)
    x_opt_B, _ = manager.solve_optimization(B_bg_flat, S, region_mask=mask_B)
    
    t_p2_end = time.perf_counter()
    duration_B_ms = (t_p2_end - t_p2_start) * 1000.0
    
    print(f"  -> New Target Location: {center_spot_B}")
    print(f"  -> SYSTEM LATENCY (Switch+Solve): {duration_B_ms:.4f} ms")
    
    # USE PHASE 2 RESULTS FOR VISUALIZATION
    print("\n[System] Proceeding to visualize Phase 2 results...")
    x_opt = x_opt_B
    region_mask = mask_B
    center_spot = center_spot_B
    radius_spot = radius_spot_B
    
    # 6. Final Physics Verification
    # -----------------------------
    print("\n[Sim] Verifying with Full Physics Simulation (Phase 2)...")
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
    
    print(f"\n[Results - Phase 2]")
    print(f"  -> Global ROI RMS [Before]: {rms_bg_global:.2f} nT")
    print(f"  -> Global ROI RMS [After]:  {rms_res_global:.2f} nT (Expect increase due to side effects)")
    print(f"  -> TARGET SPHERE RMS [Before]: {rms_bg_local:.2f} nT")
    print(f"  -> TARGET SPHERE RMS [After]:  {rms_res_local:.2f} nT")
    print(f"  -> TARGET Suppression:       {suppression_local_db:.2f} dB")
    print(f"  -> Optimization Latency:     {duration_B_ms:.2f} ms")
    
    # 7. Comprehensive Visualization (Full 3D Volumetric View - Aspect Ratio Fixed)
    # --------------------------------------------------
    print("\n[Viz] Generating 4 3D Volumetric Images (Fixed Aspect Ratio)...")

    # Shared logic for 3D sphere plots
    sphere_pts = target_points[region_mask]
    bg_sphere = B_bg_vectors[region_mask, 0] * 1e9
    total_sphere = B_total[region_mask, 0] * 1e9
    ratio_sphere = (np.abs(B_total[region_mask, 0]) / np.abs(B_bg_vectors[region_mask, 0])) * 100

    # Define common limits for cubic aspect
    limit_3d = 2.5
    
    # DETERMINE COMMON COLOR SCALE FOR FIG 2 & FIG 3
    # Use the range of the Background Field to scale both.
    # This makes the Total Field (Fig 3) appear visually "darker/suppressed" if successful.
    vmax_common = np.max(bg_sphere)
    vmin_common = np.min(bg_sphere)

    # (a) 3D Array Structure + Target Sphere Wireframe
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    for i, (top, bot, _, _) in enumerate(final_system):
        c = coil_colors[i]
        ax1.plot(top[:,0], top[:,1], top[:,2], color=c, alpha=0.15, linewidth=0.5)
        ax1.plot(bot[:,0], bot[:,1], bot[:,2], color=c, alpha=0.15, linewidth=0.5)
    
    # 3D Wireframe for Sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    sx = center_spot[0] + radius_spot * np.cos(u) * np.sin(v)
    sy = center_spot[1] + radius_spot * np.sin(u) * np.sin(v)
    sz = center_spot[2] + radius_spot * np.cos(v)
    ax1.plot_wireframe(sx, sy, sz, color="lime", alpha=0.2, linewidth=0.5)
    
    ax1.set_title("3D System Setup: Array + Target Sphere")
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.set_xlim(-limit_3d, limit_3d); ax1.set_ylim(-limit_3d, limit_3d); ax1.set_zlim(-limit_3d, limit_3d)
    ax1.set_box_aspect((1, 1, 1)) # CRITICAL: Fixes the ellipsoid distortion
    
    path1 = output_dir / "1_array_3d_setup.svg"
    plt.savefig(path1)
    plt.close(fig1)

    # (b) Background Field (3D Sphere Volume)
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], 
                      c=bg_sphere, cmap='plasma', s=30, alpha=0.8,
                      vmin=vmin_common, vmax=vmax_common) # Explicitly set limits
    plt.colorbar(sc2, ax=ax2, label='Bx (nT)', shrink=0.7)
    ax2.set_title("Initial Background Bx (3D Sphere View)")
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
    
    # Focus view on the sphere itself for better detail
    buf = radius_spot * 0.5
    ax2.set_xlim(center_spot[0]-radius_spot-buf, center_spot[0]+radius_spot+buf)
    ax2.set_ylim(center_spot[1]-radius_spot-buf, center_spot[1]+radius_spot+buf)
    ax2.set_zlim(center_spot[2]-radius_spot-buf, center_spot[2]+radius_spot+buf)
    ax2.set_box_aspect((1, 1, 1))
    
    path2 = output_dir / "2_background_field_3d_sphere.svg"
    plt.savefig(path2)
    plt.close(fig2)

    # (c) Total Field (3D Sphere Volume)
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    # USE COMMON LIMITS to show relative suppression visually
    sc3 = ax3.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], 
                      c=total_sphere, cmap='plasma', s=30, alpha=0.8,
                      vmin=vmin_common, vmax=vmax_common) 
    plt.colorbar(sc3, ax=ax3, label='Bx (nT)', shrink=0.7)
    ax3.set_title("Total Residual Bx (3D Sphere View) [SAME SCALE]")
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
    ax3.set_xlim(ax2.get_xlim()); ax3.set_ylim(ax2.get_ylim()); ax3.set_zlim(ax2.get_zlim())
    ax3.set_box_aspect((1, 1, 1))
    
    path3 = output_dir / "3_total_field_3d_sphere.svg"
    plt.savefig(path3)
    plt.close(fig3)

    # (d) Suppression Ratio (3D Sphere Volume)
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    sc4 = ax4.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], 
                      c=ratio_sphere, cmap='viridis', s=30, alpha=0.8)
    cbar4 = plt.colorbar(sc4, ax=ax4, label='% of Original', shrink=0.7)
    ax4.set_title("Suppression Ratio % (3D Sphere View)")
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)'); ax4.set_zlabel('Z (m)')
    ax4.set_xlim(ax2.get_xlim()); ax4.set_ylim(ax2.get_ylim()); ax4.set_zlim(ax2.get_zlim())
    ax4.set_box_aspect((1, 1, 1))
    
    path4 = output_dir / "4_suppression_ratio_3d_sphere.svg"
    plt.savefig(path4)
    plt.close(fig4)
    
    # (e) Coil Breakdown (3 Subplots: Bx, By, Bz)
    print("  -> Generating Coil Breakdown Figure...")
    fig5 = plt.figure(figsize=(18, 6))
    
    # Subplot 1: Bx Coils (Red)
    ax5_1 = fig5.add_subplot(131, projection='3d')
    count_bx = 0
    for i, (top, bot, _, _) in enumerate(final_system):
        if coil_colors[i] == 'r':
            ax5_1.plot(top[:,0], top[:,1], top[:,2], color='r', alpha=0.5, linewidth=0.8)
            ax5_1.plot(bot[:,0], bot[:,1], bot[:,2], color='r', alpha=0.5, linewidth=0.8)
            count_bx += 1
    ax5_1.set_title(f"Bx Coils (Count: {count_bx})")
    
    # Subplot 2: By Coils (Green)
    ax5_2 = fig5.add_subplot(132, projection='3d')
    count_by = 0
    for i, (top, bot, _, _) in enumerate(final_system):
        if coil_colors[i] == 'g':
            ax5_2.plot(top[:,0], top[:,1], top[:,2], color='g', alpha=0.5, linewidth=0.8)
            ax5_2.plot(bot[:,0], bot[:,1], bot[:,2], color='g', alpha=0.5, linewidth=0.8)
            count_by += 1
    ax5_2.set_title(f"By Coils (Count: {count_by})")

    # Subplot 3: Bz Coils (Blue)
    ax5_3 = fig5.add_subplot(133, projection='3d')
    count_bz = 0
    for i, (top, bot, _, _) in enumerate(final_system):
        if coil_colors[i] == 'b':
            ax5_3.plot(top[:,0], top[:,1], top[:,2], color='b', alpha=0.5, linewidth=0.8)
            ax5_3.plot(bot[:,0], bot[:,1], bot[:,2], color='b', alpha=0.5, linewidth=0.8)
            count_bz += 1
    ax5_3.set_title(f"Bz Coils (Count: {count_bz})")
    
    # Common settings for all subplots
    for ax in [ax5_1, ax5_2, ax5_3]:
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(-limit_3d, limit_3d)
        ax.set_ylim(-limit_3d, limit_3d)
        ax.set_zlim(-limit_3d, limit_3d)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=30, azim=45)

    path5 = output_dir / "5_coils_breakdown.svg"
    plt.savefig(path5, bbox_inches='tight')
    plt.close(fig5)
    print(f"  -> Saved: {path5}")

    # (f) Vector Field Comparison (3D Arrows: Before vs After)
    print("  -> Generating Vector Field Comparison...")
    fig6 = plt.figure(figsize=(16, 8))
    
    # Common Data
    pts_vec = sphere_pts # Points inside sphere
    
    # Vectors & Magnitudes
    vec_bg = B_bg_vectors[region_mask] * 1e9
    mag_bg = np.linalg.norm(vec_bg, axis=1)
    
    vec_res = B_total[region_mask] * 1e9
    mag_res = np.linalg.norm(vec_res, axis=1)
    
    # Use the max magnitude of the BACKGROUND field to set the color scale for BOTH plots
    # This ensures the visual comparison is valid (After plot should look "dimmer" or cooler color)
    v_max = np.max(mag_bg)
    v_min = 0 # or np.min(mag_res) if we want to span full range, but 0 anchors it better
    
    # Subsampling for cleaner arrows if too dense
    if len(pts_vec) > 500:
        skip = int(len(pts_vec) / 300) # Target ~300 arrows
        idx_arrow = np.arange(0, len(pts_vec), skip)
    else:
        idx_arrow = np.arange(len(pts_vec))
        
    def plot_quiver(ax, pts, vecs, mags, title):
        # Normalize vectors for direction-only arrows (color shows magnitude)
        # Avoid division by zero
        mags_safe = np.where(mags == 0, 1, mags)
        u = vecs[:, 0] / mags_safe
        v = vecs[:, 1] / mags_safe
        w = vecs[:, 2] / mags_safe
        
        # Plot
        q = ax.quiver(pts[:,0], pts[:,1], pts[:,2], 
                      u, v, w, 
                      length=0.15, # Arrow length (visual units)
                      normalize=True,
                      cmap='plasma', 
                      array=mags, 
                      pivot='middle')
        q.set_clim(v_min, v_max) # Lock color scale
        
        ax.set_title(title)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        # Focus on sphere
        buf = radius_spot * 0.5
        ax.set_xlim(center_spot[0]-radius_spot-buf, center_spot[0]+radius_spot+buf)
        ax.set_ylim(center_spot[1]-radius_spot-buf, center_spot[1]+radius_spot+buf)
        ax.set_zlim(center_spot[2]-radius_spot-buf, center_spot[2]+radius_spot+buf)
        ax.set_box_aspect((1, 1, 1))
        return q

    # Plot 1: Before
    ax6_1 = fig6.add_subplot(121, projection='3d')
    q1 = plot_quiver(ax6_1, pts_vec[idx_arrow], vec_bg[idx_arrow], mag_bg[idx_arrow], "Before Suppression (Background)")
    
    # Plot 2: After
    ax6_2 = fig6.add_subplot(122, projection='3d')
    q2 = plot_quiver(ax6_2, pts_vec[idx_arrow], vec_res[idx_arrow], mag_res[idx_arrow], "After Suppression (Residual)")
    
    # Colorbar (Shared)
    cbar = fig6.colorbar(q1, ax=[ax6_1, ax6_2], shrink=0.6, pad=0.05)
    cbar.set_label('|B| (nT)')
    
    path6 = output_dir / "6_field_vectors_comparison.svg"
    plt.savefig(path6, bbox_inches='tight')
    plt.close(fig6)
    print(f"  -> Saved: {path6}")

    print(f"  -> All 3D volumetric reports saved to: {output_dir}")
    print("\n[Done] Active Shielding Mission Accomplished.")

if __name__ == "__main__":
    run_array_demo()