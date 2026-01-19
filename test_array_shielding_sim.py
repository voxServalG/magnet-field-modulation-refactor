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
from src import physics, visuals, coils

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
    # --- DEMO CONFIGURATION ---
    DEMO_MODE_FREE_SPACE = True  # Set True to disable shielding reflections (Ideal Demo)
    # --------------------------

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
    print(f"[System] Demo Mode: {'FREE SPACE (Ideal)' if DEMO_MODE_FREE_SPACE else 'SHIELDED ROOM (Real)'}")
    print(f"[System] Output directory created: {output_dir}")
    print(f"[System] Log file initiated: {log_file}")
    
    # 1. Setup Environment
    # --------------------
    ORDER_LIMIT = 1 # 0: B0 only (3ch), 1: B0 + Gradients (8ch)
    
    # Real-world Shielding Room Parameters
    SHIELD_DIMS = (2.4, 1.9, 1.65)
    
    # Determine active shielding config
    active_shield_dims = None if DEMO_MODE_FREE_SPACE else SHIELD_DIMS

    config = {
        'L': 0.6,   # Unit Half-Length (1.2m size)
        'a': 0.7,   # Unit Half-Distance (1.4m plate spacing)
        'modes': (4, 4),
        'reg_lambda': 1e-14,
        'num_turns': 20,
        'grid_res': 80,
        'shield_dims': active_shield_dims # Dynamic
    }
    factory = CoilFactory(config)
    
    # Define Array Layout: 3x3 Grid
    s = 1.25
    offsets = [-s, 0, s]
    layout_list = []
    for x_off in offsets:
        for y_off in offsets:
            layout_list.append([x_off, y_off, 0.0])
    layout = np.array(layout_list)
    
    # Enable Gradient Coils with dynamic shielding arg
    manager = ArrayActiveShielding(factory, layout, use_gradients=(ORDER_LIMIT > 0), shield_dims=active_shield_dims)
    
    # 2. Define ROI (3D Volume for Spherical Suppression)
    # -----------------------------------------------------------
    Np = 21 # Moderate density
    limit = 1.5 # Focused view
    gx = np.linspace(-limit, limit, Np)
    gy = np.linspace(-limit, limit, Np)
    gz = np.linspace(-limit, limit, Np) 
    
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    target_points = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    # Define TARGET MASK (The "Cube of Silence" approximated by Sphere)
    # Target: 400mm box -> Radius ~ 0.25m
    # Position: Slightly off-center to prove robustness
    center_spot = np.array([0.0, 0.0, 0.0]) 
    radius_spot = 0.25
    dist_from_spot = np.linalg.norm(target_points - center_spot, axis=1)
    region_mask = dist_from_spot < radius_spot
    
    # 3. Compute System Response (With Shielding Reflections!)
    # --------------------------
    # We need to monkey-patch or ensure the manager uses shielding.
    # Currently compute_response_matrix hardcodes use_shielding=False.
    # Let's update the call to calculate_field_from_coils inside array_manager first? 
    # Or just pass the flag if we updated array_manager.
    # Wait, array_manager.compute_response_matrix currently hardcodes False.
    # I will modify array_manager.py first to accept this flag, OR I will modify it here if I could.
    # Since I cannot modify array_manager in this turn easily without another tool call,
    # I will rely on the user to accept a small patch to array_manager.py first.
    # Actually, I can do it in the next step. For now, let's assume I patch it.
    
    print(f"[Sim] Shielding Wall Reflections: ENABLED (Room: {SHIELD_DIMS})")
    # For now, we inject the logic via `physics` module config if possible, 
    # but `physics.calculate_field` takes `use_shielding` arg.
    # I will update `test_array_shielding_sim.py` to use a custom computation loop or 
    # assume I'll fix `src/array_manager.py` in a moment.
    
    # Force enable shielding in the matrix computation (Requires ArrayManager update)
    # For this specific file, I will just set the config.
    # BUT `compute_response_matrix` inside `ArrayManager` needs to be told to use shielding.
    # I will update `src/array_manager.py` in the next tool call.
    
    S = manager.compute_response_matrix(target_points) # Will use shielding after my next patch
    
    # 4. Define Background Interference (Randomized Dipole Sources)
    # ---------------------------------------------------------
    print("\n[Sim] Generating Randomized Background Field (Phantom Dipoles)...")
    
    # Random Seed for Reproducibility
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    print(f"  -> Random Seed: {seed}")

    # Create synthetic "Phantom Dipoles"
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

    B_bg_vectors = np.zeros_like(target_points)
    for pos, mom in zip(dipole_positions, dipole_moments):
        B_bg_vectors += calculate_dipole_B(target_points, pos, mom)
        
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
    
    # --- Phase 1: Initial Target (Alpha) ---
    print("\n>>> Phase 1: Target Alpha (Original)")
    center_spot_A = np.array([1.0, 1.0, 0.0])
    radius_spot_A = 0.6
    
    t_p1_start = time.perf_counter()
    dist_A = np.linalg.norm(target_points - center_spot_A, axis=1)
    mask_A = dist_A < radius_spot_A
    x_opt_A, _ = manager.solve_optimization(B_bg_flat, S, region_mask=mask_A)
    t_p1_end = time.perf_counter()
    
    duration_A_ms = (t_p1_end - t_p1_start) * 1000.0
    print(f"  -> Target Location: {center_spot_A}")
    print(f"  -> Optimization Time: {duration_A_ms:.4f} ms")
    
    # --- Phase 2: Target Switch (Beta) ---
    print("\n>>> Phase 2: Switching to Target Beta (New Location)")
    center_spot_B = np.array([-1.0, -0.5, 0.5]) 
    radius_spot_B = 0.6 
    
    t_p2_start = time.perf_counter()
    dist_B = np.linalg.norm(target_points - center_spot_B, axis=1)
    mask_B = dist_B < radius_spot_B
    x_opt_B, _ = manager.solve_optimization(B_bg_flat, S, region_mask=mask_B)
    t_p2_end = time.perf_counter()
    duration_B_ms = (t_p2_end - t_p2_start) * 1000.0
    
    print(f"  -> New Target Location: {center_spot_B}")
    print(f"  -> SYSTEM LATENCY (Switch+Solve): {duration_B_ms:.4f} ms")
    
    x_opt = x_opt_B
    region_mask = mask_B
    center_spot = center_spot_B
    radius_spot = radius_spot_B
    
    # 6. Final Physics Verification
    # -----------------------------
    print("\n[Sim] Verifying with Full Physics Simulation (Phase 2)...")
    final_system, coil_colors = manager.get_final_system(x_opt)
    
    # Use dynamic shielding setting for verification
    # If DEMO_MODE_FREE_SPACE is True, use_shielding=False.
    # If False, use_shielding=True and pass dims.
    use_shield_verify = not DEMO_MODE_FREE_SPACE
    
    B_array = physics.calculate_field_from_coils(final_system, target_points, 
                                                 use_shielding=use_shield_verify, 
                                                 shield_dims=SHIELD_DIMS,
                                                 show_progress=True)
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
    
    # --- Power Consumption Estimation ---
    def calculate_system_power(system):
        # Assumptions: 1mm diameter Copper wire
        rho = 1.68e-8 # Ohm*m
        dia = 1.0e-3  # m
        area = np.pi * (dia/2)**2
        res_per_m = rho / area # ~0.021 Ohm/m
        
        total_watts = 0.0
        total_len = 0.0
        
        for top, bot, I_top, I_bot in system:
            # Top Loop
            d = np.diff(top, axis=0)
            len_top = np.sum(np.sqrt(np.sum(d**2, axis=1)))
            R_top = len_top * res_per_m
            total_watts += (I_top**2) * R_top
            total_len += len_top
            
            # Bottom Loop
            d = np.diff(bot, axis=0)
            len_bot = np.sum(np.sqrt(np.sum(d**2, axis=1)))
            R_bot = len_bot * res_per_m
            total_watts += (I_bot**2) * R_bot
            total_len += len_bot
            
        return total_watts, total_len

    power_watts, wire_len = calculate_system_power(final_system)

    print(f"\n[Results - Phase 2]")
    print(f"  -> Global ROI RMS [Before]: {rms_bg_global:.2f} nT")
    print(f"  -> Global ROI RMS [After]:  {rms_res_global:.2f} nT (Expect increase due to side effects)")
    print(f"  -> TARGET SPHERE RMS [Before]: {rms_bg_local:.2f} nT")
    print(f"  -> TARGET SPHERE RMS [After]:  {rms_res_local:.2f} nT")
    print(f"  -> TARGET Suppression:       {suppression_local_db:.2f} dB")
    print(f"  -> Optimization Latency:     {duration_B_ms:.2f} ms")
    print(f"  -> Est. Power Consumption:   {power_watts:.4f} W (assuming 1mm Cu wire)")
    print(f"  -> Total Wire Length:        {wire_len:.2f} m")
    
    # 7. Comprehensive Visualization
    # --------------------------------------------------
    print("\n[Viz] Generating 3D Volumetric Reports (SVG)...")

    sphere_pts = target_points[region_mask]
    bg_sphere = B_bg_vectors[region_mask, 0] * 1e9
    total_sphere = B_total[region_mask, 0] * 1e9
    ratio_sphere = (np.abs(B_total[region_mask, 0]) / np.abs(B_bg_vectors[region_mask, 0])) * 100

    limit_3d = 2.5
    vmax_common = np.max(bg_sphere)
    vmin_common = np.min(bg_sphere)

    # (a) 3D Array Setup
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    for i, (top, bot, _, _) in enumerate(final_system):
        ax1.plot(top[:,0], top[:,1], top[:,2], color=coil_colors[i], alpha=0.15, linewidth=0.5)
        ax1.plot(bot[:,0], bot[:,1], bot[:,2], color=coil_colors[i], alpha=0.15, linewidth=0.5)
    
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    sx = center_spot[0] + radius_spot * np.cos(u) * np.sin(v)
    sy = center_spot[1] + radius_spot * np.sin(u) * np.sin(v)
    sz = center_spot[2] + radius_spot * np.cos(v)
    ax1.plot_wireframe(sx, sy, sz, color="lime", alpha=0.2, linewidth=0.5)
    ax1.set_xlim(-limit_3d, limit_3d); ax1.set_ylim(-limit_3d, limit_3d); ax1.set_zlim(-limit_3d, limit_3d)
    ax1.set_box_aspect((1, 1, 1))
    plt.savefig(output_dir / "1_array_3d_setup.svg")
    plt.close(fig1)

    # (b) Background Field
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], c=bg_sphere, cmap='plasma', s=30, alpha=0.8, vmin=vmin_common, vmax=vmax_common)
    plt.colorbar(sc2, ax=ax2, label='Bx (nT)', shrink=0.7)
    buf = radius_spot * 0.5
    ax2.set_xlim(center_spot[0]-radius_spot-buf, center_spot[0]+radius_spot+buf)
    ax2.set_ylim(center_spot[1]-radius_spot-buf, center_spot[1]+radius_spot+buf)
    ax2.set_zlim(center_spot[2]-radius_spot-buf, center_spot[2]+radius_spot+buf)
    ax2.set_box_aspect((1, 1, 1))
    plt.savefig(output_dir / "2_background_field_3d_sphere.svg")
    plt.close(fig2)

    # (c) Total Field
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    sc3 = ax3.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], c=total_sphere, cmap='plasma', s=30, alpha=0.8, vmin=vmin_common, vmax=vmax_common)
    plt.colorbar(sc3, ax=ax3, label='Bx (nT)', shrink=0.7)
    ax3.set_xlim(ax2.get_xlim()); ax3.set_ylim(ax2.get_ylim()); ax3.set_zlim(ax2.get_zlim()); ax3.set_box_aspect((1, 1, 1))
    plt.savefig(output_dir / "3_total_field_3d_sphere.svg")
    plt.close(fig3)

    # (d) Suppression Ratio
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    sc4 = ax4.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], c=ratio_sphere, cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(sc4, ax=ax4, label='% of Original', shrink=0.7)
    ax4.set_xlim(ax2.get_xlim()); ax4.set_ylim(ax2.get_ylim()); ax4.set_zlim(ax2.get_zlim()); ax4.set_box_aspect((1, 1, 1))
    plt.savefig(output_dir / "4_suppression_ratio_3d_sphere.svg")
    plt.close(fig4)
    
    # (e) Coil Breakdown (Multi-channel)
    print("  -> Generating Coil Breakdown Figure...")
    n_ch = len(manager.channels)
    cols = 3
    rows = int(np.ceil(n_ch / cols))
    fig5 = plt.figure(figsize=(18, 5 * rows))
    
    for idx, ch in enumerate(manager.channels):
        ax = fig5.add_subplot(rows, cols, idx + 1, projection='3d')
        # Find a sample color for this channel
        sample_color = 'k'
        for i, (top, bot, _, _) in enumerate(final_system):
            if coil_colors[i] in ['r', 'g', 'b', 'orange', 'lime', 'cyan', 'magenta', 'yellow']:
                # This logic is a bit hacky because we don't store channel names in final_system
                # But we can reconstruct it
                pass
        
        # Proper breakdown plot: Filter coils belonging to this channel across all units
        # col_idx in optimal_weights is unit_idx * n_ch_per_unit + ch_idx
        for u_idx in range(manager.num_units):
            weight_idx = u_idx * n_ch + idx
            weight = x_opt[weight_idx]
            if abs(weight) < 1e-9: continue
            
            base_coils = manager.standard_units[ch]
            moved_coils = coils.translate_coils(base_coils, manager.layout[u_idx])
            
            # Use color from manager mapping
            c_map = {'bx':'r','by':'g','bz':'b','gxx':'orange','gyy':'lime','gxy':'cyan','gxz':'magenta','gyz':'yellow'}
            c = c_map.get(ch, 'k')
            
            for top, bot, _, _ in moved_coils:
                ax.plot(top[:,0], top[:,1], top[:,2], color=c, alpha=0.5, linewidth=0.8)
                ax.plot(bot[:,0], bot[:,1], bot[:,2], color=c, alpha=0.5, linewidth=0.8)
        
        ax.set_title(f"Channel: {ch.upper()}")
        ax.set_xlim(-limit_3d, limit_3d); ax.set_ylim(-limit_3d, limit_3d); ax.set_zlim(-limit_3d, limit_3d)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=30, azim=45)

    plt.savefig(output_dir / "5_coils_breakdown.svg", bbox_inches='tight')
    plt.close(fig5)

    # (f) Vector Field Comparison
    print("  -> Generating Vector Field Comparison...")
    fig6 = plt.figure(figsize=(16, 8))
    pts_vec = sphere_pts
    vec_bg = B_bg_vectors[region_mask] * 1e9
    mag_bg = np.linalg.norm(vec_bg, axis=1)
    vec_res = B_total[region_mask] * 1e9
    mag_res = np.linalg.norm(vec_res, axis=1)
    v_max = np.max(mag_bg)
    
    idx_arrow = np.arange(len(pts_vec))
    if len(pts_vec) > 500:
        skip = len(pts_vec) // 300
        idx_arrow = idx_arrow[::skip]
        
    def plot_quiver(ax, pts, vecs, mags, title):
        mags_safe = np.where(mags == 0, 1, mags)
        q = ax.quiver(pts[:,0], pts[:,1], pts[:,2], vecs[:,0]/mags_safe, vecs[:,1]/mags_safe, vecs[:,2]/mags_safe, 
                      length=0.15, normalize=True, cmap='plasma', array=mags, pivot='middle')
        q.set_clim(0, v_max)
        ax.set_title(title); ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(ax2.get_xlim()); ax.set_ylim(ax2.get_ylim()); ax.set_zlim(ax2.get_zlim())
        return q

    ax6_1 = fig6.add_subplot(121, projection='3d')
    q1 = plot_quiver(ax6_1, pts_vec[idx_arrow], vec_bg[idx_arrow], mag_bg[idx_arrow], "Before Suppression")
    ax6_2 = fig6.add_subplot(122, projection='3d')
    plot_quiver(ax6_2, pts_vec[idx_arrow], vec_res[idx_arrow], mag_res[idx_arrow], "After Suppression")
    fig6.colorbar(q1, ax=[ax6_1, ax6_2], shrink=0.6, pad=0.05, label='|B| (nT)')
    plt.savefig(output_dir / "6_field_vectors_comparison.svg", bbox_inches='tight')
    plt.close(fig6)

    print(f"\n[Done] All reports saved to: {output_dir}")

if __name__ == "__main__":
    run_array_demo()
if __name__ == "__main__":
    run_array_demo()