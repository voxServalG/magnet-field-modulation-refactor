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
    DEMO_MODE_FREE_SPACE = False  # Set False to ENABLE Shielding Room (Real World)
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
    print(f"[System] Demo Mode: {'FREE SPACE (Ideal)' if DEMO_MODE_FREE_SPACE else 'SHIELDED ROOM (Real MSR)'}")
    print(f"[System] Output directory created: {output_dir}")
    print(f"[System] Log file initiated: {log_file}")
    
    # 1. Setup Environment
    # --------------------
    ORDER_LIMIT = 1 # 0: B0 only (3ch), 1: B0 + Gradients (8ch)
    ARRAY_REG_LAMBDA = 2.3e-16 # The 'Sweet Spot' discovered via parameter scan
    
    # Real-world Shielding Room Parameters
    # Room: 4.8m x 3.8m x 3.3m -> Half-dims: 2.4, 1.9, 1.65
    SHIELD_DIMS = (2.4, 1.9, 1.65)
    
    # Determine active shielding config
    active_shield_dims = None if DEMO_MODE_FREE_SPACE else SHIELD_DIMS

    config = {
        'L': 0.6,   # Unit Half-Length (1.2m size)
        'a': 0.7,   # Unit Half-Distance (1.4m plate spacing)
        'modes': (4, 4),
        'reg_lambda': 1e-14,
        'num_turns': 10, # Reduced from 20 to 10: 'Taking a cut' to save wire length
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
    # Now ArrayManager handles use_shielding internally based on init arg.
    
    if not DEMO_MODE_FREE_SPACE:
        print(f"[Sim] Shielding Wall Reflections: ENABLED (Room: {SHIELD_DIMS})")
        print("  -> NOTE: S-matrix computation will include mirror sources (Positive Images for MuMetal).")
    else:
        print(f"[Sim] Shielding Wall Reflections: DISABLED (Free Space Mode)")
    
    S = manager.compute_response_matrix(target_points) 
    
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
    
    # Phase 1
    x_opt_A, _ = manager.solve_optimization(B_bg_flat, S, region_mask=mask_A, reg_lambda=ARRAY_REG_LAMBDA)
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
    x_opt_B, _ = manager.solve_optimization(B_bg_flat, S, region_mask=mask_B, reg_lambda=ARRAY_REG_LAMBDA)
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
        # Assumptions: 4 sq mm (diameter 2.26mm) Copper wire
        rho = 1.68e-8 # Ohm*m
        dia = 2.26e-3  # m (2.26mm)
        area = np.pi * (dia/2)**2
        res_per_m = rho / area 
        
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
    print(f"  -> Est. Power Consumption:   {power_watts:.4f} W (assuming 2.26mm/4mm^2 Cu wire)")
    print(f"  -> Total Wire Length:        {wire_len:.2f} m")
    
    # 7. Comprehensive Visualization (Publication Quality)
    # --------------------------------------------------
    print("\n[Viz] Generating Publication-Ready Figures (PDF + SVG)...")

    # Style settings for paper
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    sphere_pts = target_points[region_mask]
    bg_sphere = B_bg_vectors[region_mask, 0] * 1e9
    total_sphere = B_total[region_mask, 0] * 1e9
    ratio_sphere = (np.abs(B_total[region_mask, 0]) / np.abs(B_bg_vectors[region_mask, 0])) * 100

    limit_3d = 2.5
    vmax_common = np.max(bg_sphere)
    vmin_common = np.min(bg_sphere)

    # Helper to save both formats
    def save_fig(fig, name):
        fig.savefig(output_dir / f"{name}.svg", bbox_inches='tight')
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches='tight') # Best for LaTeX
        print(f"  -> Saved {name}.pdf/svg")

    # (a) 3D Array Setup with Shielding Room Context
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # 1. Draw Shielding Room (MSR) Wireframe
    sx, sy, sz = SHIELD_DIMS
    # Define corners of the room
    # Floor loop
    ax1.plot([-sx, sx, sx, -sx, -sx], [-sy, -sy, sy, sy, -sy], [-sz, -sz, -sz, -sz, -sz], 'k-', alpha=0.3, label='MSR Walls')
    # Ceiling loop
    ax1.plot([-sx, sx, sx, -sx, -sx], [-sy, -sy, sy, sy, -sy], [sz, sz, sz, sz, sz], 'k-', alpha=0.3)
    # Pillars
    for x in [-sx, sx]:
        for y in [-sy, sy]:
            ax1.plot([x, x], [y, y], [-sz, sz], 'k--', alpha=0.2)

    # 2. Draw Coils
    for i, (top, bot, _, _) in enumerate(final_system):
        ax1.plot(top[:,0], top[:,1], top[:,2], color=coil_colors[i], alpha=0.2, linewidth=0.8)
        ax1.plot(bot[:,0], bot[:,1], bot[:,2], color=coil_colors[i], alpha=0.2, linewidth=0.8)
    
    # 3. Draw Target Sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    tx = center_spot[0] + radius_spot * np.cos(u) * np.sin(v)
    ty = center_spot[1] + radius_spot * np.sin(u) * np.sin(v)
    tz = center_spot[2] + radius_spot * np.cos(v)
    ax1.plot_wireframe(tx, ty, tz, color="lime", alpha=0.4, linewidth=1.0, label="Target ROI")
    
    ax1.set_title("System Setup")
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    # Set limits to room size + margin
    margin = 0.5
    ax1.set_xlim(-sx-margin, sx+margin)
    ax1.set_ylim(-sy-margin, sy+margin)
    ax1.set_zlim(-sz-margin, sz+margin)
    ax1.set_box_aspect((sx, sy, sz)) 
    ax1.legend()
    ax1.view_init(elev=25, azim=135) # Better angle
    
    save_fig(fig1, "1_array_3d_setup")
    plt.close(fig1)

    # (a.2) Artistic Concept Render (Minimalist - Coils Only)
    # ------------------------------------------
    print("  -> Generating Artistic Concept Render (Coils Only)...")
    fig1b = plt.figure(figsize=(10, 8))
    ax1b = fig1b.add_subplot(111, projection='3d')
    
    # 2. Draw Coils (The "Shield")
    # Draw all coils in a unified dark color to emphasize structure
    # Plot Bottom Coils first, then Top Coils? Matplotlib z-order is tricky.
    # We rely on painter's algorithm somewhat.
    
    for i, (top, bot, _, _) in enumerate(final_system):
        # Draw all coils with uniform style
        # Dark grey, slightly transparent
        col_style = 'k' 
        width = 0.8
        alp = 0.4
        
        ax1b.plot(top[:,0], top[:,1], top[:,2], color=col_style, alpha=alp, linewidth=width)
        ax1b.plot(bot[:,0], bot[:,1], bot[:,2], color=col_style, alpha=alp, linewidth=width)
        
    # 3. Add schematic plane indicators (optional) - Removed for pure coil view

    # 4. Cleanup
    ax1b.set_box_aspect((1, 1, 0.6)) # Compress Z slightly to look more like a slab
    ax1b.axis('off') # Remove axis completely
    
    # View angle: Isometric-ish
    ax1b.view_init(elev=20, azim=-45)
    
    save_fig(fig1b, "1b_concept_render")
    plt.close(fig1b)

    # (b) Background Field
    fig2 = plt.figure(figsize=(8, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], c=bg_sphere, cmap='plasma', s=40, alpha=0.9, vmin=vmin_common, vmax=vmax_common)
    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('Bx Field (nT)', rotation=270, labelpad=15)
    
    buf = radius_spot * 0.2
    ax2.set_xlim(center_spot[0]-radius_spot-buf, center_spot[0]+radius_spot+buf)
    ax2.set_ylim(center_spot[1]-radius_spot-buf, center_spot[1]+radius_spot+buf)
    ax2.set_zlim(center_spot[2]-radius_spot-buf, center_spot[2]+radius_spot+buf)
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_title("Background Noise (Unshielded)")
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    save_fig(fig2, "2_background_field_3d")
    plt.close(fig2)

    # (c) Total Field
    fig3 = plt.figure(figsize=(8, 7))
    ax3 = fig3.add_subplot(111, projection='3d')
    sc3 = ax3.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], c=total_sphere, cmap='plasma', s=40, alpha=0.9, vmin=vmin_common, vmax=vmax_common)
    cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.6, pad=0.1)
    cbar3.set_label('Bx Field (nT)', rotation=270, labelpad=15)
    
    ax3.set_xlim(ax2.get_xlim()); ax3.set_ylim(ax2.get_ylim()); ax3.set_zlim(ax2.get_zlim()); ax3.set_box_aspect((1, 1, 1))
    ax3.set_title(f"Residual Field (Shielded)\nSuppression: {suppression_local_db:.1f} dB")
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')

    save_fig(fig3, "3_total_field_3d")
    plt.close(fig3)

    # (d) Suppression Ratio (Optional, maybe skip for paper main body, keep for supplement)
    # fig4 = plt.figure(...)
    
    # (e) Coil Breakdown (Multi-channel)
    # Simplified for paper: just show one representative unit or the whole array in one clean shot
    # But let's keep the full breakdown for now
    print("  -> Generating Coil Breakdown Figure...")
    channels_to_plot = manager.base_channels 
    n_ch = len(channels_to_plot)
    cols = 3
    rows = int(np.ceil(n_ch / cols))
    fig5 = plt.figure(figsize=(15, 4 * rows))
    
    for idx, ch in enumerate(channels_to_plot):
        ax = fig5.add_subplot(rows, cols, idx + 1, projection='3d')
        c_map = {'bx':'r','by':'g','bz':'b','gxx':'orange','gyy':'lime','gxy':'cyan'}
        c = c_map.get(ch, 'k')
        
        has_current = False
        for u_idx in range(manager.num_units):
            b_idx = manager.base_channels.index(ch)
            stride = manager.total_ch_per_unit
            base_ptr = u_idx * stride + b_idx * 2
            
            w_top = x_opt[base_ptr]
            w_bot = x_opt[base_ptr + 1]
            
            if abs(w_top) < 1e-9 and abs(w_bot) < 1e-9:
                continue
            has_current = True
            
            base_coils = manager.standard_units[ch]
            moved_coils = coils.translate_coils(base_coils, manager.layout[u_idx])
            
            for top, bot, _, _ in moved_coils:
                if abs(w_top) > 1e-9:
                    ax.plot(top[:,0], top[:,1], top[:,2], color=c, alpha=0.6, linewidth=1.0)
                if abs(w_bot) > 1e-9:
                    ax.plot(bot[:,0], bot[:,1], bot[:,2], color=c, alpha=0.6, linewidth=1.0)
        
        ax.set_title(f"Active Channel: {ch.upper()}")
        ax.set_xlim(-limit_3d, limit_3d); ax.set_ylim(-limit_3d, limit_3d); ax.set_zlim(-limit_3d, limit_3d)
        ax.set_box_aspect((1, 1, 1))
        ax.axis('off') # Cleaner for breakdown
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    save_fig(fig5, "5_coils_breakdown")
    plt.close(fig5)

    # (f) Vector Field Comparison (The Money Shot)
    print("  -> Generating Vector Field Comparison...")
    fig6 = plt.figure(figsize=(14, 6)) # Wider for side-by-side
    
    pts_vec = sphere_pts
    vec_bg = B_bg_vectors[region_mask] * 1e9
    mag_bg = np.linalg.norm(vec_bg, axis=1)
    vec_res = B_total[region_mask] * 1e9
    mag_res = np.linalg.norm(vec_res, axis=1)
    v_max = np.max(mag_bg)
    
    # Subsample for cleaner vector plot
    idx_arrow = np.arange(len(pts_vec))
    if len(pts_vec) > 300:
        skip = len(pts_vec) // 200 # Aim for ~200 arrows
        idx_arrow = idx_arrow[::skip]
        
    def plot_quiver(ax, pts, vecs, mags, title):
        mags_safe = np.where(mags == 0, 1, mags)
        # Normalize length for uniform arrows, color denotes magnitude
        q = ax.quiver(pts[:,0], pts[:,1], pts[:,2], vecs[:,0]/mags_safe, vecs[:,1]/mags_safe, vecs[:,2]/mags_safe, 
                      length=0.12, normalize=True, cmap='jet', array=mags, pivot='middle') # 'jet' is high contrast
        q.set_clim(0, v_max)
        ax.set_title(title, fontsize=14)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(ax2.get_xlim()); ax.set_ylim(ax2.get_ylim()); ax.set_zlim(ax2.get_zlim())
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        return q

    ax6_1 = fig6.add_subplot(121, projection='3d')
    q1 = plot_quiver(ax6_1, pts_vec[idx_arrow], vec_bg[idx_arrow], mag_bg[idx_arrow], "OFF (Unshielded)")
    
    ax6_2 = fig6.add_subplot(122, projection='3d')
    plot_quiver(ax6_2, pts_vec[idx_arrow], vec_res[idx_arrow], mag_res[idx_arrow], "ON (Active Shielding)")
    
    # Single colorbar for both
    cbar_ax = fig6.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig6.colorbar(q1, cax=cbar_ax)
    cbar.set_label('Magnetic Field Magnitude (nT)', fontsize=12)

    save_fig(fig6, "6_field_vectors_comparison")
    plt.close(fig6)

    # (g) Unit Fingerprints (The "Activation Map")
    print("  -> Generating Unit Fingerprints (12 Channels per Unit)...")
    
    # Setup 3x3 Grid
    fig7 = plt.figure(figsize=(18, 18))
    plt.suptitle(f"Array Activation Map: Current Distribution per Unit\nTotal Current Norm: {np.linalg.norm(x_opt):.2f} A", fontsize=16, y=0.95)
    
    # Global current normalization for color scale
    # We want to see relative strength across the whole array
    max_current = np.max(np.abs(x_opt))
    if max_current < 1e-9: max_current = 1.0 # Avoid div zero
    
    # Layout order: Top-Left to Bottom-Right (0..8)
    # Physical layout matches plot layout
    # Layout indices:
    # 6 7 8  (y=s)
    # 3 4 5  (y=0)
    # 0 1 2  (y=-s)
    # Our manager.layout is ordered by loops: x in [-s,0,s], y in [-s,0,s]
    # layout[0]=(-s,-s), layout[1]=(-s,0), layout[2]=(-s,s) ... NO
    # Let's check layout order:
    # for x in offsets: for y in offsets:
    # 0:(-s,-s), 1:(-s,0), 2:(-s,s)
    # 3:( 0,-s), 4:( 0,0), 5:( 0,s) ...
    # So it is column-major? Let's map it to visual 3x3 grid (row, col)
    # Grid (Row 0 is Top y=+s, Row 2 is Bot y=-s)
    # Unit 2 is (-s, s) -> Row 0, Col 0
    # Unit 5 is ( 0, s) -> Row 0, Col 1
    # Unit 8 is ( s, s) -> Row 0, Col 2
    
    # Unit 1 is (-s, 0) -> Row 1, Col 0
    # ...
    # Unit 0 is (-s,-s) -> Row 2, Col 0
    
    # Map unit_index to subplot_index (1..9)
    # Subplot 1 (Top-Left) -> Unit with y=+s, x=-s -> Find it
    
    layout_map = {}
    for i, pos in enumerate(manager.layout):
        # Quantize pos to find grid index
        x, y = pos[0], pos[1]
        # Col: -s->0, 0->1, s->2
        col = int(np.round((x/s) + 1))
        # Row: s->0, 0->1, -s->2 (Inverted Y)
        row = int(np.round(1 - (y/s)))
        layout_map[(row, col)] = i

    for row in range(3):
        for col in range(3):
            subplot_idx = row * 3 + col + 1
            ax = fig7.add_subplot(3, 3, subplot_idx, projection='3d')
            
            unit_idx = layout_map.get((row, col))
            if unit_idx is None: continue
            
            unit_pos = manager.layout[unit_idx]
            
            # Extract currents for this unit
            # 12 channels
            stride = manager.total_ch_per_unit
            base_ptr = unit_idx * stride
            unit_weights = x_opt[base_ptr : base_ptr+stride]
            
            # Plot each channel
            # We iterate base_channels
            # ch_idx 0 (bx): Top=w[0], Bot=w[1]
            # ch_idx 1 (by): Top=w[2], Bot=w[3] ...
            
            plotted_something = False
            
            for b_i, ch_name in enumerate(manager.base_channels):
                w_top = unit_weights[b_i * 2]
                w_bot = unit_weights[b_i * 2 + 1]
                
                # Get Geometry
                base_coils = manager.standard_units[ch_name]
                moved_coils = coils.translate_coils(base_coils, unit_pos)
                
                # Color based on Magnitude
                # Top
                if abs(w_top) > 1e-3 * max_current: # Threshold to reduce clutter
                    color_val = abs(w_top) / max_current
                    c = plt.cm.plasma(color_val)
                    
                    for top, bot, _, _ in moved_coils:
                         # Plot Line
                         ax.plot(top[:,0], top[:,1], top[:,2], color=c, linewidth=1.5 + 2*color_val)
                         # Plot Arrow (Direction) - Just one in the middle
                         mid = len(top) // 2
                         # Direction vector
                         vec = top[mid+1] - top[mid]
                         if w_top < 0: vec = -vec # Reverse if current is negative
                         
                         ax.quiver(top[mid,0], top[mid,1], top[mid,2], 
                                   vec[0], vec[1], vec[2], 
                                   color='cyan' if w_top>0 else 'magenta', length=0.1, normalize=True)
                    plotted_something = True

                # Bot
                if abs(w_bot) > 1e-3 * max_current:
                    color_val = abs(w_bot) / max_current
                    c = plt.cm.plasma(color_val)
                    
                    for top, bot, _, _ in moved_coils:
                         ax.plot(bot[:,0], bot[:,1], bot[:,2], color=c, linewidth=1.5 + 2*color_val)
                         mid = len(bot) // 2
                         vec = bot[mid+1] - bot[mid]
                         if w_bot < 0: vec = -vec
                         
                         ax.quiver(bot[mid,0], bot[mid,1], bot[mid,2], 
                                   vec[0], vec[1], vec[2], 
                                   color='cyan' if w_bot>0 else 'magenta', length=0.1, normalize=True)
                    plotted_something = True

            ax.set_title(f"Unit {unit_idx}\nPos: ({unit_pos[0]:.1f}, {unit_pos[1]:.1f})", fontsize=12)
            
            # Set limits local to the unit for zoom effect
            zoom = 0.5
            ax.set_xlim(unit_pos[0]-zoom, unit_pos[0]+zoom)
            ax.set_ylim(unit_pos[1]-zoom, unit_pos[1]+zoom)
            ax.set_zlim(-zoom, zoom)
            ax.set_box_aspect((1, 1, 1))
            ax.axis('off')
            ax.view_init(elev=45, azim=45)
            
            if not plotted_something:
                ax.text(unit_pos[0], unit_pos[1], 0, "Idle", ha='center')

    # Add shared Colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=max_current))
    sm.set_array([])
    cbar = fig7.colorbar(sm, ax=fig7.axes, shrink=0.5, pad=0.05)
    cbar.set_label('Current Magnitude (A)', fontsize=14)
    
    save_fig(fig6, "6_field_vectors_comparison")
    plt.close(fig6)

    # (g) Unit Fingerprints (Detailed Breakdown per Unit)
    print("  -> Generating Detailed Unit Fingerprints (9 PDFs)...")
    
    fp_dir = output_dir / "unit_fingerprints"
    fp_dir.mkdir(exist_ok=True)
    
    # Global scale for consistent color
    max_current = np.max(np.abs(x_opt))
    if max_current < 1e-9: max_current = 1.0

    # Iterate over all 9 units
    for u_idx in range(manager.num_units):
        unit_pos = manager.layout[u_idx]
        
        # Create a figure for this unit: 6 Base Types x 2 (Top/Bot) = 12 Subplots
        # Layout: 4 Rows x 3 Cols
        fig_u = plt.figure(figsize=(15, 20))
        fig_u.suptitle(f"Unit {u_idx} Fingerprint\nPos: ({unit_pos[0]:.1f}, {unit_pos[1]:.1f}, {unit_pos[2]:.1f})", fontsize=20, y=0.92)
        
        stride = manager.total_ch_per_unit
        base_ptr = u_idx * stride
        unit_weights = x_opt[base_ptr : base_ptr+stride]
        
        # Channel Order in base_channels: bx, by, bz, gxx, gyy, gxy
        # We want to display Top and Bot for each.
        
        # Subplot Layout Strategy:
        # Col 0: Bx, Gxx
        # Col 1: By, Gyy
        # Col 2: Bz, Gxy
        # Rows: Top(B), Bot(B), Top(G), Bot(G) ?
        # Let's keep it simple: 
        # Row 0: Bx (Top/Bot), By (Top) -> A bit messy.
        
        # Let's just loop linearly:
        # 0: Bx Top, 1: Bx Bot, 2: By Top, 3: By Bot ...
        # This will fill 4x3 grid (12 plots).
        
        for ch_i, ch_name in enumerate(manager.base_channels):
            # Base channel index ch_i (0..5)
            # Weights
            w_top = unit_weights[ch_i * 2]
            w_bot = unit_weights[ch_i * 2 + 1]
            
            base_coils = manager.standard_units[ch_name]
            moved_coils = coils.translate_coils(base_coils, unit_pos)
            
            # --- Plot Top Channel ---
            ax_top = fig_u.add_subplot(4, 3, ch_i * 2 + 1, projection='3d')
            c_val = abs(w_top) / max_current
            color = plt.cm.plasma(c_val)
            
            # Plot geometry
            for top, bot, _, _ in moved_coils:
                # Top Coil
                ax_top.plot(top[:,0], top[:,1], top[:,2], color=color, linewidth=2)
                # Direction Arrow
                if abs(w_top) > 1e-6:
                     mid = len(top) // 2
                     vec = top[mid+1] - top[mid]
                     if w_top < 0: vec = -vec
                     ax_top.quiver(top[mid,0], top[mid,1], top[mid,2], vec[0], vec[1], vec[2], 
                                   color='k', length=0.1, normalize=True)
            
            ax_top.set_title(f"{ch_name.upper()} TOP\nI = {w_top:.2f} A", fontsize=12, 
                             color='red' if abs(w_top)>0.1 else 'black')
            ax_top.axis('off'); ax_top.view_init(elev=90, azim=-90) # Top view
            
            # --- Plot Bot Channel ---
            # Wait, linear index ch_i*2 + 1 fills: 1, 3, 5... 
            # We want to fill grid nicely.
            # Let's map explicitly.
            # Grid: 
            # Row 0: Bx_T, By_T, Bz_T
            # Row 1: Bx_B, By_B, Bz_B
            # Row 2: Gxx_T, Gyy_T, Gxy_T
            # Row 3: Gxx_B, Gyy_B, Gxy_B
            
            # Re-do loop structure for better layout
            pass # Reset loop below
            
        fig_u.clf() # Clear and restart with better layout
        fig_u.suptitle(f"Unit {u_idx} Fingerprint\nPos: ({unit_pos[0]:.1f}, {unit_pos[1]:.1f})", fontsize=20, y=0.95)

        # Better Layout Logic
        # Columns: X-related, Y-related, Z/Mix-related
        # But we have 6 types.
        # Let's do:
        # Col 0: Bx, Gxx
        # Col 1: By, Gyy
        # Col 2: Bz, Gxy
        # Within each cell, Top is upper, Bot is lower? 
        # No, let's stick to 4x3 grid.
        
        layout_grid = [
            ['bx', 'by', 'bz'],      # Row 0: Top
            ['bx', 'by', 'bz'],      # Row 1: Bot
            ['gxx', 'gyy', 'gxy'],   # Row 2: Top
            ['gxx', 'gyy', 'gxy']    # Row 3: Bot
        ]
        
        for r in range(4):
            for c in range(3):
                ax = fig_u.add_subplot(4, 3, r*3 + c + 1, projection='3d')
                ch_name = layout_grid[r][c]
                is_top = (r % 2 == 0)
                
                # Get weight
                base_idx = manager.base_channels.index(ch_name)
                w_idx = base_idx * 2 + (0 if is_top else 1)
                weight = unit_weights[w_idx]
                
                # Plot
                base_coils = manager.standard_units[ch_name]
                moved_coils = coils.translate_coils(base_coils, unit_pos)
                
                # Color Logic: Red for Positive, Blue for Negative
                # Scale intensity by magnitude
                norm_val = abs(weight) / max_current
                # Add a base intensity (0.3) so small currents are still visible but faint
                color_intensity = 0.3 + 0.7 * norm_val
                
                if weight > 0:
                    color = plt.cm.Reds(color_intensity)
                else:
                    color = plt.cm.Blues(color_intensity)
                
                # Make very small currents transparent/grey to reduce noise
                if abs(weight) < 1e-3:
                    color = 'lightgrey'
                    alpha = 0.3
                else:
                    alpha = 1.0
                
                for top, bot, _, _ in moved_coils:
                    target_loop = top if is_top else bot
                    ax.plot(target_loop[:,0], target_loop[:,1], target_loop[:,2], color=color, linewidth=2.5, alpha=alpha)
                    
                    # No arrows anymore
                
                label_type = "TOP" if is_top else "BOT"
                ax.set_title(f"{ch_name.upper()} {label_type}\n{weight:.2f} A", fontsize=10)
                ax.axis('off')
                ax.set_box_aspect((1,1,1))
                # View: Top view for top plates, Bottom view for bot plates?
                # Consistent top view is easier to read.
                ax.view_init(elev=90, azim=-90) 
        
        # Save individual unit PDF
        fig_u.savefig(fp_dir / f"unit_{u_idx}_detail.pdf", bbox_inches='tight')
        fig_u.savefig(fp_dir / f"unit_{u_idx}_detail.svg", bbox_inches='tight')
        plt.close(fig_u)
        print(f"    -> Saved Unit {u_idx} fingerprint.")

    # (h) Export COMSOL Geometry (108 txt files)
    # ------------------------------------------
    print("\n[Export] Generating COMSOL geometry files (1.txt - 108.txt)...")
    comsol_dir = output_dir / "comsol_coils_txt"
    comsol_dir.mkdir(exist_ok=True)
    
    # Counter for file names (1..108)
    # This must match the order of 'x_opt' vector for consistency
    file_idx = 1
    LAYER_THICKNESS_MM = 1.6 # 1.6mm per layer stacking
    
    for u_idx, pos in enumerate(manager.layout):
        for ch_i, ch_name in enumerate(manager.base_channels): # Order: bx, by, bz, gxx, gyy, gxy
            base_coils = manager.standard_units[ch_name]
            moved_coils = coils.translate_coils(base_coils, pos)
            
            # Stacking logic:
            # Top coils stack UP (+Z direction)
            # Bot coils stack DOWN (-Z direction)
            z_offset_top = ch_i * LAYER_THICKNESS_MM
            z_offset_bot = - (ch_i * LAYER_THICKNESS_MM)
            
            # --- 1. Top Channel ---
            top_loops = [c[0] for c in moved_coils]
            fname_top = comsol_dir / f"{file_idx}.txt"
            
            with open(fname_top, 'w') as f:
                for loop in top_loops:
                    pts_mm = loop * 1000.0
                    # Apply Physical Stacking Offset
                    pts_mm[:, 2] += z_offset_top
                    
                    for p in pts_mm:
                        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            file_idx += 1
            
            # --- 2. Bottom Channel ---
            bot_loops = [c[1] for c in moved_coils]
            fname_bot = comsol_dir / f"{file_idx}.txt"
            
            with open(fname_bot, 'w') as f:
                for loop in bot_loops:
                    pts_mm = loop * 1000.0
                    # Apply Physical Stacking Offset
                    pts_mm[:, 2] += z_offset_bot
                    
                    for p in pts_mm:
                        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            file_idx += 1
            
    print(f"  -> Exported {file_idx-1} coil geometry files to {comsol_dir}")
    print(f"\n[Done] All publication figures and data saved to: {output_dir}")

if __name__ == "__main__":
    run_array_demo()
