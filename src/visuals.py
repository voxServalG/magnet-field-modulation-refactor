import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_coil_geometry_3d(coils_3d: list, filename: str = None):
    '''
    Visualize the 3D geometry of the bi-planar coil system.
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Unpack 4-element tuples (top, bot, I_top, I_bot)
    # We ignore currents for pure geometry plotting
    for top_loop, bottom_loop, _, _ in coils_3d:
        ax.plot(top_loop[:, 0], top_loop[:, 1], top_loop[:, 2], 'r-', linewidth=1.5, alpha=0.8)
        ax.plot(bottom_loop[:, 0], bottom_loop[:, 1], bottom_loop[:, 2], 'b-', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Coil Geometry')
    
    all_coords = []
    for t, b, _, _ in coils_3d:
        all_coords.append(t)
        all_coords.append(b)
        
    if all_coords:
        all_points = np.vstack(all_coords)
        max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                              all_points[:,1].max()-all_points[:,1].min(), 
                              all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
        mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_field_heatmap(field_B: np.ndarray, target_points: np.ndarray, component_idx: int = 0, filename: str = None):
    '''
    Plot a 2D heatmap of a specific magnetic field component.
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    x = target_points[:, 0]
    y = target_points[:, 1]
    b_comp = field_B[:, component_idx] * 1e9 
    cntr = ax.tricontourf(x, y, b_comp, levels=20, cmap='viridis')
    cbar = fig.colorbar(cntr, ax=ax)
    cbar.set_label(f'B_{{"xyz"[component_idx]}} (nT)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Magnetic Field Distribution (B{{"xyz"[component_idx]}})')
    ax.set_aspect('equal')

    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_stream_function(phi_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, filename: str = None):
    '''
    Plot the 2D stream function contour map.
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        XX, YY = np.meshgrid(x_grid, y_grid)
    else:
        XX, YY = x_grid, y_grid
    cp = ax.contourf(XX, YY, phi_grid, levels=30, cmap='RdBu_r')
    fig.colorbar(cp, ax=ax, label='Stream Function (A)')
    ax.contour(XX, YY, phi_grid, levels=15, colors='k', linewidths=0.5, alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Stream Function & Coil Windings')
    ax.set_aspect('equal')
    
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_field_vectors_3d(field_B: np.ndarray, 
                          target_points: np.ndarray, 
                          coils_3d: list = None,
                          filename: str = None, 
                          sample_rate: int = 1):
    '''
    Visualize magnetic field vectors in 3D with a clean, modern aesthetic.
    '''
    import matplotlib.ticker as ticker
    
    fig = plt.figure(figsize=(10, 10), dpi=100) # Square figure for better 3D aspect
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Data Preparation
    # Convert to mm and nT
    pts_mm = target_points * 1000
    vecs_nt = field_B * 1e9
    
    # Calculate magnitude for color mapping
    magnitudes = np.linalg.norm(vecs_nt, axis=1)
    
    # Smart Subsampling logic
    # If we have too many points, visualization gets messy.
    # We aim for ~125 arrows max (5x5x5)
    total_points = len(pts_mm)
    if total_points > 150:
        # Calculate a dynamic stride to get roughly 100-150 points
        stride = int(np.ceil((total_points / 125)**(1/3)))
        # We need to reshape to grid to stride properly, or just stride linear array if ordered
        # Assuming ordered grid from meshgrid
        mask = np.zeros(total_points, dtype=bool)
        mask[::max(1, stride)] = True # Naive striding
    else:
        mask = np.ones(total_points, dtype=bool)

    # Apply manual sample_rate if provided and > 1 (overrides auto logic)
    if sample_rate > 1:
        mask = np.zeros(total_points, dtype=bool)
        mask[::sample_rate] = True
        
    x, y, z = pts_mm[mask, 0], pts_mm[mask, 1], pts_mm[mask, 2]
    u, v, w = vecs_nt[mask, 0], vecs_nt[mask, 1], vecs_nt[mask, 2]
    c = magnitudes[mask]
    
    # 2. Plotting Vectors
    # pivot='middle' centers the arrow on the grid point
    q = ax.quiver(x, y, z, u, v, w, 
                  length=30, # Length in data units (mm) - needs tuning based on grid spacing
                  normalize=True,
                  pivot='middle',
                  cmap='plasma', 
                  array=c, 
                  linewidths=1.5,
                  arrow_length_ratio=0.3) # Fatter arrow heads
    
    # 3. Aesthetics
    cbar = fig.colorbar(q, ax=ax, shrink=0.6, aspect=20, pad=0.05)
    cbar.set_label('|B| (nT)')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Magnetic Field Vectors', loc='left', fontsize=14, fontweight='bold')
    
    # Set Limits & Aspect
    limit = 200
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(100))
    
    # Clean Panes (make them transparent/white)
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.5))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.5))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.5))
    
    # Isometric-ish view
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect((1, 1, 1))

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_orthogonal_slices_3d(field_B: np.ndarray, 
                              target_points: np.ndarray, 
                              component_idx: int = 0,
                              filename: str = None):
    '''
    Visualize the magnetic field on three orthogonal planes (XY, XZ, YZ) matching COMSOL style.
    '''
    import matplotlib.ticker as ticker
    from scipy.interpolate import griddata
    from matplotlib import cm

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    pts_mm = target_points * 1000
    B_vals = field_B[:, component_idx] * 1e9
    x0, y0, z0 = 0, 0, 0 

    def add_surface_slice(axis_idx, fixed_val):
        mask = np.abs(pts_mm[:, axis_idx] - fixed_val) < 1.0 
        if not np.any(mask): return None
        
        p_slice = pts_mm[mask]
        b_slice = B_vals[mask]
        
        others = [i for i in range(3) if i != axis_idx]
        u_raw, v_raw = p_slice[:, others[0]], p_slice[:, others[1]]
        
        ui, vi = np.unique(u_raw), np.unique(v_raw)
        UI, VI = np.meshgrid(ui, vi)
        BI = griddata((u_raw, v_raw), b_slice, (UI, VI), method='linear')
        
        if axis_idx == 0:
            X, Y, Z = np.full_like(UI, fixed_val), UI, VI
        elif axis_idx == 1:
            X, Y, Z = UI, np.full_like(UI, fixed_val), VI
        else:
            X, Y, Z = UI, VI, np.full_like(UI, fixed_val)
            
        return ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet',
                               linewidth=0, antialiased=False, alpha=0.9,
                               vmin=B_vals.min(), vmax=B_vals.max())

    print(f"  [Debug] Field range for plotting: {B_vals.min():.2f} to {B_vals.max():.2f} nT")
    
    _ = add_surface_slice(2, z0)
    _ = add_surface_slice(1, y0)
    surf = add_surface_slice(0, x0)
    
    if surf:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
        cbar.set_label(f'B_{{"xyz"[component_idx]}} (nT)')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    comp_name = "xyz"[component_idx]
    ax.set_title(f'(i) B_{comp_name}', loc='left', fontsize=14, fontweight='bold')

    limit = 200
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.view_init(elev=20, azim=-35)
    ax.set_box_aspect((1, 1, 1)) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(200))

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def save_simulation_report(filename: str, params: dict, results: dict):
    '''
    Generates a text report summarizing the simulation parameters and results.
    '''
    with open(filename, 'w') as f:
        f.write("========================================================\n")
        f.write("      Project Nuke: Simulation Report\n")
        f.write("========================================================\n\n")
        f.write("1. System Configuration\n")
        f.write("-----------------------\n")
        for key, value in params.items():
            f.write(f"  {key:<20}: {value}\n")
        f.write("\n2. Simulation Results\n")
        f.write("---------------------\n")
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"  {key:<20}: {value:.6f}\n")
            else:
                f.write(f"  {key:<20}: {value}\n")
        f.write("\n3. Target Field Description\n")
        f.write("---------------------------\n")
        f.write("  Type                : Uniform Bx Field\n")
        f.write("  Target Intensity    : 1.0 (Arbitrary Units in Optimization)\n")
        f.write("  Region Shape        : Cubic Grid (DSV)\n")
        f.write("\nGenerated by Python Refactor System\n")
        f.write("========================================================\n")
