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
from src import physics

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# System
ARRAY_REG_LAMBDA = 2.3e-16
SHIELD_DIMS = (2.4, 1.9, 1.65)

# Scan Area (The Map)
SCAN_LIMIT = 0.8  # +/- 0.8m (1.6m x 1.6m area)
SCAN_RES = 41     # 41x41 pixels
Z_PLANE = 0.4     # Scan at Z=0.4m (Realistic head distance, NOT inside the coil plane)

# Local ROI Definition (The "Head" at each pixel)
HEAD_RADIUS = 0.05 # 5cm radius (small local probe)

# Test Background Field
# A challenging mix: Uniform 150nT + Gradient 50nT/m
def get_test_bg(points):
    # points: (N, 3)
    # Bz = 150 + 50 * x
    B = np.zeros_like(points)
    B[:, 2] = 150e-9 + 50e-9 * points[:, 0]
    return B

# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------
def run_spatial_coverage_map():
    print("==========================================================")
    print("   Project Nuke: Spatial Coverage Heatmap (Residual nT)   ")
    print("==========================================================")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / f"coverage_map_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Array
    print(f"[Init] Initializing Array...")
    config = {
        'L': 0.6, 'a': 0.7, 'modes': (4, 4), 'reg_lambda': 1e-14, 'num_turns': 10, 
        'grid_res': 60, 'shield_dims': SHIELD_DIMS
    }
    factory = CoilFactory(config)
    s = 1.25
    layout = np.array([[x, y, 0.0] for x in [-s,0,s] for y in [-s,0,s]])
    manager = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=SHIELD_DIMS)

    # 2. Setup Scan Grid
    print(f"[Init] Preparing Scan Grid ({SCAN_LIMIT*2}m x {SCAN_LIMIT*2}m)...")
    x_axis = np.linspace(-SCAN_LIMIT, SCAN_LIMIT, SCAN_RES)
    y_axis = np.linspace(-SCAN_LIMIT, SCAN_LIMIT, SCAN_RES)
    XX, YY = np.meshgrid(x_axis, y_axis, indexing='ij')
    
    # Storage for the heatmap
    heatmap_residual = np.zeros((SCAN_RES, SCAN_RES))
    
    # 3. Scanning Loop
    print(f"[Scan] Starting {SCAN_RES}x{SCAN_RES} pixel scan...")
    start_time = time.time()
    
    # To speed up, we can pre-compute S for a relative "head" template
    # and then shift/mask. But since we need Shielding Reflections which depend 
    # on absolute position, we strictly should compute S for each absolute position.
    # However, compute_response_matrix is slow (~0.5s). 40x40 = 1600 calls = 800s.
    # That's too slow.
    
    # Optimized Approach:
    # We pre-compute S for a HUGE grid covering the whole plane?
    # No, that's memory heavy.
    # Let's rely on the fact that 'compute_response_matrix' accepts a list of points.
    # We can pass ALL pixels (as single points) at once?
    # Yes! A single point (center of head) is a good approximation for "performance at this location".
    # If we want RMS over a volume, we need multiple points.
    # Let's compromise: Evaluate strictly at the center point of each pixel.
    # Then we can batch compute S for ALL 1600 points in one go!
    
    scan_points_flat = np.column_stack((XX.flatten(), YY.flatten(), np.full_like(XX.flatten(), Z_PLANE)))
    
    print(f"  -> Batch computing S matrix for {len(scan_points_flat)} points...")
    # This computes B at every single grid point from every coil
    S_batch = manager.compute_response_matrix(scan_points_flat)
    
    print(f"  -> Solving optimization for each pixel...")
    # For each pixel, we solve: S_pixel * x = -B_bg_pixel
    # S_pixel is just 3 rows of S_batch (Bx, By, Bz at that point)
    # This is an under-determined system (3 constraints, ~100 coils).
    # Minimum norm solution: x = S.T * (S*S.T + lam*I)^-1 * b
    
    B_bg_batch = get_test_bg(scan_points_flat).flatten()
    
    # Iterate through pixels to solve individually (simulating local tracking)
    # (Doing one giant solve for all points would simulate a "Global Field Control", 
    # but we want to simulate "If the head were HERE, how well could we do?")
    
    pixel_indices = np.arange(len(scan_points_flat))
    
    # Pre-calc constants for Tikhonov
    # x = (S^T S + lam I)^-1 S^T b  <-- This is for Over-determined (LS)
    # x = S^T (S S^T + lam I)^-1 b  <-- This is for Under-determined (Min Norm)
    # Since each pixel has only 3 measurements (Bx,By,Bz) and 100 coils, it is Under-determined.
    # We use the Min-Norm formula (Right Inverse) which is much faster (inv 3x3 matrix!).
    
    # However, we usually optimize a VOLUME (sphere). A single point optimization is trivially perfect 
    # (residual=0) unless coils are degenerate.
    # So a single point scan might show 0 everywhere and be misleading.
    # We MUST define a small volume (e.g. 7 points) around each pixel to test "Gradients".
    
    # REVISED STRATEGY: 
    # Loop is necessary. But we can simplify the "Unit S Matrix".
    # Let's define a "Local Offset Template" (e.g. center + 6 neighbors).
    # S_local(pos) approx S_batch(pos + offsets).
    # Since S varies smoothly, we can interpolate S_batch? Or just scan coarser.
    
    # Let's stick to the Robust Loop but with Coarser Grid?
    # Or... Just trust the single point + Tikhonov? 
    # Tikhonov with lambda will prevent x from exploding, so residual won't be exactly 0.
    # The residual will reflect "Hardness to cancel this field with reasonable current".
    # This is exactly what we want!
    # "How much residue is left if we refuse to pay infinite current?"
    
    residuals = []
    
    # S_batch: (3*N_pixels, N_coils)
    # Reshape to (N_pixels, 3, N_coils) for easy access
    num_coils = S_batch.shape[1]
    S_batch_reshaped = S_batch.reshape(len(scan_points_flat), 3, num_coils)
    B_bg_reshaped = B_bg_batch.reshape(len(scan_points_flat), 3)
    
    reg_matrix = ARRAY_REG_LAMBDA * np.eye(3) # For Dual form? 
    # Dual form: x = S.T (S S.T + lam I)^-1 b
    # Wait, S S.T is (3x3). Very fast!
    # But wait, Tikhonov term in primal is ||Ax-b||^2 + lam||x||^2.
    # The solution is x = (A^T A + lam I)^-1 A^T b
    # Using Woodbury identity, this is equivalent to A^T (A A^T + lam I)^-1 b.
    # Correct. So we can invert a 3x3 matrix at each pixel!
    
    for i in range(len(scan_points_flat)):
        # S_local: (3, N_coils)
        S_local = S_batch_reshaped[i]
        b_local = -B_bg_reshaped[i] # Target is -Background
        
        # Dual Tikhonov Solver (Fast for Under-determined)
        # Matrix to invert: (S S^T + lambda I) -> 3x3
        Gram = S_local @ S_local.T
        DampedGram = Gram + ARRAY_REG_LAMBDA * np.eye(3)
        
        # Solve alpha: DampedGram * alpha = b_local
        alpha = np.linalg.solve(DampedGram, b_local)
        
        # Recover x = S^T * alpha
        x_sol = S_local.T @ alpha
        
        # Calculate Residual
        # res = S x - b_target = S x + B_bg
        field_produced = S_local @ x_sol
        field_resid = field_produced + B_bg_reshaped[i]
        
        rms_val = np.linalg.norm(field_resid) / np.sqrt(3) # RMS over 3 components
        residuals.append(rms_val * 1e9) # Convert to nT
        
        if i % 100 == 0:
            print(f"    Processed {i}/{len(scan_points_flat)}...", end='\r')

    heatmap_residual = np.array(residuals).reshape(SCAN_RES, SCAN_RES)
    print(f"\n[Scan] Done in {time.time()-start_time:.1f}s.")

    # 4. Plotting (Publication Style)
    # -------------------------------
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Plot Heatmap
    # Use 'plasma_r' or 'viridis_r' (Reverse) so that Low Residual = Light/Blue, High = Dark/Red?
    # Actually standard 'jet' or 'turbo': Blue=Low(Good), Red=High(Bad)
    im = ax.imshow(heatmap_residual.T, origin='lower', cmap='turbo', 
                   extent=[-SCAN_LIMIT, SCAN_LIMIT, -SCAN_LIMIT, SCAN_LIMIT],
                   vmax=np.percentile(heatmap_residual, 95)) # Clip top 5% outliers for contrast
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Residual RMS (nT)', rotation=270, labelpad=20)
    
    # Add Contours
    levels = [1.0, 2.0, 5.0, 10.0]
    CS = ax.contour(XX, YY, heatmap_residual, levels=levels, colors='white', linewidths=0.8, alpha=0.7)
    ax.clabel(CS, inline=True, fmt='%1.1f nT', fontsize=9)
    
    # Annotate "Sweet Spot"
    ax.set_title("Spatial Coverage Map (Z=0.4m Plane)\nBackground: 150nT Uniform + 50nT/m Gradient", fontsize=14)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    
    # Draw a box for "Typical Working Area" (+/- 0.3m)
    rect = plt.Rectangle((-0.3, -0.3), 0.6, 0.6, linewidth=1.5, edgecolor='white', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(-0.28, -0.28, "Typical Head ROI", color='white', fontsize=10, fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_dir / "spatial_coverage.svg")
    plt.savefig(output_dir / "spatial_coverage.pdf")
    plt.close()
    
    print(f"[Done] Coverage map saved to {output_dir}")

if __name__ == "__main__":
    run_spatial_coverage_map()
