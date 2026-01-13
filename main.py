import sys
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src import solver, coils, physics, visuals

def main():
    print("========================================================")
    print("   Project Nuke: Bx Coil Design & Verification (Python)   ")
    print("========================================================")
    
    # ---------------------------------------------------------
    # 1. Configuration & Constants
    # ---------------------------------------------------------
    start_time = time.time()
    
    # Paths
    # Adjust these relative paths based on your actual folder structure
    matlab_base_dir = current_dir.parent / "离散化磁场计算及优化算法Bx"
    path_A = matlab_base_dir / "A_mirror.mat"
    path_T = matlab_base_dir / "T_mirror.mat"
    output_dir = current_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Physical Parameters
    L = 0.85          # Coil half-length (m)
    a = 0.7           # Coil Z-plane position (m)
    reg_lambda = 1.6544e-20 # Tikhonov regularization parameter
    target_points_count = 216
    
    # Discretization Parameters
    modes = (4, 4)    # Fourier modes (M, N)
    num_turns = 22    # Number of discrete wire loops
    grid_res = 400    # Resolution for stream function (400x400)
    
    print(f"[Config] lambda: {reg_lambda}, turns: {num_turns}, L: {L}m, a: {a}m")

    # ---------------------------------------------------------
    # 2. Solver: Inverse Problem
    # ---------------------------------------------------------
    print("\n[Step 1] Solving Inverse Problem...")
    try:
        A, T = solver.load_matrices(path_A, path_T, var_name=('A', 'T'))
        print(f"  -> Loaded matrices: A {A.shape}, T {T.shape}")
    except Exception as e:
        print(f"  [Error] Failed to load MAT files: {e}")
        print("  Please check the file paths in main.py")
        return

    # Target field: Uniform Bx=1 at all target points
    target_field = np.ones((target_points_count, 1))
    
    # Solve for coefficients
    phi_coeffs = solver.solve_stream_function_coeffs(A, T, reg_lambda, target_field)
    print(f"  -> Solved coefficients. Shape: {phi_coeffs.shape}")

    # Reconstruct Stream Function
    print("\n[Step 2] Reconstructing Stream Function...")
    x_grid = np.linspace(-L, L, grid_res)
    y_grid = np.linspace(-L, L, grid_res)
    
    phi_grid = solver.reconstruct_stream_function(phi_coeffs, x_grid, y_grid, L, modes)
    print(f"  -> Stream function grid generated: {phi_grid.shape}")
    
    # Plot Stream Function
    visuals.plot_stream_function(phi_grid, x_grid, y_grid, 
                                 filename=str(output_dir / "stream_function.png"))
    print("  -> Saved: results/stream_function.png")

    # ---------------------------------------------------------
    # 3. Coils: Discretization & Geometry
    # ---------------------------------------------------------
    print("\n[Step 3] Extracting Coil Geometry...")
    # Extract 2D contours
    coils_2d = coils.extract_contour_paths(phi_grid, x_grid, y_grid, num_turns)
    print(f"  -> Extracted {len(coils_2d)} discrete loops from contours.")
    
    # Generate 3D vertices (Bi-planar)
    coils_3d = coils.generate_coil_vertices(coils_2d, z_position=a, downsample_factor=5)
    print(f"  -> Generated 3D geometry (Top & Bottom planes).")
    
    # Plot 3D Geometry
    visuals.plot_coil_geometry_3d(coils_3d, filename=str(output_dir / "coil_geometry_3d.png"))
    print("  -> Saved: results/coil_geometry_3d.png")

    # ---------------------------------------------------------
    # 4. Physics: Forward Verification (Biot-Savart)
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Magnetic Field (Physics Simulation)...")
    
    # Define Target Region (DSV): 5x5x5 grid inside 20cm sphere
    lp = 0.2 
    Np_field = 5
    eval_x = np.linspace(-lp, lp, Np_field)
    eval_y = np.linspace(-lp, lp, Np_field)
    eval_z = np.linspace(-lp, lp, Np_field)
    
    # Create evaluation grid points (Meshgrid -> Flatten)
    EX, EY, EZ = np.meshgrid(eval_x, eval_y, eval_z, indexing='ij') # Standard order
    target_points_flat = np.column_stack((EX.flatten(), EY.flatten(), EZ.flatten()))
    print(f"  -> Evaluating field at {len(target_points_flat)} points (5x5x5 grid).")
    print("  -> Computing... (This uses vectorized Biot-Savart + 26 Mirrors)")
    
    sim_start = time.time()
    total_B = physics.calculate_field_from_coils(coils_3d, target_points_flat, current_I=1.0, use_shielding=True)
    sim_time = time.time() - sim_start
    print(f"  -> Calculation finished in {sim_time:.2f}s.")
    
    # ---------------------------------------------------------
    # 5. Analysis & Visualization
    # ---------------------------------------------------------
    print("\n[Step 5] Analyzing Results...")
    
    # Extract Bx component
    Bx = total_B[:, 0] * 1e9 # nT
    
    center_idx = len(target_points_flat) // 2
    B0 = Bx[center_idx]
    
    # Error Calculation
    error_rel = np.abs((Bx - B0) / B0)
    max_error = np.max(error_rel)
    
    print(f"  -> Center Field B0 (Bx): {B0:.4f} nT")
    print(f"  -> Max Relative Error:   {max_error:.6%} (Target: <5%)")
    
    # Plot Heatmap (Center Slice Z=0 roughly)
    # We filter points where z is close to 0
    z_center_mask = np.abs(target_points_flat[:, 2]) < 1e-5
    if np.any(z_center_mask):
        visuals.plot_field_heatmap(total_B[z_center_mask], target_points_flat[z_center_mask], 
                                   component_idx=0, 
                                   filename=str(output_dir / "field_heatmap_Bx.png"))
        print("  -> Saved: results/field_heatmap_Bx.png")
    
    elapsed = time.time() - start_time
    print(f"\n[Done] Total execution time: {elapsed:.2f}s")
    print("========================================================")

if __name__ == "__main__":
    main()