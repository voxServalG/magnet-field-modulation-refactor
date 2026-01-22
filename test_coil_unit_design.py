import sys
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src import solver, coils, physics, visuals
from src.factory import CoilFactory

def run_simulation_pipeline(factory, component_name, output_dir, start_time_total):
    '''
    Runs the full simulation pipeline for a specific component with classic Step 1-6 output.
    '''
    comp_label = component_name.upper()
    print("========================================================")
    print(f"   Project Nuke: {comp_label} Coil Design & Verification   ")
    print("========================================================")
    print(f"[Config] lambda: {factory.config['reg_lambda']}, turns: {factory.num_turns}, L: {factory.L}m, a: {factory.config['a']}m")

    # Step 1, 2, 3 are handled inside factory.create_component
    start_time_comp = time.time()
    try:
        coils_3d = factory.create_component(component_name)
    except Exception as e:
        print(f"  [Error] Factory failed to create {comp_label}: {e}")
        return

    if not coils_3d:
        print(f"  [Warning] No coils generated for {comp_label}. Skipping.")
        return

    # 4. Physics Verification
    print(f"\n[Step 4] Verifying Magnetic Field (Physics Simulation)...")
    lp = 0.2 
    Np_field = 7
    eval_x = np.linspace(-lp, lp, Np_field)
    eval_y = np.linspace(-lp, lp, Np_field)
    eval_z = np.linspace(-lp, lp, Np_field)
    
    EX, EY, EZ = np.meshgrid(eval_x, eval_y, eval_z, indexing='ij') 
    target_points_flat = np.column_stack((EX.flatten(), EY.flatten(), EZ.flatten()))
    
    print(f"  -> Evaluating field at {len(target_points_flat)} points ({Np_field}x{Np_field}x{Np_field} grid).")
    print("  -> Computing... (This uses vectorized Biot-Savart + 26 Mirrors)")
    
    sim_start = time.time()
    total_B = physics.calculate_field_from_coils(coils_3d, target_points_flat, current_I=1.0, use_shielding=True)
    sim_time = time.time() - sim_start
    print(f"  -> Calculation finished in {sim_time:.2f}s.")
    
    # 5. Analysis & Visualization
    print("\n[Step 5] Analyzing Results...")
    comp_idx_map = {'bx': 0, 'by': 1, 'bz': 2}
    comp_idx = comp_idx_map.get(component_name.lower(), 0)
    
    B_main = total_B[:, comp_idx] * 1e9 # nT
    center_idx = len(target_points_flat) // 2
    B0 = B_main[center_idx]
    
    error_rel = np.abs((B_main - B0) / (B0 if abs(B0) > 1e-9 else 1.0))
    max_error = np.max(error_rel)
    
    print(f"  -> Center Field B0 (B{component_name[-1]}): {B0:.4f} nT")
    print(f"  -> Max Relative Error:   {max_error:.6%} (Target: <5%)")
    
    # Visuals
    prefix = component_name.lower()
    visuals.plot_coil_geometry_3d(coils_3d, filename=str(output_dir / f"{prefix}_geometry_3d.svg"))
    visuals.plot_field_heatmap(total_B, target_points_flat, component_idx=comp_idx, 
                               filename=str(output_dir / f"{prefix}_heatmap.svg"))
    print(f"  -> Saved: results/{prefix}_heatmap.svg")
    
    print("  -> Generating 3D Vector Plot...")
    visuals.plot_field_vectors_3d(total_B, target_points_flat, 
                                  filename=str(output_dir / f"{prefix}_vectors_3d.svg"))
    print(f"  -> Saved: results/{prefix}_vectors_3d.svg")
    
    print("  -> Generating 3D Orthogonal Slices Plot (COMSOL Style)...")
    visuals.plot_orthogonal_slices_3d(total_B, target_points_flat, component_idx=comp_idx,
                                      filename=str(output_dir / f"{prefix}_slices_3d.svg"))
    print(f"  -> Saved: results/{prefix}_slices_3d.svg")

    # 6. Report
    print("\n[Step 6] Generating Report...")
    sim_params = {
        "Coil Component": comp_label,
        "Coil Half-Length L (m)": factory.L,
        "Coil Position a (m)": factory.config['a'],
        "Num Turns": factory.num_turns,
        "Verification Grid": f"{Np_field}x{Np_field}x{Np_field}"
    }
    sim_results = {
        "Center Field B0 (nT)": B0,
        "Max Relative Error (%)": max_error * 100,
        "Comp Execution Time (s)": time.time() - start_time_comp
    }
    visuals.save_simulation_report(str(output_dir / f"report_{prefix}.txt"), sim_params, sim_results)
    print(f"  -> Saved: results/report_{prefix}.txt")

    elapsed_total = time.time() - start_time_total
    print(f"\n[Done] Total execution time so far: {elapsed_total:.2f}s")
    print("========================================================\n")


def main():
    start_time_total = time.time()
    
    # Create Timestamped Output Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[System] Output directory created: {output_dir}")
    
    # Configuration
    config = {
        'data_dir': current_dir / "data",
        'L': 0.85,
        'a': 0.7,
        'reg_lambda': 1.0e-14, # Updated based on matrix inspection (balance point ~9e-15)
        'modes': (4, 4),
        'num_turns': 22,
        'grid_res': 400
    }
    
    factory = CoilFactory(config)
    
    # Sequential Pipeline Runs
    run_simulation_pipeline(factory, 'bx', output_dir, start_time_total)
    # 2. Build By (Rotated Bx) - Note: Factory now handles By natively if requested
    run_simulation_pipeline(factory, 'by', output_dir, start_time_total)
    
    # 3. Build Bz (Now fully implemented!)
    run_simulation_pipeline(factory, 'bz', output_dir, start_time_total)
    
    print("All requested components have been processed successfully.")

if __name__ == "__main__":
    main()
