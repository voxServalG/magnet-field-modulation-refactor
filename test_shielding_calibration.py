import numpy as np
import pathlib
import sys
import time
import matplotlib.pyplot as plt

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src.factory import CoilFactory
from src.array_manager import ArrayActiveShielding
from src import physics, visuals, coils

def run_calibration_test():
    print("========================================================")
    print("   Project Nuke: Spatial Calibration & Mirror Test   ")
    print("========================================================")

    # 1. Setup Environment
    # --------------------
    # Room: 2.4m x 1.9m x 1.65m (Half-dims)
    SHIELD_DIMS = (2.4, 1.9, 1.65)
    
    config = {
        'L': 0.6,   # Unit Half-Length (1.2m size)
        'a': 0.7,   # Unit Half-Distance (1.4m plate spacing)
        'modes': (4, 4),
        'reg_lambda': 1e-18,
        'num_turns': 5, # Low turns for fast matrix compute
        'grid_res': 60,
        'shield_dims': SHIELD_DIMS
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
    
    # 2. Condition Number Analysis
    # ----------------------------
    # We compare Free Space vs Shielded Room
    print("\n[Step 1] Comparing Matrix Health (Condition Number)...")
    
    # Target ROI: Central sphere
    Np = 15
    gx = np.linspace(-1.0, 1.0, Np)
    target_points = np.column_stack((gx, gx, gx)) # Small diagonal sample for speed
    
    # A) Free Space
    manager_free = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=None)
    S_free = manager_free.compute_response_matrix(target_points)
    cond_free = np.linalg.cond(S_free)
    
    # B) Shielded Room (The new logic!)
    manager_shield = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=SHIELD_DIMS)
    S_shield = manager_shield.compute_response_matrix(target_points)
    cond_shield = np.linalg.cond(S_shield)
    
    print(f"\n[Matrix Health Report]")
    print(f"  -> FREE SPACE Condition Number: {cond_free:.2e}")
    print(f"  -> SHIELDED   Condition Number: {cond_shield:.2e}")
    
    ratio = cond_shield / cond_free
    print(f"  -> Degredation Ratio: {ratio:.2f}x")
    
    if ratio > 100:
        print("  [Warning] High Ill-conditioning detected! Mirror overlap is significant.")
    else:
        print("  [Safe] System remains robust in shielded environment.")

    # 3. Visualization of the "Virtual Array"
    # ---------------------------------------
    print("\n[Step 2] Visualizing Real + Mirror Coils...")
    # Generate one sample coil (BX) and its mirrors
    sample_coils = factory.create_component('bx')
    
    # Extract only the first segment's mirrors for clarity
    p_start, p_end, _ = sample_coils[0][0][0], sample_coils[0][0][1], 1.0 # Top loop first segment
    segments = physics.get_mirror_positions(p_start, p_end, shield_dims=SHIELD_DIMS)
    
    print(f"  -> Found {len(segments)} mirror segments for a single wire.")
    
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Shield Boundaries
    x1, y1, z1 = SHIELD_DIMS
    corners = [
        [-x1, -y1, -z1], [x1, -y1, -z1], [x1, y1, -z1], [-x1, y1, -z1], [-x1, -y1, -z1],
        [-x1, -y1, z1], [x1, -y1, z1], [x1, y1, z1], [-x1, y1, z1], [-x1, -y1, z1]
    ]
    corners = np.array(corners)
    ax.plot(corners[:,0], corners[:,1], corners[:,2], 'k--', alpha=0.3, label="Shield Walls")
    
    # Plot Mirrors
    for s, e, scale in segments:
        color = 'red' if scale > 0 else 'blue' # Red for Positive Image, Blue for Negative
        alpha = 1.0 if (s == p_start).all() else 0.4
        label = "Original" if alpha == 1.0 else None
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color, alpha=alpha, linewidth=2, label=label)
    
    ax.set_title("Magnetic Mirror Test: Positive (Red) vs Negative (Blue)")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    
    output_path = pathlib.Path("results/mirror_calibration_check.svg")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"  -> Visualization saved to: {output_path}")

if __name__ == "__main__":
    run_calibration_test()
