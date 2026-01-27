import numpy as np
import sys
import pathlib

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src.factory import CoilFactory
from src.array_manager import ArrayActiveShielding

# Config matching the heatmap script
ARRAY_REG_LAMBDA = 2.3e-16
SHIELD_DIMS = (2.4, 1.9, 1.65)
Z_PLANE = 0.4

def get_test_bg(points):
    # Bz = 150 + 50 * x
    B = np.zeros_like(points)
    B[:, 2] = 150e-9 + 50e-9 * points[:, 0]
    return B

def inspect_values():
    print(f"--- Heatmap Numerical Inspection (Z={Z_PLANE}m) ---")
    
    # Setup System
    config = {'L': 0.6, 'a': 0.7, 'modes': (4, 4), 'reg_lambda': 1e-14, 'num_turns': 10, 'grid_res': 60, 'shield_dims': SHIELD_DIMS}
    factory = CoilFactory(config)
    s = 1.25
    layout = np.array([[x, y, 0.0] for x in [-s,0,s] for y in [-s,0,s]])
    manager = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=SHIELD_DIMS)

    # Define Probe Points
    probes = {
        "Center (0,0)": [0, 0, Z_PLANE],
        "Inner Ring (0.2,0)": [0.2, 0, Z_PLANE],
        "Mid Ring (0.4,0)": [0.4, 0, Z_PLANE],
        "Outer Ring (0.6,0)": [0.6, 0, Z_PLANE],
        "Edge (0.8,0)": [0.8, 0, Z_PLANE],
        "Corner (0.6,0.6)": [0.6, 0.6, Z_PLANE]
    }
    
    # Compute
    pts = np.array(list(probes.values()))
    S_batch = manager.compute_response_matrix(pts)
    B_bg = get_test_bg(pts).flatten()
    
    # Solve per point
    # Reshape S: (N_pts, 3, N_coils)
    S_reshaped = S_batch.reshape(len(pts), 3, -1)
    B_bg_reshaped = B_bg.reshape(len(pts), 3)
    
    print(f"{'Location':<20} | {'Resid(nT)':<10} | {'Suppr(dB)':<10} | {'Status'}")
    print("-" * 60)
    
    keys = list(probes.keys())
    for i in range(len(pts)):
        S_loc = S_reshaped[i]
        b_loc = -B_bg_reshaped[i]
        
        # Solver
        Gram = S_loc @ S_loc.T
        alpha = np.linalg.solve(Gram + ARRAY_REG_LAMBDA * np.eye(3), b_loc)
        x = S_loc.T @ alpha
        
        # Resid
        res_vec = S_loc @ x + B_bg_reshaped[i]
        res_rms = np.linalg.norm(res_vec) / np.sqrt(3) * 1e9
        bg_rms = np.linalg.norm(B_bg_reshaped[i]) / np.sqrt(3) * 1e9
        
        db = 20 * np.log10(bg_rms / res_rms)
        
        status = "Unknown"
        if res_rms < 1.0: status = "PERFECT (Black/Blue)"
        elif res_rms < 5.0: status = "GOOD (Green)"
        elif res_rms > 10.0: status = "BAD (Red)"
        
        print(f"{keys[i]:<20} | {res_rms:7.4f}    | {db:7.2f}    | {status}")

if __name__ == "__main__":
    inspect_values()
