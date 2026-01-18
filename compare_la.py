import sys
import pathlib
import numpy as np
from scipy.io import loadmat

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src import solver

def verify_la():
    # Paths
    data_dir = current_dir / "data"
    path_sensitivity_file = data_dir / "A_mirror.mat"
    path_gamma_file = data_dir / "T_mirror.mat"
    path_la = data_dir / "la_saved.mat"
    
    # 1. Calculate Python Result
    print("Calculating Python C_coeffs...")
    reg_lambda = 1.6544e-20
    target_points_count = 216
    
    try:
        A, Gamma = solver.load_matrices(path_sensitivity_file, path_gamma_file, var_name=('A', 'T'))
    except Exception as e:
        print(f"Error loading matrices: {e}")
        return

    b_target = np.ones((target_points_count, 1))
    # Solve using the new API
    C_coeffs_py = solver.solve_stream_function_coeffs(A, Gamma, reg_lambda, b_target)
    
    # 2. Load MATLAB Reference
    print(f"Loading reference from {path_la}...")
    try:
        mat_la = loadmat(str(path_la))
        # Check keys to find the variable name, usually 'la'
        key_la = [k for k in mat_la.keys() if not k.startswith('__')][0]
        la_ref = mat_la[key_la]
        print(f"Found variable '{key_la}' with shape {la_ref.shape}")
    except Exception as e:
        print(f"Error loading la.mat: {e}")
        return

    # 3. Compare
    # Ensure shapes match
    if C_coeffs_py.shape != la_ref.shape:
        print(f"Shape mismatch! Python: {C_coeffs_py.shape}, MATLAB: {la_ref.shape}")
        # Try reshaping python result to match reference if it's just (N,1) vs (N,)
        if C_coeffs_py.flatten().shape == la_ref.flatten().shape:
            C_coeffs_py = C_coeffs_py.reshape(la_ref.shape)
            print("Reshaped Python result to match reference.")
        else:
            return

    diff = np.abs(C_coeffs_py - la_ref)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print("-" * 30)
    print("Comparison Result:")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    # Check absolute values to see relative error
    max_val = np.max(np.abs(la_ref))
    print(f"Max value in ref: {max_val:.6e}")
    if max_val > 0:
        print(f"Max relative error: {max_diff / max_val:.6%}")
    else:
        print("Reference is all zeros?")

    if max_diff < 1e-10: # Strict tolerance
        print(">>")
        print("MATCH: The calculated coefficients match the MATLAB reference.")
    else:
        print(">>")
        print("MISMATCH: There is a significant difference.")
        print("Top 5 Python values:\n", C_coeffs_py.flatten()[:5])
        print("Top 5 MATLAB values:\n", la_ref.flatten()[:5])

if __name__ == "__main__":
    verify_la()
