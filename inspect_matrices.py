import numpy as np
from scipy.io import loadmat
import pathlib
import matplotlib.pyplot as plt
import sys

# Add src to path to use MatrixGenerator
sys.path.append('src')
try:
    from matrix_generator import MatrixGenerator
except ImportError:
    # Handle case where src is not in path or running from wrong dir
    sys.path.append(str(pathlib.Path(__file__).parent / 'src'))
    from matrix_generator import MatrixGenerator

def inspect_and_compare():
    data_dir = pathlib.Path("data")
    path_A = data_dir / "A_mirror.mat"
    path_T = data_dir / "T_mirror.mat"
    
    print("=== Matrix Forensics: Deep Dive ===")
    
    if not path_A.exists():
        print("Error: A_mirror.mat not found.")
        return
        
    mat_A = loadmat(path_A)
    A_legacy = mat_A['A'] 
    mat_T = loadmat(path_T)
    T_legacy = mat_T['T']
    
    # Generate New Matrices (Bx Mode)
    L = 0.85
    a = 0.7
    grid_res = 100
    gen = MatrixGenerator(L, a, grid_res=grid_res)
    
    # Target points: 6x6x6 grid in 20cm DSV
    lp = 0.2
    Np = 6
    ex = np.linspace(-lp, lp, Np)
    ey = np.linspace(-lp, lp, Np)
    ez = np.linspace(-lp, lp, Np)
    EX, EY, EZ = np.meshgrid(ex, ey, ez, indexing='ij')
    target_points = np.column_stack((EX.flatten(), EY.flatten(), EZ.flatten()))
    
    # Compute full A matrix using our generator (which now has -2.0 scaling)
    modes = (4, 4)
    print(f"\n[Generator] Computing full A matrix for Bx (4x4)...")
    A_new = gen.compute_A(target_points, modes, 'bx')
    
    # Compute full Gamma matrix (which now has 10.0 scaling)
    print(f"[Generator] Computing Gamma matrix...")
    Gamma_new = gen.compute_Gamma(modes, 'bx')
    
    # --- Analysis 1: Regularization Matrix Comparison ---
    print("\n--- 1. Regularization Matrix (T vs Gamma) ---")
    diag_T = np.diag(T_legacy)
    diag_G = np.diag(Gamma_new)
    
    print(f"Legacy T (diag) range: [{diag_T.min():.2f}, {diag_T.max():.2f}]")
    print(f"New Gamma (diag) range: [{diag_G.min():.2f}, {diag_G.max():.2f}]")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(diag_T, 'o-', label='Legacy T (Penalty)')
    ax1.plot(diag_G, 'x--', label='New Gamma (Power Dissipation)')
    ax1.set_title('Regularization Penalty Spectrum')
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Penalty Weight')
    ax1.legend()
    ax1.grid(True)
    plt.savefig('comparison_regularization.png')
    print("-> Saved 'comparison_regularization.png'")
    
    # --- Analysis 2: A Matrix Correlation Spectrum ---
    print("\n--- 2. Sensitivity Matrix Correlation Spectrum ---")
    correlations = []
    ratios = []
    
    for i in range(16):
        col_old = A_legacy[:, i]
        col_new = A_new[:, i]
        
        # Correlation
        corr = np.corrcoef(col_old, col_new)[0, 1]
        correlations.append(corr)
        
        # Scale Ratio (Mean Abs)
        if np.mean(np.abs(col_new)) > 1e-12:
            ratio = np.mean(np.abs(col_old)) / np.mean(np.abs(col_new))
        else:
            ratio = 0
        ratios.append(ratio)
        
    print(f"Min Correlation: {min(correlations):.4f}")
    print(f"Max Correlation: {max(correlations):.4f}")
    print(f"Avg Correlation: {np.mean(correlations):.4f}")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(range(16), correlations, color='skyblue')
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_title('Correlation Coefficient per Mode')
    ax2.set_xlabel('Mode Index')
    ax2.set_ylabel('Pearson Correlation')
    ax2.axhline(0.9, color='green', linestyle='--', label='Good Match (>0.9)')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.legend()
    plt.savefig('comparison_correlation.png')
    print("-> Saved 'comparison_correlation.png'")
    
    # Plot ratios
    print(f"Ratios: {ratios}")

if __name__ == "__main__":
    inspect_and_compare()
