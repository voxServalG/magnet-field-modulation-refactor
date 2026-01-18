import pathlib
import numpy as np
from scipy.io import loadmat

def load_matrices(file_path_a: pathlib.Path, 
                  file_path_gamma: pathlib.Path, 
                  var_name: tuple[str, str] = ('A', 'T')
                  ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Loads sensitivity (A) and regularization (Gamma) matrices from MATLAB files.

    Extracts the matrices typically named 'A' and 'T' (mapped to Gamma) required for the coil design inverse problem.

    Args:
        file_path_a (pathlib.Path): Path to the MATLAB file containing the sensitivity matrix 'A'.
        file_path_gamma (pathlib.Path): Path to the MATLAB file containing the regularization matrix 'Gamma' (stored as 'T').
        var_name (tuple[str, str], optional): Names of the matrices in the MATLAB files. Defaults to ('A', 'T').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the sensitivity matrix 'A' and the regularization matrix 'Gamma'.

    Raises:
        FileNotFoundError: If either file path does not exist.
        KeyError: If the expected variable keys ('A', 'T') are not found in the files.

    '''
    mat_data_a = loadmat(file_path_a, appendmat=False)
    A = mat_data_a.get(var_name[0])
    if A is None:
        raise KeyError(f"Variable '{var_name[0]}' not found in {file_path_a}")

    mat_data_gamma = loadmat(file_path_gamma, appendmat=False)
    Gamma = mat_data_gamma.get(var_name[1])
    if Gamma is None:
        raise KeyError(f"Variable '{var_name[1]}' not found in {file_path_gamma}")
     
    return A, Gamma

def solve_stream_function_coeffs(A : np.ndarray,
                                 Gamma : np.ndarray,
                                 reg_lambda : float,
                                 b_target : np.ndarray)-> np.ndarray:
    '''
    Solves the Tikhonov regularization problem to find stream function coefficients C.

    Computes C = (A'A + lambda * Gamma'Gamma)^-1 * A' * b_target using a linear system solver.

    Equation:
        (A^T A + lambda * Gamma^T Gamma) * C = A^T * b_target

    Args:
        A (np.ndarray): Sensitivity matrix A of shape (M_points, N_coeffs).
        Gamma (np.ndarray): Regularization matrix Gamma of shape (N_coeffs, N_coeffs).
        reg_lambda (float): Regularization parameter lambda.
        b_target (np.ndarray): Target magnetic field vector b of shape (M_points, 1).


    Returns:
        np.ndarray: Coefficient vector 'C' of shape (N_coeffs, 1).

    Raises:
        ValueError: If input matrix dimensions are incompatible.
        numpy.linalg.LinAlgError: If the system matrix is singular or solver fails to converge.
    '''

    # Tikhonov Regularization Normal Equation
    lhs = (A.T @ A) + (reg_lambda * (Gamma.T @ Gamma))
    rhs = A.T @ b_target
    
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("Incompatible matrix dimensions for solving the system.")  

    C_coeffs = np.linalg.solve(lhs, rhs)
    return C_coeffs

def reconstruct_stream_function(C_coeffs    : np.ndarray,
                                x_grid_1d   : np.ndarray,
                                y_grid_1d   : np.ndarray,
                                coil_L      : float,
                                modes       : tuple[int, int],
                                coil_type   : str = 'bx') -> np.ndarray:
    '''
    Reconstructs the stream function from the coefficient vector 'C_coeffs'.

    This function performs the inverse transformation from the coefficient space back to the physical space 
    using a double Fourier series expansion adapted for the specific symmetry of the target coil (Bx, By, Bz).

    Formulas (based on Target Field Method):
    - Bx: sin(m*pi*x/L) * cos((2n-1)*pi*y/2L)
    - By: cos((2m-1)*pi*x/2L) * sin(n*pi*y/L)
    - Bz: cos((2m-1)*pi*x/2L) * cos((2n-1)*pi*y/2L)

    Args:
        C_coeffs (np.ndarray): Coefficient vector 'C_coeffs' of shape (M * N, 1) or (M * N,).
        x_grid_1d (np.ndarray): 1D grid of x-coordinates.
        y_grid_1d (np.ndarray): 1D grid of y-coordinates.
        coil_L (float): Half-length of the square coil plane.
        modes (tuple[int, int]): Number of Fourier modes (M, N).
        coil_type (str): Type of coil to reconstruct ('bx', 'by', 'bz'). Defaults to 'bx'.

    Returns:
        np.ndarray: Reconstructed stream function 'phi_grid'.
    '''

    XX, YY = np.meshgrid(x_grid_1d, y_grid_1d)
    phi_grid = np.zeros_like(XX)
    
    ctype = coil_type.lower()
    M, N = modes
    
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            # Convert 1-based m,n to 0-based array index
            coeff_index = (m - 1) * N + (n - 1)
            # Handle case where coeffs might be smaller than M*N if truncated
            if coeff_index >= len(C_coeffs):
                continue
                
            C_mn = C_coeffs[coeff_index]

            # Select basis functions based on symmetry
            if ctype == 'bx':
                # Bx: sin(x) * cos(y)
                term_x = np.sin(m * np.pi * XX / coil_L)
                term_y = np.cos((2 * n - 1) * np.pi * YY / (2 * coil_L))
            elif ctype == 'by':
                # By: cos(x) * sin(y) - Note indices swap roles compared to Bx usually
                # Standard from literature for By symmetry:
                term_x = np.cos((2 * m - 1) * np.pi * XX / (2 * coil_L))
                term_y = np.sin(n * np.pi * YY / coil_L)
            elif ctype == 'bz':
                # Bz: cos(x) * cos(y)
                term_x = np.cos((2 * m - 1) * np.pi * XX / (2 * coil_L))
                term_y = np.cos((2 * n - 1) * np.pi * YY / (2 * coil_L))
            else:
                raise ValueError(f"Unknown coil_type: {coil_type}")
            
            # The negative sign is a convention from the original Bx derivation
            # We keep it for consistency, though physically it just flips current direction
            term = C_mn * term_x * term_y
            phi_grid += term

    return phi_grid