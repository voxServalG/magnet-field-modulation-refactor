import pathlib
import numpy as np
from scipy.io import loadmat

def load_matrices(file_path_a: pathlib.Path, 
                  file_path_t: pathlib.Path, 
                  var_name: tuple[str, str] = ('A', 'T')
                  ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Loads sensitivity (A) and regularization (T) matrices from MATLAB files.

    Extracts the matrices typically named 'A' and 'T' that are required for the upcoming coil design inverse problem.

    Args:
        file_path_a (pathlib.Path): Path to the MATLAB file containing the sensitivity matrix 'A'.
        file_path_t (pathlib.Path): Path to the MATLAB file containing the regularization matrix 'T'.
        var_name (tuple[str, str], optional): Names of the matrices in the MATLAB files. Defaults to ('A', 'T').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the sensitivity matrix 'A' and the regularization matrix 'T'.

    Raises:
        FileNotFoundError: If either file path does not exist.
        KeyError: If the expected variable keys ('A', 'T') are not found in the files.

    '''
    mat_data_a = loadmat(file_path_a, appendmat=False)
    A = mat_data_a.get(var_name[0])
    if A is None:
        raise KeyError(f"Variable '{var_name[0]}' not found in {file_path_a}")

    mat_data_t = loadmat(file_path_t, appendmat=False)
    T = mat_data_t.get(var_name[1])
    if T is None:
        raise KeyError(f"Variable '{var_name[1]}' not found in {file_path_t}")
     
    return A, T

def solve_stream_function_coeffs(sensitivity_mAtrix : np.ndarray,
                                 reg_maTrix         : np.ndarray,
                                 reg_lambda         : float,
                                 target_fieldB       : np.ndarray)-> np.ndarray:
    '''
    Solves the Tikhonov regularization problem to find stream function coefficients.

    Computes la = (A'A + lambda * T'T)^-1 * A' * B_target using a linear system solver.

    Args:
        sensitivity_mAtrix (np.ndarray): Matrix A of shape (M_points, N_coeffs).
        reg_maTrix (np.ndarray): Matrix T of shape (N_coeffs, N_coeffs).
        reg_lambda (float): Regularization parameter lambda.
        target_fieldB (np.ndarray): Target magnetic field vector of shape (M_points, 1).


    Returns:
        np.ndarray: Coefficient vector 'la' of shape (N_coeffs, 1).

    Raises:
        ValueError: If input matrix dimensions are incompatible.
        numpy.linalg.LinAlgError: If the system matrix is singular or solver fails to converge.
    '''

    # 在 Python 中，A.T 是转置，@ 是矩阵乘法
    lhs = (sensitivity_mAtrix.T @ sensitivity_mAtrix) + (reg_lambda * (reg_maTrix.T @ reg_maTrix))
    rhs = sensitivity_mAtrix.T @ target_fieldB
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("Incompatible matrix dimensions for solving the system.")  

    phi_coeffs = np.linalg.solve(lhs, rhs)
    return phi_coeffs

def reconstruct_stream_function(phi_coeffs  : np.ndarray,
                                x_grid_1d   : np.ndarray,
                                y_grid_1d   : np.ndarray,
                                coil_L      : float,
                                modes       : tuple[int, int]) -> np.ndarray:
    '''
    Reconstructs the stream function from the coefficient vector 'phi_coeffs'.

    This function performs the inverse transformation from the coefficient space back to the physical space using a double Fourier series expansion.
    The stream function phi_grid defines the current density distribution on the coil plane.
    $$ \Phi(x, y) = - \sum_{m=1}^{M} \sum_{n=1}^{N} \lambda_{mn} \cdot \sin\left(\frac{n \pi x}{L}\right) \cdot \cos\left(\frac{(2m-1) \pi y}{2L}\right) $$
    where
        - $\lambda_{mn}$ are the coefficients from 'phi_coeffs'.
        - $L$ is the half-length of the coil plane.
        - $M$ and $N$ are the number of modes in the Fourier series.

    Args:
        phi_coeffs (np.ndarray): Coefficient vector 'phi_coeffs' of shape (M * N, 1) or (M * N,).
        x_grid_1d (np.ndarray): 1D grid of x-coordinates of the evaluation grid. (float64, (Np,))
        y_grid_1d (np.ndarray): 1D grid of y-coordinates of the evaluation grid. (float64, (Np,))
        coil_L (float): Half-length of the square coil plane.
        modes (tuple[int, int], optional): Number of Fourier modes (M, N).

    Returns:
        np.ndarray: Reconstructed stream function 'phi_grid' of shape (len(y_grid), len(x_grid)).

    Raises:
        ValueError:
            - If the length of `phi_coeffs` does not match the 'M * N'.
            - If `coil_L` is zero or negative (physical impossibility causing division by zero).
    '''

    XX, YY = np.meshgrid(x_grid_1d, y_grid_1d)
    phi_grid = np.zeros_like(XX)

    M, N = modes
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            # 注意：将 1-based 的 m,n 转换为 0-based 的数组索引
            coeff_index = (m - 1) * N + (n - 1)
            lambda_mn = phi_coeffs[coeff_index]

            sin_term = np.sin(n * np.pi * XX / coil_L)
            cos_term = np.cos((2 * m - 1) * np.pi * YY / (2 * coil_L))
            
            term = lambda_mn * sin_term * cos_term
            phi_grid -= term

    return phi_grid