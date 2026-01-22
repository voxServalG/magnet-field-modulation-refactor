import numpy as np
from scipy.constants import mu_0
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Mock
    def njit(*args, **kwargs):
        def decorator(f): return f
        return decorator
    prange = range

@njit(fastmath=True, parallel=True)
def _biot_savart_surface_kernel(src_pos, src_J, target_pos, factor):
    '''
    Numba-accelerated kernel for Surface Biot-Savart Integration.
    B = factor * sum( (J x r) / |r|^3 )
    
    Args:
        src_pos: (Ns, 3) source coordinates
        src_J:   (Ns, 3) source current densities
        target_pos: (Nt, 3) target coordinates
        factor:  float, (mu0 / 4pi) * dA
        
    Returns:
        B_total: (Nt, 3)
    '''
    Nt = target_pos.shape[0]
    Ns = src_pos.shape[0]
    B_total = np.zeros((Nt, 3))
    
    for i in prange(Nt):
        tx = target_pos[i, 0]
        ty = target_pos[i, 1]
        tz = target_pos[i, 2]
        
        sum_bx = 0.0
        sum_by = 0.0
        sum_bz = 0.0
        
        for j in range(Ns):
            sx = src_pos[j, 0]
            sy = src_pos[j, 1]
            sz = src_pos[j, 2]
            
            jx = src_J[j, 0]
            jy = src_J[j, 1]
            jz = src_J[j, 2]
            
            # vector r = target - source
            rx = tx - sx
            ry = ty - sy
            rz = tz - sz
            
            r_sq = rx*rx + ry*ry + rz*rz
            
            # Avoid singularity
            if r_sq < 1e-12:
                continue
                
            r_mag = np.sqrt(r_sq)
            r_cube = r_mag * r_sq
            
            # Cross Product: J x r
            cx = jy*rz - jz*ry
            cy = jz*rx - jx*rz
            cz = jx*ry - jy*rx
            
            # Add to sum: (J x r) / r^3
            inv_r3 = 1.0 / r_cube
            sum_bx += cx * inv_r3
            sum_by += cy * inv_r3
            sum_bz += cz * inv_r3
            
        B_total[i, 0] = sum_bx * factor
        B_total[i, 1] = sum_by * factor
        B_total[i, 2] = sum_bz * factor
        
    return B_total

class MatrixGenerator:
    '''
    Generates Sensitivity Matrix (A) and Regularization Matrix (Gamma) for coil design.
    
    This class performs the numerical integration of the Biot-Savart law to link
    stream function basis coefficients to magnetic field values at target points.
    It supports different symmetries (Bx, By, Bz) by selecting appropriate basis functions.
    '''

    def __init__(self, L: float, a: float, grid_res: int = 100, shield_dims: tuple = (0.95, 0.95, 0.8)):
        '''
        Initialize the MatrixGenerator with coil geometry and shielding parameters.

        Args:
            L (float): Half-length of the square coil plane (m).
            a (float): Half-distance between the two coil planes (m). z = +/- a.
            grid_res (int): Resolution of the integration grid on the coil plane (default 100).
            shield_dims (tuple): Dimensions of the magnetic shielding room (x1, y1, z1). 
                                 Default (0.95, 0.95, 0.8) matches the physics verification module.
        '''
        self.L = L
        self.a = a
        self.grid_res = grid_res
        self.shield_dims = shield_dims
        
        # Pre-compute integration grid on the coil plane
        # We use a cell-centered grid to avoid singularities at edges if any
        self.x_coil = np.linspace(-L, L, grid_res)
        self.y_coil = np.linspace(-L, L, grid_res)
        self.XX, self.YY = np.meshgrid(self.x_coil, self.y_coil)
        self.dA = (self.x_coil[1] - self.x_coil[0]) * (self.y_coil[1] - self.y_coil[0])
        
        # Flatten coil grid for vectorized operations
        # Shape: (N_source_points, 1)
        self.X_flat = self.XX.flatten()[:, np.newaxis]
        self.Y_flat = self.YY.flatten()[:, np.newaxis]
        self.N_source = len(self.X_flat)

    def _get_mirror_sources(self, src_pos: np.ndarray, src_J: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        '''
        Generates positions and current densities for the original source and its mirrors.
        Based on the Method of Images for a rectangular magnetic shield.
        
        Args:
            src_pos: (Ns, 3) Source positions [x, y, z]
            src_J: (Ns, 3) Current densities [Jx, Jy, Jz]
            
        Returns:
            List of tuples (mirror_pos, mirror_J)
        '''
        x1, y1, z1 = self.shield_dims
        mirrors = []
        
        # Iterate through mirror indices corresponding to x, y, z reflections (-1, 0, 1)
        # i, j, k = 0 is the original source
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    # Parity determines if the coordinate is reflected
                    # (-1)^0 = 1 (No reflection), (-1)^1 = -1 (Reflection)
                    parity_x = (-1)**abs(i)
                    parity_y = (-1)**abs(j)
                    parity_z = (-1)**abs(k)
                    
                    # 1. Calculate Mirror Positions
                    # x' = 2*i*L + x*parity
                    # We utilize broadcasting. src_pos is (Ns, 3)
                    
                    # Shift term
                    shift = np.array([2 * i * x1, 2 * j * y1, 2 * k * z1])
                    
                    # Parity vector for coordinates
                    parity_vec = np.array([parity_x, parity_y, parity_z])
                    
                    mirror_pos = shift + src_pos * parity_vec
                    
                    # 2. Calculate Mirror Currents
                    # Current vector transformation under reflection:
                    # For magnetic walls (high mu), tangential components reverse, normal component stays?
                    # Wait, let's look at the physics.py logic which was verified to work.
                    # It transforms POINTS: p_start -> p_start_mirror, p_end -> p_end_mirror.
                    # The vector dL = p_end - p_start transforms as:
                    # dL_x' = (end_x' - start_x') = (shift_x + end_x*p) - (shift_x + start_x*p) = (end_x - start_x)*p = dL_x * p
                    # So the current vector components multiply by the SAME parity as the coordinates!
                    
                    mirror_J = src_J * parity_vec
                    
                    mirrors.append((mirror_pos, mirror_J))
                    
        return mirrors

    def _get_current_density(self, m: int, n: int, basis_type: str) -> tuple[np.ndarray, np.ndarray]:
        '''
        Calculates the current density Jx, Jy for a specific mode (m, n).

        J = curl(Phi * z_hat) = (dPhi/dy) * x_hat - (dPhi/dx) * y_hat

        Args:
            m (int): Fourier mode index m.
            n (int): Fourier mode index n.
            basis_type (str): 'sc', 'cs', 'cc', or 'ss' (Fourier primitives).
                              Legacy names 'bx' (sc), 'by' (cs), 'bz' (cc) also supported.

        Returns:
            tuple[np.ndarray, np.ndarray]: Flattened Jx and Jy arrays on the coil grid.
        '''
        pi = np.pi
        L = self.L
        X = self.X_flat
        Y = self.Y_flat
        
        # Mapping legacy names to primitives
        typ = basis_type.lower()
        if typ == 'bx': typ = 'sc'
        elif typ == 'by': typ = 'cs'
        elif typ == 'bz': typ = 'cc'

        if typ == 'sc': # Sine-Cosine (for Bx and Gxz)
            k_x, k_y = m * pi / L, (2 * n - 1) * pi / (2 * L)
            dPhi_dx = k_x * np.cos(k_x * X) * np.cos(k_y * Y)
            dPhi_dy = -k_y * np.sin(k_x * X) * np.sin(k_y * Y)
            
        elif typ == 'cs': # Cosine-Sine (for By and Gyz)
            k_x, k_y = (2 * m - 1) * pi / (2 * L), n * pi / L
            dPhi_dx = -k_x * np.sin(k_x * X) * np.sin(k_y * Y)
            dPhi_dy = k_y * np.cos(k_x * X) * np.cos(k_y * Y)
            
        elif typ == 'cc': # Cosine-Cosine (for Bz and Gzz)
            k_x, k_y = (2 * m - 1) * pi / (2 * L), (2 * n - 1) * pi / (2 * L)
            dPhi_dx = -k_x * np.sin(k_x * X) * np.cos(k_y * Y)
            dPhi_dy = -k_y * np.cos(k_x * X) * np.sin(k_y * Y)

        elif typ == 'ss': # Sine-Sine (for Gxy)
            k_x, k_y = m * pi / L, n * pi / L
            dPhi_dx = k_x * np.cos(k_x * X) * np.sin(k_y * Y)
            dPhi_dy = k_y * np.sin(k_x * X) * np.cos(k_y * Y)
            
        else:
            raise ValueError(f"Unknown basis_type: {basis_type}")
            
        Jx, Jy = dPhi_dy, -dPhi_dx
        return Jx, Jy

    def compute_A(self, target_points: np.ndarray, modes: tuple[int, int], basis_type: str, 
                  explicit_parity: float = None, target_component: str = None) -> np.ndarray:
        '''
        Computes the Sensitivity Matrix A with Mirroring.

        Args:
            target_points (np.ndarray): (N_target, 3) coordinates of target points.
            modes (tuple): (M, N) number of modes.
            basis_type (str): 'sc', 'cs', 'cc', 'ss' (or legacy 'bx', 'by', 'bz').
            explicit_parity (float): Optional. Force +1.0 or -1.0 parity.
            target_component (str): Optional. Which component to extract ('x','y','z'). 
                                    Defaults to the one implied by basis_type.

        Returns:
            np.ndarray: Matrix A of shape (N_target, M*N). 
        '''
        basis_type = basis_type.lower()
        M, N = modes
        num_coeffs = M * N
        num_targets = len(target_points)
        
        A = np.zeros((num_targets, num_coeffs))
        
        # Define source planes
        Z_top = np.full_like(self.X_flat, self.a)
        src_top = np.hstack([self.X_flat, self.Y_flat, Z_top]) 
        
        Z_bot = np.full_like(self.X_flat, -self.a)
        src_bot = np.hstack([self.X_flat, self.Y_flat, Z_bot]) 
        
        # Determine symmetry parity
        if explicit_parity is not None:
            current_parity = explicit_parity
        else:
            # Legacy/Fallback logic
            if basis_type in ['bx', 'by', 'sc', 'cs', 'ss']:
                current_parity = -1.0 # Anti-symmetric (Odd)
            else: # bz / cc
                current_parity = 1.0 # Symmetric (Even)
            
        # Determine extraction component
        if target_component is None:
            if 'x' in basis_type: target_component = 'x'
            elif 'y' in basis_type: target_component = 'y'
            else: target_component = 'z'
        target_component = target_component.lower()

        print(f"  [MatrixGen] Computing A matrix for {basis_type} ({M}x{N} modes) [Parity={current_parity}, Comp={target_component}]...")
        
        total_modes = M * N
        processed_count = 0
        
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                col_idx = (m - 1) * N + (n - 1)
                processed_count += 1
                
                if processed_count % max(1, total_modes // 5) == 0:
                     print(f"    -> Mode {processed_count}/{total_modes}...")

                # Get J distribution for this mode (on Top Plane)
                Jx, Jy = self._get_current_density(m, n, basis_type)
                J_top = np.hstack([Jx, Jy, np.zeros_like(Jx)]) 
                J_bot = J_top * current_parity
                
                # Compute B at all target points
                B_col = np.zeros((num_targets, 3))
                
                mirrors_top = self._get_mirror_sources(src_top, J_top)
                for m_pos, m_J in mirrors_top:
                    B_col += self._integrate_biot_savart(m_pos, m_J, target_points)
                    
                mirrors_bot = self._get_mirror_sources(src_bot, J_bot)
                for m_pos, m_J in mirrors_bot:
                    B_col += self._integrate_biot_savart(m_pos, m_J, target_points)
                
                # Extract requested component
                if target_component == 'x':
                    A[:, col_idx] = B_col[:, 0]
                elif target_component == 'y':
                    A[:, col_idx] = B_col[:, 1]
                elif target_component == 'z':
                    A[:, col_idx] = B_col[:, 2]
        
        return A

    def _integrate_biot_savart(self, src_pos: np.ndarray, src_J: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        '''
        Performs numerical integration of Biot-Savart law.
        B = (mu0 / 4pi) * integral( (J x r) / |r|^3 dA )
        
        Args:
            src_pos: (Ns, 3) Source positions
            src_J: (Ns, 3) Current density vectors
            target_pos: (Nt, 3) Target positions
            
        Returns:
            (Nt, 3) B-field vectors
        '''
        constant = (mu_0 / (4 * np.pi)) * self.dA
        
        # Numba optimization
        # Ensure arrays are contiguous for C-backend efficiency
        src_pos = np.ascontiguousarray(src_pos)
        src_J = np.ascontiguousarray(src_J)
        target_pos = np.ascontiguousarray(target_pos)
        
        return _biot_savart_surface_kernel(src_pos, src_J, target_pos, constant)

    def compute_Gamma(self, modes: tuple[int, int], basis_type: str) -> np.ndarray:
        '''
        Computes the Regularization Matrix Gamma.
        Ideally, this represents the power dissipation or stored energy.
        
        As a simplified approximation consistent with literature, 
        we can use a diagonal matrix weighted by mode indices (higher modes -> higher penalty).
        Or simply Identity for 0th order Tikhonov.
        
        Let's implement a 'Power Dissipation' approximation:
        Gamma_mn = Integral( |J_mn|^2 dA )
        
        Args:
            modes (tuple): (M, N)
            basis_type (str): 'bx', 'by', 'bz'
            
        Returns:
            np.ndarray: Matrix Gamma of shape (M*N, M*N).
        '''
        basis_type = basis_type.lower()
        M, N = modes
        num_coeffs = M * N
        Gamma = np.zeros((num_coeffs, num_coeffs))
        
        print(f"  [MatrixGen] Computing Gamma matrix (Power Dissipation)...")
        
        # Calculate diagonal elements (Self-power)
        # We assume orthogonality (mostly true for Fourier) makes off-diagonals small or zero.
        # Calculating full matrix is safer.
        
        # Actually, for Fourier sine/cosine on a square, they are orthogonal.
        # So Gamma should be diagonal.
        
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                col_idx = (m - 1) * N + (n - 1)
                
                Jx, Jy = self._get_current_density(m, n, basis_type)
                
                # J_sq = Jx^2 + Jy^2
                J_sq = Jx**2 + Jy**2
                
                # Integral(|J|^2 dA)
                power_integral = np.sum(J_sq) * self.dA
                
                # We multiply by 2 because there are two planes (top and bottom)
                # Both contribute to power dissipation.
                Gamma[col_idx, col_idx] = np.sqrt(2 * power_integral) 
                # Note: Tikhonov term is lambda * ||Gamma * C||^2. 
                # If we want lambda * Power, and Power = C' * G_sq * C
                # Then Gamma should be sqrt(G_sq). 
                
        # Scaling factor to align with legacy T matrix magnitude
        return Gamma
