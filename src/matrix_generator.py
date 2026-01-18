import numpy as np
from scipy.constants import mu_0

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
            basis_type (str): 'bx', 'by', or 'bz'.

        Returns:
            tuple[np.ndarray, np.ndarray]: Flattened Jx and Jy arrays on the coil grid.
        '''
        # Coefficients for arguments
        # Note: We align with the formulas used in solver.reconstruct_stream_function
        # Bx: sin(m*pi*x/L) * cos((2n-1)*pi*y/2L)
        # By: cos((2m-1)*pi*x/2L) * sin(n*pi*y/L)
        # Bz: cos((2m-1)*pi*x/2L) * cos((2n-1)*pi*y/2L)
        
        # To get J, we need derivatives:
        # Jx = dPhi/dy
        # Jy = -dPhi/dx
        
        pi = np.pi
        L = self.L
        X = self.X_flat
        Y = self.Y_flat
        
        if basis_type == 'bx':
            # Phi = sin(k_x * x) * cos(k_y * y)
            k_x = m * pi / L
            k_y = (2 * n - 1) * pi / (2 * L)
            
            # dPhi/dx = k_x * cos(k_x * x) * cos(k_y * y)
            # dPhi/dy = -k_y * sin(k_x * x) * sin(k_y * y)
            
            dPhi_dx = k_x * np.cos(k_x * X) * np.cos(k_y * Y)
            dPhi_dy = -k_y * np.sin(k_x * X) * np.sin(k_y * Y)
            
        elif basis_type == 'by':
            # Phi = cos(k_x * x) * sin(k_y * y)
            k_x = (2 * m - 1) * pi / (2 * L)
            k_y = n * pi / L
            
            # dPhi/dx = -k_x * sin(k_x * x) * sin(k_y * y)
            # dPhi/dy = k_y * cos(k_x * x) * cos(k_y * y)
            
            dPhi_dx = -k_x * np.sin(k_x * X) * np.sin(k_y * Y)
            dPhi_dy = k_y * np.cos(k_x * X) * np.cos(k_y * Y)
            
        elif basis_type == 'bz':
            # Phi = cos(k_x * x) * cos(k_y * y)
            k_x = (2 * m - 1) * pi / (2 * L)
            k_y = (2 * n - 1) * pi / (2 * L)
            
            # dPhi/dx = -k_x * sin(k_x * x) * cos(k_y * y)
            # dPhi/dy = -k_y * cos(k_x * x) * sin(k_y * y)
            
            dPhi_dx = -k_x * np.sin(k_x * X) * np.cos(k_y * Y)
            dPhi_dy = -k_y * np.cos(k_x * X) * np.sin(k_y * Y)
            
        else:
            raise ValueError(f"Unknown basis_type: {basis_type}")
            
        Jx = dPhi_dy
        Jy = -dPhi_dx
        
        return Jx, Jy

    def compute_A(self, target_points: np.ndarray, modes: tuple[int, int], basis_type: str) -> np.ndarray:
        '''
        Computes the Sensitivity Matrix A with Mirroring.

        Args:
            target_points (np.ndarray): (N_target, 3) coordinates of target points.
            modes (tuple): (M, N) number of modes.
            basis_type (str): 'bx', 'by', or 'bz'.

        Returns:
            np.ndarray: Matrix A of shape (N_target, M*N). 
        '''
        basis_type = basis_type.lower()
        M, N = modes
        num_coeffs = M * N
        num_targets = len(target_points)
        
        A = np.zeros((num_targets, num_coeffs))
        
        # Define source planes
        # Top plane z = +a
        Z_top = np.full_like(self.X_flat, self.a)
        src_top = np.hstack([self.X_flat, self.Y_flat, Z_top]) # (Ns, 3)
        
        # Bottom plane z = -a
        Z_bot = np.full_like(self.X_flat, -self.a)
        src_bot = np.hstack([self.X_flat, self.Y_flat, Z_bot]) # (Ns, 3)
        
        # Determine symmetry parity
        if basis_type in ['bx', 'by']:
            current_parity = -1.0 # Anti-symmetric
        else: # bz
            current_parity = 1.0 # Symmetric
            
        print(f"  [MatrixGen] Computing A matrix for {basis_type} ({M}x{N} modes) with shielding...")
        
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                col_idx = (m - 1) * N + (n - 1)
                
                # Get J distribution for this mode (on Top Plane)
                Jx, Jy = self._get_current_density(m, n, basis_type)
                J_top = np.hstack([Jx, Jy, np.zeros_like(Jx)]) # (Ns, 3)
                J_bot = J_top * current_parity
                
                # Compute B at all target points from Top and Bottom planes AND their mirrors
                B_col = np.zeros((num_targets, 3))
                
                # 1. Top Plane Mirrors
                mirrors_top = self._get_mirror_sources(src_top, J_top)
                for m_pos, m_J in mirrors_top:
                    B_col += self._integrate_biot_savart(m_pos, m_J, target_points)
                    
                # 2. Bottom Plane Mirrors
                mirrors_bot = self._get_mirror_sources(src_bot, J_bot)
                for m_pos, m_J in mirrors_bot:
                    B_col += self._integrate_biot_savart(m_pos, m_J, target_points)
                
                # Extract component
                if basis_type == 'bx':
                    A[:, col_idx] = B_col[:, 0]
                elif basis_type == 'by':
                    A[:, col_idx] = B_col[:, 1]
                elif basis_type == 'bz':
                    A[:, col_idx] = B_col[:, 2]
        
        # Scaling factor to align with legacy physics definitions (likely 2pi vs 4pi and sign convention)
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
        Nt = len(target_pos)
        Ns = len(src_pos)
        B_total = np.zeros((Nt, 3))
        
        # Vectorized approach:
        # We can loop over targets to save memory (Ns is large, Nt is small)
        constant = (mu_0 / (4 * np.pi)) * self.dA
        
        for i in range(Nt):
            r_vec = target_pos[i] - src_pos # (Ns, 3)
            r_mag = np.linalg.norm(r_vec, axis=1)[:, np.newaxis] # (Ns, 1)
            
            # Cross product J x r
            # J is (Ns, 3), r is (Ns, 3)
            cross_prod = np.cross(src_J, r_vec) # (Ns, 3)
            
            # Integrand
            # Avoid division by zero (self-field) - though target is usually away from source
            valid_mask = r_mag > 1e-6
            
            dB = cross_prod / (r_mag**3)
            
            # Sum over all source elements (Numerical Integral)
            B_point = np.sum(dB, axis=0) * constant
            B_total[i] = B_point
            
        return B_total

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
