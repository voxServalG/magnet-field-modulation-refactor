import numpy as np
import pathlib
from src import solver, coils
from src.matrix_generator import MatrixGenerator

class CoilFactory:
    '''
    Factory class to produce different coil components (0th order uniform, 1st order gradients).
    '''
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = config.get('data_dir')
        self.L = config.get('L', 0.85)
        self.modes = config.get('modes', (4, 4))
        self.num_turns = config.get('num_turns', 22)
        self.grid_res = config.get('grid_res', 400)
        
        # Initialize Matrix Generator
        # We assume target points are defined by the user config or defaults?
        # For now, let's define a standard target grid for matrix generation
        # This matches the legacy 216 points (6x6x6) but can be changed.
        self.a = config.get('a', 0.7)
        self.generator = MatrixGenerator(self.L, self.a, grid_res=100) # Increased to 100
        
        # Cache for matrices: {'bx': (A, Gamma), 'bz': (A, Gamma), ...}
        self.matrix_cache = {}
        
        # Define target points for optimization (where we want B=1)
        # Using a 6x6x6 grid in 20cm DSV to match legacy behavior
        lp = 0.2
        Np_opt = 6
        ex = np.linspace(-lp, lp, Np_opt)
        ey = np.linspace(-lp, lp, Np_opt)
        ez = np.linspace(-lp, lp, Np_opt)
        EX, EY, EZ = np.meshgrid(ex, ey, ez, indexing='ij')
        self.target_points = np.column_stack((EX.flatten(), EY.flatten(), EZ.flatten()))

    def _get_matrices(self, basis_type: str):
        '''Lazy loader for matrices'''
        if basis_type in self.matrix_cache:
            return self.matrix_cache[basis_type]
        
        print(f"[Factory] Generating matrices for {basis_type.upper()} from scratch...")
        A = self.generator.compute_A(self.target_points, self.modes, basis_type)
        Gamma = self.generator.compute_Gamma(self.modes, basis_type)
        
        # Debug: Check matrix scales
        norm_A = np.linalg.norm(A)
        norm_G = np.linalg.norm(Gamma)
        print(f"  [Debug] Norm(A): {norm_A:.4e}, Norm(Gamma): {norm_G:.4e}")
        print(f"  [Debug] A shape: {A.shape}, Gamma shape: {Gamma.shape}")
        
        self.matrix_cache[basis_type] = (A, Gamma)
        return A, Gamma

    def _solve_and_discretize(self, target_field_vec, reg_lambda, component_label):
        basis_type = component_label.lower()
        
        A, Gamma = self._get_matrices(basis_type)
            
        # 1. Solve Inverse Problem
        print(f"\n[Step 1] Solving Inverse Problem for {basis_type.upper()}...")
        # Use the passed lambda
        C_coeffs = solver.solve_stream_function_coeffs(A, Gamma, reg_lambda, target_field_vec)
        print(f"  -> Solved coefficients C. Shape: {C_coeffs.shape}")
        
        # 2. Reconstruct Stream Function
        print(f"\n[Step 2] Reconstructing Stream Function...")
        x_grid = np.linspace(-self.L, self.L, self.grid_res)
        y_grid = np.linspace(-self.L, self.L, self.grid_res)
        phi_grid = solver.reconstruct_stream_function(C_coeffs, x_grid, y_grid, self.L, self.modes, coil_type=basis_type)
        print(f"  -> Stream function grid generated: {phi_grid.shape}")
        
        # 3. Discretize to Coils
        print(f"\n[Step 3] Extracting Coil Geometry...")
        coils_2d = coils.extract_contour_paths(phi_grid, x_grid, y_grid, self.num_turns)
        print(f"  -> Extracted {len(coils_2d)} discrete loops from contours.")
        
        # Determine parity for 3D generation
        parity = -1.0 if basis_type in ['bx', 'by'] else 1.0
        
        # Calculate Current per Turn (Physical Scaling)
        # Phi represents total current stream function (Amperes).
        # We discretized it into N turns. The current in each wire is Delta_Phi.
        phi_range = phi_grid.max() - phi_grid.min()
        I_per_turn = phi_range / self.num_turns
        print(f"  -> Physical Scaling: Phi_range={phi_range:.2f} A, I_per_turn={I_per_turn:.2f} A")
        
        coils_3d = coils.generate_coil_vertices(coils_2d, 
                                                z_position=self.config.get('a', 0.7), 
                                                downsample_factor=5,
                                                current_parity=parity,
                                                current_scale=I_per_turn)
        print(f"  -> Generated 3D geometry (Top & Bottom planes) with parity {parity}.")
        
        return coils_3d

    def create_component(self, component_name: str, reg_lambda: float = None):
        '''
        Main entry point to create a coil component by name.
        '''
        name = component_name.lower()
        n_points = len(self.target_points)
        
        # Use lambda from config if not explicitly provided
        if reg_lambda is None:
            reg_lambda = self.config.get('reg_lambda', 1.6544e-20)
        
        # --- 0th Order Terms (Uniform) ---
        if name in ['bx', 'by', 'bz']:
            target = np.ones((n_points, 1)) * 50e-9
            return self._solve_and_discretize(target, reg_lambda, name)
            
        # --- 1st Order Terms (Gradients) ---
        # Future work
        
        else:
            print(f"  [Error] Unknown component: {component_name}")
            return []
