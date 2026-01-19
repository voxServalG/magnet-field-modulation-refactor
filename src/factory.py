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

    # Define Recipes for 1st Order Gradients (Scheme B)
    # Each entry defines: Fourier Primitive, Z-Parity, Primary Component, and Target Formula
    _GRADIENT_RECIPES = {
        'gxx': {'basis': 'sc', 'parity': -1.0, 'comp': 'x', 'func': lambda p: p[:, 0]}, # Bx = x
        'gyy': {'basis': 'cs', 'parity': -1.0, 'comp': 'y', 'func': lambda p: p[:, 1]}, # By = y
        'gxy': {'basis': 'ss', 'parity': -1.0, 'comp': 'x', 'func': lambda p: p[:, 1]}, # Bx = y
        'gxz': {'basis': 'sc', 'parity': 1.0,  'comp': 'x', 'func': lambda p: p[:, 2]}, # Bx = z
        'gyz': {'basis': 'cs', 'parity': 1.0,  'comp': 'y', 'func': lambda p: p[:, 2]}, # By = z
    }

    def _get_matrices(self, basis_type: str, parity: float, target_comp: str):
        '''Lazy loader for matrices with parity/component support'''
        cache_key = (basis_type, parity, target_comp)
        if cache_key in self.matrix_cache:
            return self.matrix_cache[cache_key]
        
        print(f"[Factory] Generating matrices for {basis_type.upper()} [Parity={parity}, Comp={target_comp}]...")
        A = self.generator.compute_A(self.target_points, self.modes, basis_type, 
                                     explicit_parity=parity, target_component=target_comp)
        Gamma = self.generator.compute_Gamma(self.modes, basis_type)
        
        self.matrix_cache[cache_key] = (A, Gamma)
        return A, Gamma

    def _solve_and_discretize(self, target_field_vec, reg_lambda, component_label, 
                              basis_type=None, parity=None, target_comp=None):
        label = component_label.lower()
        
        # Determine parameters if not provided (fallback to legacy defaults)
        if basis_type is None:
            basis_type = label # 'bx', 'by', 'bz'
        if parity is None:
            parity = -1.0 if basis_type in ['bx', 'by', 'sc', 'cs', 'ss'] else 1.0
        if target_comp is None:
            target_comp = 'x' if 'x' in basis_type else ('y' if 'y' in basis_type else 'z')

        A, Gamma = self._get_matrices(basis_type, parity, target_comp)
            
        # 1. Solve Inverse Problem
        print(f"\n[Step 1] Solving Inverse Problem for {label.upper()}...")
        C_coeffs = solver.solve_stream_function_coeffs(A, Gamma, reg_lambda, target_field_vec)
        
        # 2. Reconstruct Stream Function
        print(f"\n[Step 2] Reconstructing Stream Function...")
        x_grid = np.linspace(-self.L, self.L, self.grid_res)
        y_grid = np.linspace(-self.L, self.L, self.grid_res)
        phi_grid = solver.reconstruct_stream_function(C_coeffs, x_grid, y_grid, self.L, self.modes, coil_type=basis_type)
        
        # 3. Discretize to Coils
        print(f"\n[Step 3] Extracting Coil Geometry...")
        coils_2d = coils.extract_contour_paths(phi_grid, x_grid, y_grid, self.num_turns)
        
        phi_range = phi_grid.max() - phi_grid.min()
        I_per_turn = phi_range / self.num_turns
        
        coils_3d = coils.generate_coil_vertices(coils_2d, 
                                                z_position=self.a, 
                                                downsample_factor=5,
                                                current_parity=parity,
                                                current_scale=I_per_turn)
        print(f"  -> Generated {label.upper()} 3D geometry with parity {parity}.")
        
        return coils_3d

    def create_component(self, component_name: str, reg_lambda: float = None):
        '''
        Main entry point to create a coil component by name.
        '''
        name = component_name.lower()
        n_points = len(self.target_points)
        if reg_lambda is None:
            reg_lambda = self.config.get('reg_lambda', 1.6544e-20)
        
        # --- 0th Order Terms (Uniform) ---
        if name in ['bx', 'by', 'bz']:
            target = np.ones((n_points, 1)) * 50e-9 # Standard 50nT target
            return self._solve_and_discretize(target, reg_lambda, name)
            
        # --- 1st Order Terms (Gradients - Scheme B) ---
        elif name in self._GRADIENT_RECIPES:
            recipe = self._GRADIENT_RECIPES[name]
            # Generate linear target field (e.g. Bx = x)
            # We use a unit gradient (1 nT/m) for design
            raw_target = recipe['func'](self.target_points)
            target = raw_target.reshape(-1, 1) * 1e-9 # 1 nT/m gradient
            
            return self._solve_and_discretize(target, reg_lambda, name,
                                              basis_type=recipe['basis'],
                                              parity=recipe['parity'],
                                              target_comp=recipe['comp'])
        
        else:
            print(f"  [Error] Unknown component: {component_name}")
            return []
