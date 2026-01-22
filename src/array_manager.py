import numpy as np
import time
from src import coils, physics
from src.factory import CoilFactory

class ArrayActiveShielding:
    '''
    Manages the simulation and optimization of a multi-unit active magnetic shielding array.
    
    This class handles:
    1. Pre-generation of standard coil units (Bx, By, Bz).
    2. Construction of the System Response Matrix (S) for an arbitrary array layout.
    3. Solving the inverse problem to find optimal currents for background field suppression.
    '''

    def __init__(self, factory: CoilFactory, layout_offsets: np.ndarray, use_gradients: bool = False, shield_dims: tuple = None):
        '''
        Initialize the array manager and pre-generate standard coil units.

        Args:
            factory (CoilFactory): Configured factory instance to generate standard units.
            layout_offsets (np.ndarray): (N_units, 3) array of center positions for each unit.
            use_gradients (bool): If True, include Gradient coils (Gxx, Gyy, Gxy). Gxz/Gyz are removed as they are synthesized by Bx/By split control.
            shield_dims (tuple): Optional (x, y, z) half-dimensions of shielding room. If set, enables shielding reflections.
        '''
        self.factory = factory
        self.layout = np.asarray(layout_offsets)
        self.num_units = len(self.layout)
        self.use_gradients = use_gradients
        self.shield_dims = shield_dims
        
        # Define active channels (Base types)
        # Note: Gxz and Gyz are REMOVED because independent Top/Bot control of Bx/By can synthesize them.
        self.base_channels = ['bx', 'by', 'bz']
        if use_gradients:
            self.base_channels += ['gxx', 'gyy', 'gxy']
        
        # We will split each base channel into Top and Bot independent channels.
        # Total channels per unit = len(base_channels) * 2.
        self.num_base_ch = len(self.base_channels)
        self.total_ch_per_unit = self.num_base_ch * 2
        
        print(f"[ArrayManager] Initializing with {self.num_units} units.")
        print(f"  -> Base Channels: {self.base_channels} ({self.num_base_ch} types)")
        print(f"  -> Independent Control: Top/Bot Split -> {self.total_ch_per_unit} channels/unit.")
        
        if self.shield_dims:
            print(f"[ArrayManager] Shielding Reflections ENABLED. Room dims: {self.shield_dims}")
        
        # Pre-generate Standard Units (at origin)
        self.standard_units = {}
        for comp in self.base_channels:
            print(f"  -> Pre-generating Standard Unit: {comp.upper()}...")
            coils_list = self.factory.create_component(comp)
            if not coils_list:
                raise RuntimeError(f"Failed to generate standard unit for {comp}")
            self.standard_units[comp] = coils_list
            
        print("[ArrayManager] Initialization complete. Standard units ready.\n")

    def compute_response_matrix(self, target_points: np.ndarray) -> np.ndarray:
        '''
        Constructs the System Response Matrix S (Sensitivity Matrix for the Array).
        
        Splits each coil type into two independent columns: [Top_Response, Bot_Response].
        '''
        M = len(target_points)
        K = self.num_units
        num_channels = self.total_ch_per_unit * K
        num_measurements = 3 * M
        
        S = np.zeros((num_measurements, num_channels))
        
        print(f"[ArrayManager] Computing Response Matrix S ({num_measurements}x{num_channels})...")
        print("  -> Strategy: Independent Plate Control (Top/Bot Split)")
        start_time = time.time()
        
        col_idx = 0
        
        # Determine shielding config
        use_shielding = (self.shield_dims is not None)
        
        for i, offset in enumerate(self.layout):
            for comp in self.base_channels:
                base_coils = self.standard_units[comp]
                moved_coils = coils.translate_coils(base_coils, offset)
                
                # --- Split Logic ---
                # We need to compute response for Top Only and Bot Only.
                
                # 1. Top Channel (Bot Current = 0)
                coils_top_only = []
                for top, bot, it, ib in moved_coils:
                    coils_top_only.append((top, bot, it, 0.0)) # I_bot = 0
                
                if use_shielding:
                    B_top = physics.calculate_field_from_coils(coils_top_only, target_points, 
                                                               use_shielding=True, shield_dims=self.shield_dims, show_progress=False)
                else:
                    B_top = physics.calculate_field_from_coils(coils_top_only, target_points, 
                                                               use_shielding=False, show_progress=False)
                S[:, col_idx] = B_top.flatten()
                col_idx += 1
                
                # 2. Bottom Channel (Top Current = 0)
                coils_bot_only = []
                for top, bot, it, ib in moved_coils:
                    coils_bot_only.append((top, bot, 0.0, ib)) # I_top = 0
                
                if use_shielding:
                    B_bot = physics.calculate_field_from_coils(coils_bot_only, target_points, 
                                                               use_shielding=True, shield_dims=self.shield_dims, show_progress=False)
                else:
                    B_bot = physics.calculate_field_from_coils(coils_bot_only, target_points, 
                                                               use_shielding=False, show_progress=False)
                S[:, col_idx] = B_bot.flatten()
                col_idx += 1
                
            if (i + 1) % max(1, K // 5) == 0:
                print(f"  -> Processed Unit {i+1}/{K}...")
                
        elapsed = time.time() - start_time
        print(f"  -> Matrix S computed in {elapsed:.2f}s. Shape: {S.shape}")
        
        return S

    def solve_optimization(self, B_background_vec: np.ndarray, S_matrix: np.ndarray, 
                           region_mask: np.ndarray = None, 
                           reg_lambda: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        '''
        Solves for optimal currents to suppress the background field using Tikhonov Regularization.
        
        Minimizes || S * x + B_bg ||^2 + lambda * || x ||^2.
        
        Args:
            B_background_vec: Flattened background field (3M,).
            S_matrix: Response matrix (3M, 3K).
            region_mask: Boolean mask (M,).
            reg_lambda: Penalty for current magnitude (L2 norm).
            
        Returns:
            tuple: (optimal_weights, residuals)
        '''
        print(f"[ArrayManager] Solving Tikhonov Optimization (lambda={reg_lambda:.2e})...")
        rhs = -B_background_vec
        
        if region_mask is not None:
            full_mask = np.repeat(region_mask, 3)
            S_active = S_matrix[full_mask, :]
            rhs_active = rhs[full_mask]
            print(f"  -> Regional Suppression active: using {np.sum(region_mask)}/{len(region_mask)} points.")
        else:
            S_active = S_matrix
            rhs_active = rhs
        
        # Tikhonov Normal Equation: (S'S + lambda*I) x = S'b
        STS = S_active.T @ S_active
        STb = S_active.T @ rhs_active
        
        # Add regularization to diagonal
        num_ch = STS.shape[0]
        lhs = STS + reg_lambda * np.eye(num_ch)
        
        # Solve linear system
        x_opt = np.linalg.solve(lhs, STb)
        
        # Compute residuals manually for consistency
        residual_vec = S_active @ x_opt - rhs_active
        residuals = np.sum(residual_vec**2)
        
        print(f"  -> Solved. Mean current weight: {np.mean(np.abs(x_opt)):.4f}")
        return x_opt, residuals

    def get_final_system(self, optimal_weights: np.ndarray) -> tuple[list, list]:
        '''
        Constructs the final physical coil system and associated colors.
        
        Re-assembles the split Top/Bot channels into physical coils.
        '''
        final_system = []
        colors = []
        col_idx = 0
        
        # Color map
        comp_colors = {
            'bx': 'r', 'by': 'g', 'bz': 'b',
            'gxx': 'orange', 'gyy': 'lime', 'gxy': 'cyan'
        }
        
        for i, offset in enumerate(self.layout):
            for comp in self.base_channels:
                # Retrieve weights for Top and Bot channels
                w_top = optimal_weights[col_idx]
                col_idx += 1
                w_bot = optimal_weights[col_idx]
                col_idx += 1
                
                # If both are negligible, skip
                if abs(w_top) < 1e-9 and abs(w_bot) < 1e-9:
                    continue
                
                base_coils = self.standard_units[comp]
                moved_coils = coils.translate_coils(base_coils, offset)
                
                for top, bot, it_base, ib_base in moved_coils:
                    # Final Top Current = it_base * w_top
                    # Final Bot Current = ib_base * w_bot
                    
                    final_it = it_base * w_top
                    final_ib = ib_base * w_bot
                    
                    final_system.append((top, bot, final_it, final_ib))
                    colors.append(comp_colors.get(comp, 'k'))
                
        print(f"[ArrayManager] Assembled final system with {len(final_system)} active coil segments.")
        return final_system, colors
