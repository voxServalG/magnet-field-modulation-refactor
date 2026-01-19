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

    def __init__(self, factory: CoilFactory, layout_offsets: np.ndarray, use_gradients: bool = False):
        '''
        Initialize the array manager and pre-generate standard coil units.

        Args:
            factory (CoilFactory): Configured factory instance to generate standard units.
            layout_offsets (np.ndarray): (N_units, 3) array of center positions for each unit.
            use_gradients (bool): If True, include 5 types of 1st-order gradient coils per unit.
        '''
        self.factory = factory
        self.layout = np.asarray(layout_offsets)
        self.num_units = len(self.layout)
        self.use_gradients = use_gradients
        
        # Define active channels
        self.channels = ['bx', 'by', 'bz']
        if use_gradients:
            self.channels += ['gxx', 'gyy', 'gxy', 'gxz', 'gyz']
        
        num_ch = len(self.channels)
        print(f"[ArrayManager] Initializing with {self.num_units} units ({num_ch} channels/unit)...")
        
        # Pre-generate Standard Units (at origin)
        self.standard_units = {}
        for comp in self.channels:
            print(f"  -> Pre-generating Standard Unit: {comp.upper()}...")
            coils_list = self.factory.create_component(comp)
            if not coils_list:
                raise RuntimeError(f"Failed to generate standard unit for {comp}")
            self.standard_units[comp] = coils_list
            
        print("[ArrayManager] Initialization complete. Standard units ready.\n")

    def compute_response_matrix(self, target_points: np.ndarray) -> np.ndarray:
        '''
        Constructs the System Response Matrix S (Sensitivity Matrix for the Array).
        '''
        M = len(target_points)
        K = self.num_units
        num_ch_per_unit = len(self.channels)
        num_channels = num_ch_per_unit * K
        num_measurements = 3 * M
        
        S = np.zeros((num_measurements, num_channels))
        
        print(f"[ArrayManager] Computing Response Matrix S ({num_measurements}x{num_channels})...")
        start_time = time.time()
        
        col_idx = 0
        for i, offset in enumerate(self.layout):
            for comp in self.channels:
                base_coils = self.standard_units[comp]
                moved_coils = coils.translate_coils(base_coils, offset)
                
                # Compute Field
                B_vec = physics.calculate_field_from_coils(moved_coils, target_points, use_shielding=False, show_progress=False)
                S[:, col_idx] = B_vec.flatten()
                col_idx += 1
                
            if (i + 1) % max(1, K // 5) == 0:
                print(f"  -> Processed Unit {i+1}/{K}...")
                
        elapsed = time.time() - start_time
        print(f"  -> Matrix S computed in {elapsed:.2f}s. Shape: {S.shape}")
        
        return S

    def solve_optimization(self, B_background_vec: np.ndarray, S_matrix: np.ndarray, region_mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Solves for optimal currents to suppress the background field.
        '''
        print("[ArrayManager] Solving Least Squares Optimization...")
        rhs = -B_background_vec
        
        if region_mask is not None:
            full_mask = np.repeat(region_mask, 3)
            S_active = S_matrix[full_mask, :]
            rhs_active = rhs[full_mask]
            print(f"  -> Regional Suppression active: using {np.sum(region_mask)}/{len(region_mask)} points.")
        else:
            S_active = S_matrix
            rhs_active = rhs
        
        x_opt, residuals, rank, s = np.linalg.lstsq(S_active, rhs_active, rcond=None)
        
        print(f"  -> Solved. Rank: {rank}/{min(S_active.shape)}")
        return x_opt, residuals

    def get_final_system(self, optimal_weights: np.ndarray) -> tuple[list, list]:
        '''
        Constructs the final physical coil system and associated colors.
        '''
        final_system = []
        colors = []
        col_idx = 0
        
        # Color map for Bx, By, Bz and Gradients
        comp_colors = {
            'bx': 'r', 'by': 'g', 'bz': 'b',
            'gxx': 'orange', 'gyy': 'lime', 'gxy': 'cyan', 'gxz': 'magenta', 'gyz': 'yellow'
        }
        
        for i, offset in enumerate(self.layout):
            for comp in self.channels:
                weight = optimal_weights[col_idx]
                col_idx += 1
                
                if abs(weight) < 1e-9:
                    continue
                
                base_coils = self.standard_units[comp]
                moved_coils = coils.translate_coils(base_coils, offset)
                
                for top, bot, it, ib in moved_coils:
                    final_system.append((top, bot, it * weight, ib * weight))
                    colors.append(comp_colors.get(comp, 'k'))
                
        print(f"[ArrayManager] Assembled final system with {len(final_system)} active coil segments.")
        return final_system, colors
