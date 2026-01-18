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

    def __init__(self, factory: CoilFactory, layout_offsets: np.ndarray):
        '''
        Initialize the array manager and pre-generate standard coil units.

        Args:
            factory (CoilFactory): Configured factory instance to generate standard units.
            layout_offsets (np.ndarray): (N_units, 3) array of center positions for each unit.
        '''
        self.factory = factory
        self.layout = np.asarray(layout_offsets)
        self.num_units = len(self.layout)
        
        print(f"[ArrayManager] Initializing with {self.num_units} units...")
        
        # Pre-generate Standard Units (at origin)
        # We ensure they are generated with a known "Unit Current" scale.
        # Note: Factory generates based on target=50nT by default now.
        # But for linearity, the absolute scale doesn't matter as long as it's consistent.
        # We will treat the factory output as our "Basis Vector".
        self.standard_units = {}
        
        for comp in ['bx', 'by', 'bz']:
            print(f"  -> Pre-generating Standard Unit: {comp.upper()}...")
            # We use a default lambda. The generated current I0 produces field B0.
            # In the linear system S * x = b, 'x' will be the multiplier relative to this I0.
            coils_list = self.factory.create_component(comp)
            if not coils_list:
                raise RuntimeError(f"Failed to generate standard unit for {comp}")
            self.standard_units[comp] = coils_list
            
        print("[ArrayManager] Initialization complete. Standard units ready.\n")

    def compute_response_matrix(self, target_points: np.ndarray) -> np.ndarray:
        '''
        Constructs the System Response Matrix S (Sensitivity Matrix for the Array).
        
        S maps the control current weights to the magnetic field at target points.
        
        Args:
            target_points (np.ndarray): (M_points, 3) coordinates of ROI.
            
        Returns:
            np.ndarray: Matrix S of shape (3 * M_points, 3 * N_units).
                        Columns are ordered as: [Unit0_Bx, Unit0_By, Unit0_Bz, Unit1_Bx, ...].
                        Rows are ordered as: [P0_x, P0_y, P0_z, P1_x, ...].
        '''
        M = len(target_points)
        K = self.num_units
        num_channels = 3 * K
        num_measurements = 3 * M
        
        S = np.zeros((num_measurements, num_channels))
        
        print(f"[ArrayManager] Computing Response Matrix S ({num_measurements}x{num_channels})...")
        start_time = time.time()
        
        # Iterate through all units and their components
        col_idx = 0
        
        for i, offset in enumerate(self.layout):
            # For each unit position
            for comp in ['bx', 'by', 'bz']:
                # 1. Get Standard Unit
                base_coils = self.standard_units[comp]
                
                # 2. Translate to Array Position
                moved_coils = coils.translate_coils(base_coils, offset)
                
                # 3. Compute Field (Forward Physics)
                # Important: use_shielding=False for free-space linear superposition
                B_vec = physics.calculate_field_from_coils(moved_coils, target_points, use_shielding=False, show_progress=False)
                
                # 4. Flatten B_vec (Mx3 -> 3M)
                # We flatten in 'C' order (row-major): [B0x, B0y, B0z, B1x, B1y, B1z...]
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
        
        Minimizes || S * x + B_bg ||^2. If region_mask is provided, only specified points are suppressed.
        
        Args:
            B_background_vec (np.ndarray): Flattened background field vector (3M,).
            S_matrix (np.ndarray): System Response Matrix (3M, 3K).
            region_mask (np.ndarray, optional): Boolean mask of shape (M,) for target points to include.
            
        Returns:
            tuple: (optimal_weights, residuals)
        '''
        print("[ArrayManager] Solving Least Squares Optimization...")
        
        rhs = -B_background_vec
        
        # Apply mask if provided
        if region_mask is not None:
            # Expand point mask (M,) to component mask (3M,)
            # [P0, P1...] -> [P0x, P0y, P0z, P1x, P1y, P1z...]
            full_mask = np.repeat(region_mask, 3)
            S_active = S_matrix[full_mask, :]
            rhs_active = rhs[full_mask]
            print(f"  -> Regional Suppression active: using {np.sum(region_mask)}/{len(region_mask)} points.")
        else:
            S_active = S_matrix
            rhs_active = rhs
        
        # Use numpy's lstsq solver
        x_opt, residuals, rank, s = np.linalg.lstsq(S_active, rhs_active, rcond=None)
        
        print(f"  -> Solved. Rank: {rank}/{min(S_active.shape)}")
        return x_opt, residuals

    def get_final_system(self, optimal_weights: np.ndarray) -> tuple[list, list]:
        '''
        Constructs the final physical coil system and associated colors.
        
        Args:
            optimal_weights (np.ndarray): (3K,) vector of current multipliers.
            
        Returns:
            tuple:
                - final_system (list): List of 3D coils.
                - colors (list): List of color strings corresponding to each coil segment.
        '''
        final_system = []
        colors = []
        col_idx = 0
        
        # Color map for Bx, By, Bz
        comp_colors = {'bx': 'r', 'by': 'g', 'bz': 'b'}
        
        for i, offset in enumerate(self.layout):
            for comp in ['bx', 'by', 'bz']:
                weight = optimal_weights[col_idx]
                col_idx += 1
                
                # Skip inactive coils
                if abs(weight) < 1e-9:
                    continue
                
                base_coils = self.standard_units[comp]
                moved_coils = coils.translate_coils(base_coils, offset)
                
                for top, bot, it, ib in moved_coils:
                    final_system.append((top, bot, it * weight, ib * weight))
                    colors.append(comp_colors[comp])
                
        print(f"[ArrayManager] Assembled final system with {len(final_system)} active coil segments.")
        return final_system, colors
