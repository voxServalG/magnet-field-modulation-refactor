import numpy as np
import pathlib
import matplotlib.pyplot as plt
import sys
import time

# Ensure src is in python path
current_dir = pathlib.Path(__file__).parent
sys.path.append(str(current_dir))

from src.factory import CoilFactory
from src.array_manager import ArrayActiveShielding
from src import physics

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Simulation
DT = 0.01          # 10ms time step (100Hz loop)
DURATION = 5.0     # 5 seconds
TIME_STEPS = int(DURATION / DT)
T_AXIS = np.linspace(0, DURATION, TIME_STEPS)

# System
ARRAY_REG_LAMBDA = 2.3e-16
SHIELD_DIMS = (2.4, 1.9, 1.65)

# Motion Area (The "Stage")
STAGE_LIMIT = 0.3  # +/- 30cm movement range
STAGE_RES = 15     # 15x15x15 grid (approx 4cm spacing). 
                   # Total points ~3375. Manageable for pre-calc.

# The Head (ROI)
HEAD_RADIUS = 0.10 # 10cm radius sphere representing the brain area

# ---------------------------------------------------------
# Motion Profiles
# ---------------------------------------------------------
def get_head_center_position(t, mode='nod'):
    """
    Returns the [x, y, z] center of the head at time t.
    """
    if mode == 'nod':
        # Simple nod: Z-axis oscillation + slight Y drift
        # Amp 10cm, Freq 1Hz
        z = 0.10 * np.sin(2 * np.pi * 1.0 * t)
        y = 0.05 * np.sin(2 * np.pi * 0.5 * t)
        x = 0.0
        return np.array([x, y, z])
    elif mode == 'shake':
        # Fast shake "No": X-axis oscillation
        # Amp 15cm, Freq 2Hz (vigorous)
        x = 0.15 * np.sin(2 * np.pi * 2.0 * t)
        return np.array([x, 0.0, 0.0])
    return np.array([0., 0., 0.])

# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------
def run_motion_sensitivity_test():
    print("==========================================================")
    print("   Project Nuke: Motion & Latency Sensitivity Test (MVP)  ")
    print("==========================================================")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir / "results" / f"motion_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Array
    # --------------
    print(f"[Init] Initializing Array...")
    config = {
        'L': 0.6, 'a': 0.7, 'modes': (4, 4), 'reg_lambda': 1e-14, 'num_turns': 10, 
        'grid_res': 60, 'shield_dims': SHIELD_DIMS
    }
    factory = CoilFactory(config)
    s = 1.25
    layout = np.array([[x, y, 0.0] for x in [-s,0,s] for y in [-s,0,s]])
    manager = ArrayActiveShielding(factory, layout, use_gradients=True, shield_dims=SHIELD_DIMS)

    # 2. Pre-compute The "Stage" (Global Grid)
    # ----------------------------------------
    # We pre-calculate S for the entire volume where the head MIGHT go.
    # This allows O(1) lookups during the time loop.
    print(f"[Init] Pre-computing Global Stage Grid ({STAGE_LIMIT*2}m box)...")
    
    gx = np.linspace(-STAGE_LIMIT, STAGE_LIMIT, STAGE_RES)
    GX, GY, GZ = np.meshgrid(gx, gx, gx, indexing='ij')
    stage_points = np.column_stack((GX.flatten(), GY.flatten(), GZ.flatten()))
    
    # Pre-compute S_global
    # shape: (3*N_stage, N_coils)
    S_global = manager.compute_response_matrix(stage_points)
    
    # Pre-compute Background Field on Stage
    # (Using a simple gradient for this test)
    # B = B0 + G*x
    print("[Init] generating background field map...")
    B0 = np.array([0.0, 0.0, 150e-9]) # 150nT Bz
    Grad = 50e-9 # 50nT/m gradient
    
    # B_bg = B0 + z * Grad (simple linear gradient)
    B_bg_stage = np.zeros_like(stage_points)
    B_bg_stage += B0
    B_bg_stage[:, 2] += stage_points[:, 0] * Grad # dBz/dx
    B_bg_stage_flat = B_bg_stage.flatten()
    
    # 3. Simulation Loop: The "Laggy Tracker"
    # ---------------------------------------
    # We will test different Latency values
    latencies_to_test = [0, 2, 5, 10, 20] # frames (x10ms -> 0, 20, 50, 100, 200 ms)
    
    results_summary = []

    plt.figure(figsize=(10, 6))
    
    for lat_frames in latencies_to_test:
        latency_ms = lat_frames * DT * 1000
        print(f"\n>>> Running Sim: Latency = {latency_ms:.0f} ms ({lat_frames} frames)")
        
        rms_history = []
        
        # Position Buffer for Delay Simulation
        # Initialize with t=0 position
        pos_history = [get_head_center_position(0, 'shake')] * (lat_frames + 1)
        
        for t_step in range(TIME_STEPS):
            t_now = T_AXIS[t_step]
            
            # A. The Reality
            # --------------
            head_pos_real = get_head_center_position(t_now, 'shake')
            
            # Store real pos to history
            pos_history.append(head_pos_real)
            
            # B. The Sensor (Delayed)
            # -----------------------
            # What the system "thinks" the head is at
            # Read from history with delay offset
            # History length grows, we want index: current - latency
            head_pos_sensed = pos_history[-(lat_frames + 1)]
            
            # C. The Solver (Based on Sensed Pos)
            # -----------------------------------
            # Identify grid points closest to SENSED head position
            # (Simple nearest neighbor approximation for MVP)
            dist_sensed = np.linalg.norm(stage_points - head_pos_sensed, axis=1)
            mask_sensed = dist_sensed < HEAD_RADIUS
            
            if np.sum(mask_sensed) == 0:
                # Fallback if head goes out of grid (shouldn't happen with correct setup)
                x_opt = np.zeros(manager.total_ch_per_unit * manager.num_units)
            else:
                # Solve using S_global rows corresponding to masked area
                full_mask_sensed = np.repeat(mask_sensed, 3)
                S_active = S_global[full_mask_sensed, :]
                b_active = B_bg_stage_flat[full_mask_sensed]
                
                # Fast Tikhonov
                # x = (S'S + lam*I)^-1 S' b
                # Note: For speed in loop, we normally pre-factor. 
                # But here S_active changes shape every frame!
                # We use lstsq for robustness or simple solve.
                
                # To speed up MVP, assume standard solve
                STS = S_active.T @ S_active
                STb = S_active.T @ (-b_active)
                x_opt = np.linalg.solve(STS + ARRAY_REG_LAMBDA * np.eye(STS.shape[0]), STb)
            
            # D. The Consequence (Applied to Real Pos)
            # ----------------------------------------
            # System outputs current 'x_opt'.
            # We measure field at REAL head position.
            dist_real = np.linalg.norm(stage_points - head_pos_real, axis=1)
            mask_real = dist_real < HEAD_RADIUS
            
            # Calculate actual field
            # B_total = S * x + B_bg
            # But only look at real mask
            full_mask_real = np.repeat(mask_real, 3)
            
            if np.sum(mask_real) == 0:
                # Head out of bounds
                rms_val = np.nan
            else:
                S_real_view = S_global[full_mask_real, :]
                B_bg_real_view = B_bg_stage_flat[full_mask_real]
                
                B_coil_real = S_real_view @ x_opt
                B_resid_real = B_coil_real + B_bg_real_view
                
                # Calc RMS
                rms_val = np.sqrt(np.mean(B_resid_real**2)) * 1e9 # nT
            
            rms_history.append(rms_val)
            
        # Plot curve for this latency
        plt.plot(T_AXIS, rms_history, label=f'Delay {latency_ms:.0f}ms')
        
        mean_rms = np.nanmean(rms_history)
        max_rms = np.nanmax(rms_history)
        results_summary.append({
            'Latency_ms': latency_ms,
            'Mean_RMS_nT': mean_rms,
            'Max_RMS_nT': max_rms
        })
        print(f"  -> Result: Mean RMS = {mean_rms:.2f} nT, Max = {max_rms:.2f} nT")

    # Finalize Plot (Publication Quality)
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10
    })
    
    plt.title("Motion Robustness: Residual Field vs Tracking Latency", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Residual RMS (nT)")
    plt.legend(frameon=True, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(output_dir / "latency_impact.svg")
    plt.savefig(output_dir / "latency_impact.pdf") # PDF for LaTeX
    plt.close()
    
    # Save Summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Motion Sensitivity Test - {timestamp}\n")
        f.write("=======================================\n")
        f.write(f"Lambda: {ARRAY_REG_LAMBDA}\n")
        f.write(f"Motion: 2Hz Shake (+/- 15cm)\n\n")
        f.write("Latency(ms) | Mean RMS(nT) | Max RMS(nT)\n")
        f.write("----------------------------------------\n")
        for res in results_summary:
            f.write(f"{res['Latency_ms']:11.1f} | {res['Mean_RMS_nT']:12.2f} | {res['Max_RMS_nT']:11.2f}\n")
            
    print(f"\n[Done] Results saved to {output_dir}")

if __name__ == "__main__":
    run_motion_sensitivity_test()
