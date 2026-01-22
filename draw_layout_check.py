import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_layout():
    # Parameters from simulation
    L_unit = 0.6  # Half-width
    s_spacing = 1.25
    offsets = [-s_spacing, 0, s_spacing]
    
    # Shield Room
    room_dim = (2.4, 1.9) # Half dims (x, y)
    L_room = 2.4 * 2
    W_room = 1.9 * 2
    
    # Target
    r_spot = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Draw Shield Room
    rect_room = patches.Rectangle((-2.4, -1.9), 4.8, 3.8, 
                                  linewidth=3, edgecolor='black', facecolor='none', linestyle='--', label='Shielding Room Wall')
    ax.add_patch(rect_room)
    
    # 2. Draw 3x3 Array
    # Unit size = 2*L = 1.2m
    unit_w = 2 * L_unit
    
    for i, x_c in enumerate(offsets):
        for j, y_c in enumerate(offsets):
            # Bottom-left corner of the unit
            x0 = x_c - L_unit
            y0 = y_c - L_unit
            
            # Color code based on position
            color = 'blue' if (x_c==0 and y_c==0) else 'gray'
            alpha = 0.8 if (x_c==0 and y_c==0) else 0.3
            label = 'Coil Unit (1.2x1.2m)' if (i==0 and j==0) else None
            
            rect = patches.Rectangle((x0, y0), unit_w, unit_w, 
                                     linewidth=2, edgecolor=color, facecolor=color, alpha=0.2, label=label)
            ax.add_patch(rect)
            
            # Center marker
            ax.plot(x_c, y_c, '+', color=color, markersize=10)
            
    # 3. Draw Target
    circle = patches.Circle((0, 0), r_spot, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='Target ROI (R=0.25m)')
    ax.add_patch(circle)
    
    # Aesthetics
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"System Layout Top View (3x3 Array inside {L_room}x{W_room}m Room)")
    ax.legend(loc='upper right')
    
    # Annotations
    ax.annotate(f'Gap: {(s_spacing - 2*L_unit)*100:.1f} cm', xy=(0.6, 0), xytext=(0.8, 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    # Save
    plt.savefig('results/7_system_layout_2d.svg', bbox_inches='tight')
    print("Saved layout drawing to results/7_system_layout_2d.svg")

if __name__ == "__main__":
    draw_layout()
