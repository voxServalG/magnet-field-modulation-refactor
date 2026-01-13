import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_coil_geometry_3d(coils_3d: list[tuple[np.ndarray, np.ndarray, int]], filename: str = None):
    '''
    Visualize the 3D geometry of the bi-planar coil system.

    Displays the top and bottom coil planes in a 3D coordinate system. 
    By convention, top coils are plotted in red and bottom coils in blue. 
    The function also attempts to equalize the aspect ratio of the 3D plot 
    to prevent geometric distortion.

    Args:
        coils_3d (list[tuple]): A list of 3D coils. Each item is a tuple:
            - top_loop (np.ndarray): (N, 3) array of vertices for the top plane.
            - bottom_loop (np.ndarray): (N, 3) array of vertices for the bottom plane.
            - direction_sign (int): Sign indicating the relative current direction (unused for plotting).
        filename (str, optional): Path to save the figure (e.g., 'coil.png'). 
            If None, the plot is displayed interactively using plt.show().

    Returns:
        None: The function renders the plot to a window or saves it to a file.

    Examples:
        >>> top = np.array([[0,0,1], [1,0,1], [1,1,1], [0,0,1]])
        >>> bot = np.array([[0,0,-1], [1,0,-1], [1,1,-1], [0,0,-1]])
        >>> plot_coil_geometry_3d([(top, bot, 1)])
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for top_loop, bottom_loop, _ in coils_3d:
        # Plot Top Coil (Red)
        ax.plot(top_loop[:, 0], top_loop[:, 1], top_loop[:, 2], 'r-', linewidth=1.5, alpha=0.8)
        # Plot Bottom Coil (Blue)
        ax.plot(bottom_loop[:, 0], bottom_loop[:, 1], bottom_loop[:, 2], 'b-', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Coil Geometry')
    
    # Simple trick to equalize aspect ratio for 3D plots
    # Create a cubic bounding box
    all_coords = []
    for t, b, _ in coils_3d:
        all_coords.append(t)
        all_coords.append(b)
    if all_coords:
        all_points = np.vstack(all_coords)
        max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                              all_points[:,1].max()-all_points[:,1].min(), 
                              all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
        mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_field_heatmap(field_B: np.ndarray, target_points: np.ndarray, component_idx: int = 0, filename: str = None):
    '''
    Plot a 2D heatmap of a specific magnetic field component.

    Visualizes the distribution of Bx, By, or Bz across a target region. 
    This function uses 'tricontourf' to handle potentially unstructured 
    or non-grid target points, performing automatic interpolation. 
    The field values are converted from Tesla to nanotesla (nT) for display.

    Args:
        field_B (np.ndarray): (M, 3) array of magnetic field vectors (Tesla).
        target_points (np.ndarray): (M, 3) array of evaluation point coordinates (m).
        component_idx (int, optional): The component to plot (0: Bx, 1: By, 2: Bz). Defaults to 0.
        filename (str, optional): Path to save the figure. If None, displays the plot.

    Returns:
        None: The function renders the plot to a window or saves it to a file.

    Examples:
        >>> points = np.random.rand(100, 3)
        >>> fields = np.random.rand(100, 3) * 1e-6
        >>> plot_field_heatmap(fields, points, component_idx=0)
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract coordinates
    x = target_points[:, 0]
    y = target_points[:, 1]
    z = target_points[:, 2]
    
    # Select the field component to plot
    b_comp = field_B[:, component_idx] * 1e9 # Convert to nT
    
    # We plot X vs Y (top view) - assuming Z is roughly constant or we project
    # Use tricontourf for automatic interpolation
    cntr = ax.tricontourf(x, y, b_comp, levels=20, cmap='viridis')
    
    cbar = fig.colorbar(cntr, ax=ax)
    cbar.set_label(f'B_{"xyz"[component_idx]} (nT)')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Magnetic Field Distribution (B{"xyz"[component_idx]})')
    ax.set_aspect('equal')

    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_stream_function(phi_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, filename: str = None):
    '''
    Plot the 2D stream function contour map.

    Renders a filled contour plot of the stream function $\Phi$, which represents 
    the continuous current distribution on the coil plane. It also overlays 
    discrete contour lines to visualize potential discrete wire loop paths.

    Args:
        phi_grid (np.ndarray): (Ny, Nx) array of stream function values (Amperes).
        x_grid (np.ndarray): 1D array of x-coordinates (m) or 2D meshgrid XX.
        y_grid (np.ndarray): 1D array of y-coordinates (m) or 2D meshgrid YY.
        filename (str, optional): Path to save the figure. If None, displays the plot.

    Returns:
        None: The function renders the plot to a window or saves it to a file.

    Examples:
        >>> x = np.linspace(-1, 1, 50)
        >>> y = np.linspace(-1, 1, 50)
        >>> phi = np.sin(x) * np.cos(y[:, np.newaxis])
        >>> plot_stream_function(phi, x, y)
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create meshgrid for plotting if inputs are 1D
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        XX, YY = np.meshgrid(x_grid, y_grid)
    else:
        XX, YY = x_grid, y_grid
        
    cp = ax.contourf(XX, YY, phi_grid, levels=30, cmap='RdBu_r')
    fig.colorbar(cp, ax=ax, label='Stream Function (A)')
    
    # Also plot contour lines for clarity
    ax.contour(XX, YY, phi_grid, levels=15, colors='k', linewidths=0.5, alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Stream Function & Coil Windings')
    ax.set_aspect('equal')
    
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()
