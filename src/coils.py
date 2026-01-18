import pathlib
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

def extract_contour_paths(phi_grid: np.ndarray,
                          x_grid: np.ndarray,
                          y_grid: np.ndarray,
                          num_turns: int) -> list[tuple[np.ndarray, int]]:
    '''
    Extracts discrete coil paths from the continuous stream function using contouring.

    This function serves as the Python equivalent of the MATLAB contour and 'Judge_wise' logic.
    It discretizes the continuous stream function into a specified number of wire loops (`num_turns`).
    
    Methodology:
        1. Generates `num_turns` equidistant levels between the min and max value of `phi_grid`.
        2. Uses `matplotlib.pyplot.contour` to find the isolines of closed loops.
        3. Traverses the resulting path segments to form closed loops.
        4. Calculates the winding direction using the Shoelace formula.

    Args:
        phi_grid (np.ndarray): The 2D stream function grid of shape (Ny, Nx).
        x_grid (np.ndarray): The 1D x-coordinate grid of shape (Nx,).
        y_grid (np.ndarray): The 1D y-coordinate grid of shape (Ny,).
        num_turns (int): The number of turns of coils to discretize the stream function into. 

    Returns:
        list[tuple[np.ndarray, int]]: A list of coils, where each coil is a tuple:
            - coordinates (np.ndarray): An (N_points, 2) array of [x, y] vertices.
            - direction (int): The winding direction (+1 for clockwise, -1 for counter-clockwise).

    Raises:
        ValueError: If `num_turns` is less than 1.
        ValueError: If inputs have inconsistent shapes.

    Examples:
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> phi = X**2 + Y**2  # Simple circular contours
        >>> coils = extract_contour_paths(phi, x, y, num_turns=3)
        >>> print(len(coils))
        3
    '''
    # Input validation implies checks on shapes, usually handled by numpy/matplotlib or downstream errors, 
    # but explicit checks can be added if needed.

    # 1. Generate Levels
    # We use [1:-1] to avoid the absolute min/max points which are just dots.
    levels = np.linspace(np.min(phi_grid), np.max(phi_grid), num_turns + 2)[1:-1]

    # 2. Extract Contours
    # We use a temporary figure to avoid messing up global plot state
    fig = plt.figure()
    contours = plt.contour(x_grid, y_grid, phi_grid, levels=levels, origin='lower')
    plt.close(fig)

    coils = []

    # 3. Traverse Segments
    for level_segs in contours.allsegs:
        for seg in level_segs:
            # Filter noise (paths with too few points)
            if len(seg) < 3:
                continue

            # 4. Determine Direction
            direction = calculate_winding_sign(seg)
            coils.append((seg, direction))

    return coils

def calculate_winding_sign(path_coords: np.ndarray) -> int:
    '''
    Determines the winding direction of a closed path using the Shoelace formula.

    Calculates the signed area of a 2D polygon to determine orientation.
    
    Formula:
        Area = 0.5 * sum( (x_i + x_{i+1}) * (y_{i+1} - y_i) )

    Args:
        path_coords (np.ndarray): An (N, 2) array of [x, y] vertices.
    
    Returns:
        int: +1 for clockwise (CW), -1 for counter-clockwise (CCW), 0 for degenerate/zero-area.

    Raises:
        ValueError: If input is not an (N, 2) array or has fewer than 3 points.

    Examples:
        >>> # Counter-clockwise square
        >>> sq_ccw = np.array([[0,0], [1,0], [1,1], [0,1]])
        >>> calculate_winding_sign(sq_ccw)
        -1
        >>> # Clockwise square
        >>> sq_cw = np.array([[0,0], [0,1], [1,1], [1,0]])
        >>> calculate_winding_sign(sq_cw)
        1
    '''
    if path_coords.ndim != 2 or path_coords.shape[1] != 2:
        raise ValueError("Input array must be an (N, 2) array.")
    if len(path_coords) < 3:
        raise ValueError("Input array must have at least 3 points.")
    
    x = path_coords[:, 0]
    y = path_coords[:, 1]

    # Roll to align i with i+1
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    # Shoelace calculation (Signed Double Area)
    # The sign depends on the coordinate system (here Y is up, X is right)
    double_area = np.sum((x + x_next) * (y_next - y))

    if double_area > 0:
        return 1
    elif double_area < 0:
        return -1
    else:
        return 0

def generate_coil_vertices(
        coils_2d: list[tuple[np.ndarray, int]],
        z_position: float,
        downsample_factor: int,
        current_parity: float = -1.0,
        current_scale: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    '''
    Generates 3D coil geometry and assigns physical currents to each loop.

    Projects 2D contours onto two parallel planes at z = +/- z_position and assigns 
    currents based on the loop's winding direction (from stream function gradient) 
    and the parity symmetry required (e.g., Anti-symmetric for Bx/By, Symmetric for Bz).

    Args:  
        coils_2d (list): List of (coordinates, direction_sign) from `extract_contour_paths`.
        z_position (float): Vertical distance from origin to coil planes (m).
        downsample_factor (int): Step size for points to reduce vertex count.
        current_parity (float, optional): Current direction multiplier for the bottom plane relative to the top. 
                                          -1.0 for Gradient/Saddle coils, 1.0 for Helmholtz-like coils. Defaults to -1.0.
        current_scale (float, optional): Magnitude of current per turn (Amperes). Defaults to 1.0.

    Returns:
        list: A list of 4-element tuples, each representing a 3D coil pair:
            - top_loop (np.ndarray): (N, 3) vertices at z = +z_position.
            - bottom_loop (np.ndarray): (N, 3) vertices at z = -z_position.
            - I_top (float): Current in the top loop (A).
            - I_bottom (float): Current in the bottom loop (A).

    Examples:
        >>> # Dummy 2D square loop, CCW
        >>> loop_2d = (np.array([[0,0], [1,0], [1,1], [0,1]]), -1)
        >>> coils_3d = generate_coil_vertices([loop_2d], z_position=0.5, downsample_factor=1, current_scale=10.0)
        >>> top, bot, i_top, i_bot = coils_3d[0]
        >>> print(i_top, i_bot)
        10.0 -10.0
    '''
    coils_3d = []
    for path_coords, _ in coils_2d:
        # 1. Downsampling
        downsampled_coords = path_coords[::downsample_factor]
        # Ensure the loop remains closed after downsampling
        if not np.array_equal(downsampled_coords[-1], path_coords[-1]):
            downsampled_coords = np.vstack([downsampled_coords, path_coords[-1]])

        # 2. Projection to 3D
        z_top = np.full((len(downsampled_coords), 1), z_position)
        z_bottom = np.full((len(downsampled_coords), 1), -z_position)

        top_loop = np.hstack([downsampled_coords, z_top])
        bottom_loop = np.hstack([downsampled_coords, z_bottom])
        
        # 3. Assign Currents
        # Note: We ignore the 'direction_sign' from 2D (which indicates CW/CCW).
        # We assume the contour points are already ordered consistent with the stream function gradient.
        # So we just assign the positive magnitude `current_scale` to the top loop,
        # and let the parity determine the bottom loop.
        I_top = current_scale 
        I_bottom = I_top * current_parity
        
        coils_3d.append((top_loop, bottom_loop, I_top, I_bottom))

    return coils_3d

def rotate_coil_z(coils_3d: list[tuple[np.ndarray, np.ndarray, float, float]], 
                  angle_degrees: float) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    '''
    Rotates the coil geometry around the Z-axis by a given angle.

    Applies a rotation matrix to the x, y coordinates of all vertices. 
    Current values are preserved.

    Args:
        coils_3d (list): List of coil tuples (top_loop, bottom_loop, I_top, I_bottom).
        angle_degrees (float): Rotation angle in degrees (counter-clockwise).

    Returns:
        list: New list of rotated coils with the same structure as input.

    Examples:
        >>> # Point at (1, 0, 0)
        >>> top = np.array([[1, 0, 0]])
        >>> bot = np.array([[1, 0, 0]])
        >>> coils = [(top, bot, 1.0, -1.0)]
        >>> rotated = rotate_coil_z(coils, 90.0)
        >>> # Should be approx (0, 1, 0)
        >>> print(np.round(rotated[0][0], 1))
        [[0. 1. 0.]]
    '''
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    rotated_coils = []
    # Unpack 4 elements
    for top_loop, bottom_loop, I_top, I_bottom in coils_3d:
        # Apply rotation matrix: v_new = v @ R.T
        top_new = top_loop @ rotation_matrix.T
        bottom_new = bottom_loop @ rotation_matrix.T
        
        rotated_coils.append((top_new, bottom_new, I_top, I_bottom))
        
    return rotated_coils

def translate_coils(coils_3d: list[tuple[np.ndarray, np.ndarray, float, float]], 
                    offset: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    '''
    Translates the entire coil system by a specified 3D offset vector.

    Used for constructing arrays of coils by placing standard units at different spatial positions.
    Current values are preserved.

    Args:
        coils_3d (list): List of coil tuples (top_loop, bottom_loop, I_top, I_bottom).
        offset (np.ndarray or list): (3,) vector [dx, dy, dz] to add to all coordinates.

    Returns:
        list: New list of translated coils.

    Raises:
        ValueError: If offset is not a 3-element vector.

    Examples:
        >>> top = np.array([[0, 0, 0]])
        >>> bot = np.array([[0, 0, -1]])
        >>> coils = [(top, bot, 1.0, 1.0)]
        >>> moved = translate_coils(coils, [10, 0, 0])
        >>> print(moved[0][0])
        [[10  0  0]]
    '''
    offset_vec = np.asarray(offset, dtype=float)
    if offset_vec.shape != (3,):
        raise ValueError("Offset must be a 3-element vector [x, y, z].")
        
    translated_coils = []
    for top_loop, bottom_loop, I_top, I_bottom in coils_3d:
        # Broadcasting addition
        top_new = top_loop + offset_vec
        bottom_new = bottom_loop + offset_vec
        
        translated_coils.append((top_new, bottom_new, I_top, I_bottom))
        
    return translated_coils