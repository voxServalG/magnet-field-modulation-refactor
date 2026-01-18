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

    This fuction serves as the Python equivalent of the MATLAB contour and 'Judge_wise' logic.
    This fuction discretizes the continuous stream function into a specified number of wire loops(`num_turns`).
    Methodology:
        1. Generates `num_turns` equidistant levels between the min and max value of `phi_grid`.
        2. Uses `matplotlib.pyplot.contour` to find the isolines of closed loops.
        3. Traverses the resulting path segments to form closed loops.
        4. Calculates the winding direction (clockwise or counter-clockwise) for each loop to assign the correct current orientation (+1 for clockwise, -1 for counter-clockwise).
    
    Args:
        `phi_grid` (np.ndarray): The 2D stream function grid of shape (Ny, Nx).
        `x_grid` (np.ndarray): The 1D x-coordinate grid of shape (Nx,).
        `y_grid` (np.ndarray): The 1D y-coordinate grid of shape (Ny,).
        `num_turns` (int): The number of turns of coils to discretize the stream function into. 

    Returns:
        list[tuple[np.ndarray, int]]: A list of coils, where each coil is a tuple:
            - coordinates(np.ndarray): An (N_points, 2) array of [x, y] vertices where N_points is the number of points in this coil.
            - direction(int): The current direction sign (+1 for clockwise, -1 for counter-clockwise).
    
    Raises:
        ValueError: If `num_turns` is less than 1.
        ValueError: If `phi_grid` is not a 2D array.
        ValueError: If `x_grid` or `y_grid` is not a 1D array.
    '''

    # Generates `num_turns` equidistant levels between the min and max value of `phi_grid`.
    '''
    为什么是 + 2？
    如果你想在线圈板上切出 22 条线：
   如果你直接 np.linspace(min, max, 22)：
       * 第一条线正好在最小值点（可能只是一个极小的点，不是圈）。
       * 最后一条线正好在最大值点（同理，可能只是一个孤立的极值点）。
   如果你生成 24 个点（num_turns + 2），你就拥有了 22 个处在中间的切片。

  2. 为什么是 [1:-1]？
  在 Python 的切片语法中，[1:-1] 的意思是：“去掉第一个，去掉最后一个，保留剩下的。”

   * 去掉第 0 个（最小值）：避免生成一个几乎看不见的、缩成一个点的无效线圈。
   * 去掉最后一个（最大值）：避免在流函数顶峰处生成一个点。

  3. 物理意义
  通过这种方式，你确保了这 22 条等值线都在 $\Phi$ 值变化比较平缓的区间内，生成的线圈形状更稳定、更清晰。

  例子：
  假设 $\Phi$ 范围是 0 到 100，你要 3 条线。
   * np.linspace(0, 100, 5) -> [0, 25, 50, 75, 100]
   * [1:-1] -> [25, 50, 75]
   * 这三条线完美地分布在海拔 25, 50, 75 的地方，避开了地势最低的 0 和最高的 100。

    '''
    levels = np.linspace(np.min(phi_grid), np.max(phi_grid), num_turns + 2)[1:-1]

    # Uses `matplotlib.pyplot.contour` to find the isolines of closed loops.
    plt.figure()
    contours = plt.contour(x_grid, y_grid, phi_grid, levels=levels, origin='lower')
    plt.close()

    # Traverses the resulting path segments to form closed loops.
    coils = []

    # 遍历每一层
    for level_segs in contours.allsegs:
        # 遍历该层的每条线
        for seg in level_segs:
            # filter segment with too less points
            if len(seg) < 3:
                continue

            # call `calculate_winding_sign` to determine sign of current.
            direction = calculate_winding_sign(seg)

            # 保存线圈的坐标和方向
            coils.append((seg, direction))

    plt.close()

    return coils

def calculate_winding_sign(path_coords: np.ndarray) -> int:

    '''
    Determines the winding direction of a closed path using the Shoelace formula.

    This function implements the core logic of the MATLAB `Judge_wise.m` script.
    It calculates the signed area of a 2D polygon.
    The sign of the area indicates whether the path is traced clockwise or counter-clockwise.

    Formula:
        Area = 0.5 * sum( (x_i + x_{i+1}) * (y_{i+1} - y_i) )

    Interpretation:
        * If Area > 0, the path is traced clockwise.
        * If Area < 0, the path is traced counter-clockwise.
        * If Area = 0, the path is self-intersecting or is not a closed loop.

    Args:
        path_coords (np.ndarray): An (N_points, 2) array of [x, y] vertices where N_points is the number of points in the path.
    
    Returns:
        int: The winding direction sign (+1 for clockwise, -1 for counter-clockwise and 0 for zero-area).

    Raises:
        ValueError: 
            - If `path_coords` does not have exactly 2 columns (x, y).
            - If `path_coords` has fewer than 3 points. 
        TypeError:
            - If `path_coords` is not a 2D numpy array or similar numeric container.
    '''

    if path_coords.ndim != 2 or path_coords.shape[1] != 2:
        raise ValueError("Input array must be an (N, 2) array.")
    if len(path_coords) < 3:
        raise ValueError("Input array must have at least 3 points.")
    
    # 提取x y分量
    # 利用 np.roll 将坐标数组循环左移一位，获取每个点的“下一点”坐标。
    # 这样 x_next[i] 实际上就是 x[i+1]，y_next[i]实际上就是y[i+1]。
    x = path_coords[:, 0]
    y = path_coords[:, 1]

    # 产生滚动
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    # Shoelace calculate
    # 向量计算是如此的，不要保持item-wise想法。
    # sum是最后才进行的，那之前我们应该拼好一个正确的被求和项。
    # 我们应该把x+x_next的每一项都拿出来，然后和y_next-y对应项相乘。
    # 这样，我们就得到了一个向量，每一项都是一个被求和项。
    # 最后，我们把这个向量的每一项都加起来，就得到了最终的结果。
    # No need to mind the shit 0.5
    double_area = np.sum((x + x_next) * (y_next-y))

   
    return 1 if double_area > 0 else -1 if double_area < 0 else 0


def generate_coil_vertices(
        coils_2d: list[tuple[np.ndarray, int]],
        z_position: float,
        downsample_factor: int,
        current_parity: float = -1.0,
        current_scale: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    '''
    Generates 3d coil vertices and assigns physical currents to each loop.

    Args:  
        coils_2d: List of (coordinates, direction_sign).
        z_position: Vertical distance z.
        downsample_factor: Step size for points.
        current_parity: Relationship between top and bottom plane currents.
        current_scale: Current magnitude per turn (Amperes).

    Returns:
        list: A list of (top_loop, bottom_loop, I_top, I_bottom).
    '''
    coils_3d = []
    for path_coords, direction_sign in coils_2d:
        # Downsampling
        downsampled_coords = path_coords[::downsample_factor]
        if not np.array_equal(downsampled_coords[-1], path_coords[-1]):
            downsampled_coords = np.vstack([downsampled_coords, path_coords[-1]])

        # Projection
        z_top = np.full((len(downsampled_coords), 1), z_position)
        z_bottom = np.full((len(downsampled_coords), 1), -z_position)

        top_loop = np.hstack([downsampled_coords, z_top])
        bottom_loop = np.hstack([downsampled_coords, z_bottom])
        
        # Assign Currents based on winding direction and plane parity
        # Base current is scaled by current_scale
        # Vertices are already ordered (CCW for Hill, CW for Valley). 
        # We want to preserve this flow, so we use positive current magnitude.
        I_top = current_scale 
        I_bottom = I_top * current_parity
        
        coils_3d.append((top_loop, bottom_loop, I_top, I_bottom))

    return coils_3d

def rotate_coil_z(coils_3d: list[tuple[np.ndarray, np.ndarray, int]], 
                  angle_degrees: float) -> list[tuple[np.ndarray, np.ndarray, int]]:
    '''
    Rotates the coil geometry around the Z-axis by a given angle.

    Args:
        coils_3d (list): List of 3D coil tuples.
        angle_degrees (float): Rotation angle in degrees (counter-clockwise).

    Returns:
        list: New list of rotated coils.
    '''
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    rotated_coils = []
    for top_loop, bottom_loop, direction in coils_3d:
        # Apply rotation matrix to all points (N, 3) -> (N, 3)
        # (R @ v.T).T = v @ R.T
        top_new = top_loop @ rotation_matrix.T
        bottom_new = bottom_loop @ rotation_matrix.T
        
        rotated_coils.append((top_new, bottom_new, direction))
        
    return rotated_coils

