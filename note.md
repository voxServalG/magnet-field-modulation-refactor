# 验证记录

## 2026-01-11
- **任务**: 比较 `离散化磁场计算及优化算法Bx` 目录下的 `la_saved.mat` 与 `python_refactor/compare_la.py` 的计算结果。
- **结果**: **匹配 (Match)**。
- **详细数据**:
    - 最大绝对误差: `9.88e-07`
    - 参考值量级: `6.82e+06`
    - 相对误差: 约 `1.45e-13`
- **结论**: Python 重构代码成功复现了 MATLAB 的流函数系数 (la) 计算逻辑。微小差异属于正常的浮点运算精度范围。

## 2026-01-11 补充：物理仿真核心坑点记录

### 1. 任务：验证 `main.py` 正向物理仿真结果
- **现象**: 初始仿真结果显示中心磁场 $B_0$ 接近 0（$2.3 \times 10^{-3} \text{ nT}$），相对误差高达 0^7 \%$。
- **定位**: 发现 `physics.py` 中的电流方向逻辑存在“二次修正”错误。
- **坑点描述**: 
    - **误区**: 认为必须手动判断线圈的顺逆时针（Shoelace 算法）并强制修正电流正负。
    - **真相**: 流函数 $\Phi$ 的等值线提取函数 (`contour`) 返回的点序列已经天然包含了梯度方向信息。
    - **后果**: 手动修正方向导致了“山峰”（正值区）与“盆地”（负值区）的磁场由相互配合变成了相互抵消。
- **修复**: 在 `calculate_field_from_coils` 中移除 `direction_sign` 的干扰，直接信任 `contour` 生成的路径方向。
- **最终结果**: 
    - 中心磁场恢复至 **383.53 nT**。
    - 均匀度误差 **1.27%**，完美对标 MATLAB 原版。

---
**下班总结**: 尊重数学原生的拓扑结构，不要在物理积分层做多余的逻辑干预。

## 15 Jan
- 在特定线圈平面位置及特定目标点位置下，A作为系统 连接了 流函数系数(i) 与 目标点的磁感应强度(o)
- 基于 need for varied target point, attempt to develop "A generator" and replace static `A` into active ones.
- **参数单位及物理意义**
  - `L` / m
    - 线圈0.5倍边长。
    - 线圈板是一个边长为 `2L` 的正方形.
  - `a` / m
    - 两个Bx线圈分别位于 `z=a`, `z=-a`上。

## 18 Jan: 阵列化主动屏蔽仿真流程 (Array Simulation Workflow)

### 抽象目标
构建一个基于**线性叠加原理**的系统。该系统由多个三轴线圈单元 (Bx/By/Bz) 组成阵列，通过主动优化各通道电流，来压制/补偿任意给定的背景磁场。

### 实务步骤清单 (Practical Steps Checklist)

#### 步骤 1: 准备“标准单元” (Prepare the Unit)
*   **动作**: 实例化 `CoilFactory`。
*   **配置**: `L=0.85`, `a=0.7`, `modes=(4,4)`, `reg_lambda=1e-14`, `use_shielding=False` (暂忽略屏蔽室)。
*   **执行**: 生成 `unit_bx`, `unit_by`, `unit_bz` 列表，包含几何坐标和单位电流信息。

#### 步骤 2: 定义阵列与目标区域 (Define Array & Target ROI)
*   **阵列几何**: 定义 `Array_Pos` (Nx3) 矩阵，包含每个单元中心的偏移量。
*   **目标区域 (ROI)**: 定义 `Target_Points` (Mx3)，覆盖需要进行磁场压制的空间范围。

#### 步骤 3: 构建响应矩阵 S (Build Response Matrix)
*   **逻辑**: 建立映射关系：单元电流输入 ($3K$) $\to$ ROI 磁场输出 ($3M$)。
*   **结构**: 矩阵 `S`，形状为 $(3M, 3K)$。
*   **循环构建**:
    *   遍历每个 单元 $i$ 和 轴 $j$:
        1.  将标准线圈几何平移 `Array_Pos[i]`。
        2.  调用 `physics.py` 计算其在 `Target_Points` 产生的磁场。
        3.  将结果向量拉直 (Flatten) 为 $3M \times 1$。
        4.  填入 `S` 矩阵的第 $(3i + j)$ 列。

#### 步骤 4: 设定“假想敌” (Define Background Field)
*   **动作**: 创建函数 `B_bg_func(points)` 模拟环境干扰场（如梯度场）。
*   **数据**: 计算并生成 ROI 上的背景场向量 `B_bg_data` $(3M \times 1)$。

#### 步骤 5: 求解与反击 (Solve & Counter-Attack)
*   **数学目标**: 求解方程 $\mathbf{S} \cdot \vec{I} = -\vec{B}_{bg}$。
*   **求解器**: 使用 `np.linalg.lstsq` (最小二乘法) 计算最优电流向量 $\vec{I}_{opt}$。

#### 步骤 6: 最终仿真与验证 (Final Simulation & Verification)
*   **组装**: 构建 `Final_System` 列表。遍历所有通道，平移线圈并赋予 $\vec{I}_{opt}$ 中计算出的对应电流权重。
*   **物理计算**: 计算 `B_total = physics(Final_System) + B_bg`。
*   **验证**: 统计 `B_total` 的残差 (RMS/Max)，验证“压制”效果是否显著（接近零）。