# A python refactor of `../离散化磁场计算及优化算法Bx`

## Roadmap
- Refactor key functions in `src`
- From abstract to concrete
  
  
  1. 矩阵逆向求解 (Inverse Problem Solver)
   * 输入:
       * 灵敏度矩阵 $A$ (216x16)，来自 A_mirror.mat。
       * 正则化矩阵 $T$ (16x16)，来自 T_mirror.mat。
       * 正则化参数 $\lambda$ (lamida_a = 1.6544e-20)。
       * 目标场向量 $B_{target}$ (216x1，全为 1，即要求均匀场)。
   * 计算公式:
       * 使用 Tikhonov 正则化求解流函数系数向量 $la$ (16x1)：
        $$ la = (A^T A + \lambda T^T T)^{-1} A^T B_{target} $$
   * 输出: 系数向量 $la$ (包含 16 个傅里叶级数系数)。

  2. 流函数空间重构 (Stream Function Reconstruction)
   * 输入: 系数向量 $la$。
   * 参数:
       * 傅里叶项数 $M=4, N=4$。
       * 线圈尺寸 $L=0.85$m。
       * 网格分辨率 $400 \times 400$。
   * 计算:
       * 在一个二维网格 $(X, Y)$ 上，利用双重傅里叶正弦/余弦级数重建连续流函数 $\Phi(x,y)$。
       * 公式形式为：$\Phi = -\sum \sum la_{mn} \cdot \sin(...) \cdot \cos(...)$。
   * 输出: 一个 $400 \times 400$ 的二维矩阵 Fai，代表线圈平面上每一点的电流势能。

  3. 等值线离散化 (Discretization / Contouring)
   * 输入: 流函数矩阵 Fai。
   * 参数: 线圈匝数 $CL = 22$。
   * 操作:
       * 计算 Fai 的最大最小值，按 22 等分确定切片阈值。
       * 调用 contour 函数提取这 22 条等值线的坐标路径。
       * 调用自定义函数 Judge_wise 分离每一圈，并计算电流方向（正/反）。
   * 输出: 22 组独立的闭合曲线坐标 $(x, y)$，这就是最终的线圈几何形状。

## 函数编写时必须描述的内容
   1. 一句话摘要 (Summary Line):
       * 必须: 用动词开头，简明扼要地说明函数“做什么”（而不是“怎么做”）。
       * 示例: "Load sensitivity and regularization matrices from .mat files."

   2. 详细描述 (Extended Description) (可选):
       * 如果逻辑复杂，补充说明算法背景、公式依据或关键假设。
       * 示例: "Loads the A matrix (sensitivity) and T matrix (regularization) used for the Tikhonov inverse problem solver. Expects standard MATLAB .mat format."

   3. 参数列表 (Parameters / Args):
       * 必须: 列出所有参数名。
       * 必须: 标明每个参数的数据类型 (Type)。
       * 必须: 解释参数的含义、单位（如有）、取值范围或默认值。
       * 格式: name (type): Description.

   4. 返回值 (Returns):
       * 必须: 说明返回了什么数据。
       * 必须: 标明返回值的类型。
       * 必须: 描述返回值的结构（如矩阵形状 (N, M)）、单位或含义。如果是元组，要分别说明每个元素。

   5. 异常抛出 (Raises) (如有):
       * 必须: 列出函数可能主动抛出的错误类型（如 FileNotFoundError, ValueError）及其触发条件。

   6. 示例 (Examples) (强烈推荐):
       * 提供一段可运行的代码片段，展示如何调用该函数以及预期的输出。这对于理解接口契约最有帮助。