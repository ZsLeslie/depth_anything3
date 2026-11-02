# 第二步：拟合可见表面

输入第一步导出的像素关联点云，输出连续曲面模型、解析法向、残差、可靠区域及三角网格。不要求分割掩码，不重新运行 DA3，使用 CPU 即可。

本版本采用**透视逆深度 B 样条曲面**。对当前图像进行初步几何试验时，平面与圆柱/锥面的距离分组不能可靠对应左右物理表面。因此这里先拟合可见几何，不把简单几何模型的分类当作真实部件分区。

## 运行

在仓库根目录执行，本机已安装的 DA3 环境包含所需的 NumPy、SciPy 和 Matplotlib：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.surface \
  --pointcloud outputs/pointcloud/frame_0000.npz \
  --output-dir outputs/surface
```

在其他已安装依赖的环境中，可将解释器路径替换成 `python`。`--no-plots` 可跳过图表生成，此时仅需要 NumPy 和 SciPy。第一步 PLY 不包含像素关联和内参，因此这里读取 NPZ，不直接读取 PLY。

支持第一步的相机坐标与世界坐标点云；世界坐标输入会根据保存的 `world_to_camera` 转回相机坐标。本步骤所有输出均采用 OpenCV 相机坐标，尺度保持不变。

## 拟合方法

在处理后图像上定义 `s = u/(W-1)`、`t = v/(H-1)`，对归一化逆深度进行三次张量积 B 样条拟合：

```text
g(s,t) = Σ C[j,i] B_i(s) B_j(t)
Z(u,v) = depth_scale / g(s,t)
S(u,v) = Z(u,v) × inverse(K) × [u,v,1]
```

`depth_scale` 是训练点深度的中位数，仅用于数值稳定，恢复时会乘回；不会改变点云尺度。控制点矩阵 C 的形状为 `(controls_v, controls_u)`。

拟合使用 Huber 迭代重加权最小二乘降低离群点影响，同时惩罚控制网格的二阶差分。解析求导 `∂S/∂u` 与 `∂S/∂v` 后计算单位法向，并朝向相机。

把图像划分为 8×8 像素块，其中约 20% 作为验证块，始终不参与拟合。训练最多抽取 16,000 个点，随机种子固定为 42。置信度不是校准概率，默认不直接把它作为统计权重；可以使用 `--confidence-min` 排除低置信度点。

**这些 s、t 是图像参数，不是结构的轴向距离和周向角。** 后续要生成参考图中的纵向线与截面线，还需要建立结构坐标，不能直接把 s、t 等值线当作目标网格。

## 输出

| 文件 | 内容 |
| --- | --- |
| `model.json` | 样条次数、节点、控制系数、深度归一化尺度、内参、处理图像尺寸 |
| `surface.npz` | 原始点、拟合点、法向、像素关联、残差、训练/验证/有效点标记、三角面索引 |
| `surface_points.ply` | 拟合后且通过残差阈值的彩色点云 |
| `surface_mesh.ply` | 仅连接可靠且连续区域的彩色三角网格，可用 CloudCompare/MeshLab 打开 |
| `report.json` | 训练及验证统计、参数、收敛状态、网格面数 |
| `diagnostics.png` | RGB、原始/拟合深度、残差、法向、有效区域 |
| `surface_3d.png` | 在同一坐标和相同三角连接下，对比原始与拟合表面 |

`surface.npz` 的所有逐点数组与输入点顺序相同，`faces` 索引此完整数组；PLY 网格会单独压缩为被三角面使用的顶点，不能把 PLY 的索引直接用于 NPZ。

- `eligible`：深度有效且通过可选置信度阈值。
- `evaluated`：样条在该像素处能给出有限的正深度和法向；不代表该处可靠。
- `inlier`：eligible 且相对 z 深度残差不超过阈值，默认 5%。
- `train` / `validation`：实际参与拟合的采样点 / 留出的验证块像素，二者不重叠。
- `depth_residual`：`Z_fit - Z_DA3`，保持原预测单位。
- `relative_depth_residual`：`abs(Z_fit - Z_DA3) / Z_DA3`。
- `point_displacement`：沿原相机射线移动前后的三维距离，**不是点到曲面的最短距离**。

默认每 3 像素构建一个网格单元。单元内部有任何孔洞、置信度排除点或残差超标点时不连接；相邻原始深度或三角面顶点深度存在超过 10% 的相对跳变时也不连接。默认输出不补全隐藏表面。

连续样条仍可能平滑真实折角；网格裁剪只保证遵循观测支持及深度检查，不能保证识别所有物理分界。对于第一步以 stride>1 导出的稀疏点云，可以拟合曲面，但严格支持检查可能不生成三角面；建议使用第一步默认 stride=1。

## 主要参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--controls-u` / `--controls-v` | 16 / 10 | 增大可表示更细变化，也更容易跟随深度噪声 |
| `--smoothness` | 0.05 | 越大越平滑，也可能增加几何偏差 |
| `--relative-threshold` | 0.05 | 输出点云及网格的最大相对 z 深度残差 |
| `--confidence-min` | 不设 | 按原始 DA3 置信度排除点 |
| `--max-samples` | 16000 | 实际训练采样上限 |
| `--iterations` | 30 | 稳健拟合最大迭代数，满足系数收敛条件会提前停止 |
| `--validation-block` | 8 | 验证块边长，单位为处理图像像素 |
| `--mesh-step` | 3 | 三角网格采样间隔 |
| `--max-relative-jump` | 0.1 | 网格连接的相对深度跳变上限 |

比较参数时应同时查看未参与拟合点的误差、三维形状和异常区域，不只追求训练误差更小。同名输出文件会覆盖，参数对比应使用不同目录。

## 后续调用曲面

```python
import json
import numpy as np
from depth_anything_3.utils.surface import evaluate_surface

with open("outputs/surface/model.json") as f:
    model = json.load(f)
pixels = np.array([[100.5, 80.0], [300.0, 180.5]])
points, normals, valid = evaluate_surface(model, pixels, np.asarray(model["intrinsics"]))
```

该函数支持亚像素位置，但 `valid` 只表示函数可求值。下一步生成网格时，还必须用 `surface.npz` 的可靠区域限制绘制范围，避免把没有观测约束的插值当作真实表面。

## 验证

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_surface.py tests/test_pointcloud.py
```

测试包含透视平面恢复、解析法向与数值导数一致性、含离群点的曲面恢复、验证集隔离、孔洞与深度跳变裁剪、世界坐标还原、三角 PLY 和命令行导出。

对当前 `1920_45_000001.jpg` 的默认参数运行在第 18 次迭代收敛：141,120 个输入点中保留 134,098 个拟合点，生成 29,222 个三角面。28,224 个验证点的相对深度误差中位数约 0.445%，95 分位约 5.077%。这些是相对于 DA3 预测的拟合误差，不是对真实结构的测量误差。DA3-BASE 的非米制尺度及其几何偏差仍然保留。
