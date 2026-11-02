# 第四步：周向坐标与稀疏网格预览

读取第二步的有效曲面和第三步的轴线，建立轴向位置 s、周向角 theta，生成横截面线与纵向线，并投影到处理图像或原图。无需重新运行 DA3，不需要 GPU 或分割掩码。

这是基于候选轴线的预览。输入轴线的 `needs_review` 状态会保留，不因为生成了网格就自动确认其几何正确性。

## 对当前图像运行

本次使用目录中已有的 **1920 图像人工参考线轴线**，而不是早期自动轴线：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.grid \
  --surface outputs/1920/surface/surface.npz \
  --axis outputs/1920/axis_annotate/axis.json \
  --sections 5 \
  --sectors 12 \
  --original-image images/1920_45_000001.jpg \
  --resize-only \
  --output-dir outputs/1920/grid
```

需要 NumPy、SciPy、OpenCV、Pillow；诊断图还使用 Matplotlib。`--no-plots` 跳过诊断图，仍保存网格叠加图。

`--surface` 和 `--axis` 都是必需参数。程序校验二者的相机内参与处理尺寸是否一致；同尺寸、同内参不能证明属于同一帧，因此调用时仍需配对使用同一图像的结果。输出目录中同名文件会覆盖。

## 预览颜色与文件

- **青色**：s 固定的横截面线，默认 5 条。
- **橙色**：theta 固定的纵向线，默认每 30° 一条，共 12 个周向分区。
- **红色**：theta=0 的纵向参考线。

| 文件 | 内容 |
| --- | --- |
| `grid_overlay.png` | 处理图像上的网格预览，本例为 504×280 |
| `grid_overlay_original.png` | 指定原图时输出，本例为 1920×1080 |
| `grid_validation.png` | 叠加图、周向角场、三维曲面与曲线、诊断说明 |
| `frame.json` | 轴线原点、轴向、周向基向量 e1/e2、轴向范围及各网格线参数 |
| `coordinates.npz` | 与曲面逐点对应的 s、theta、radius、有效性及像素位置 |
| `grid.npz` | 按顺序连接的三维曲线、断点索引、线族/线号及可见曲线投影 |
| `grid_curves.obj` | 分段的三维线对象，保留每条曲线的连接关系 |
| `grid_points.ply` | 可见曲线上的彩色采样点，可与表面 PLY 一起查看 |
| `report.json` | 输入、轴线状态、分支诊断、可见片段数和原图变换 |

本次结果包含 **5 条横截面线和 12 条纵向线**。因有效曲面存在孔洞，横截面线共形成 8 段，纵向线各形成 1 段，合计 20 段；可见性裁剪后保留 20 段。当前采样检查未检测到同一周向角、同一轴向位置对应多个半径的情况。

## 坐标与曲线定义

以 c 为轴线上参考点，a 为单位轴向，e1/e2 为垂直于 a 的两个正交单位向量：

```text
s = dot(P-c, a)
x = dot(P-c, e1)
y = dot(P-c, e2)
theta = atan2(y, x) mod 2π
radius = sqrt(x²+y²)
```

默认 e1 为相机 X 方向在横截面内的投影，e2=a×e1。theta 按绕 +a 的右手方向增大。这个默认零位只是显示约定，不代表实际接缝或设计基准。

横截面线通过 `dot(P-c,a)=s_i` 的平面与有效三角网格求交得到。纵向线通过包含轴线的半平面与网格求交得到，限制正径向半侧，以区分 theta 与 theta+π。纵向线同时裁剪到选定的轴向范围。

默认轴向范围沿用 `axis.json` 的 `s_range`，其中等间距放置 5 个截面；此范围通常只覆盖有效曲面的中间部分，不会强行延伸到远处暗区。周向等角度不等于等表面弧长，也不保证网格等面积。长度仍为 DA3 的预测单位。

## 曲线连续性与分支处理

- 利用共享三角网格边/顶点标识拼接曲线，不按点之间的空间距离盲目连接，因此不会把相邻但拓扑分离的表面粘在一起。
- 对共享边重复交线去重；孔洞处保持断开。出现多分支交点时分段，不猜测物理表面的延续关系。
- 截平面与整个三角面共面时没有唯一交线，这类面片会跳过并计入诊断。
- 对每条纵向线，在轴向范围内抽取 40 个位置检查是否存在不同半径的交点。如果检测到，记录为歧义并保留分支，不自动选择外壁或腹板。

**未检测到分支不等于已经识别正确的物理表面。** 40 个位置的检查可能漏过很短的分支；平滑接合的腹板和外壁也不一定产生拓扑分支。应结合原图检查，必要时再引入表面区域约束。

## 可见性与原图映射

曲线按最多 0.5 个处理图像像素的间隔进行透视正确采样。采样点必须位于第二步的有效区域，且其 z 深度与该像素的拟合深度相差不超过默认 3%。无效区域或深度不一致处会断开，线宽覆盖区域也被有效掩码裁剪。

这里的掩码来自几何有效性，不要求分割模型。可见性检查仍依赖 DA3：如果深度没有恢复某个线缆、灯具等遮挡物，不能保证自动避开该物体。

若原图与处理图像尺寸不同，必须显式说明变换：

- `--resize-only`：确认只有缩放，分别使用原图/处理图的宽、高比例。本例为单张图片的 `upper_bound_resize`，可使用此选项。
- `--processed-to-original transform.json`：提供 3×3 矩阵；JSON 顶层为矩阵数组，映射方向是处理图像到原图。存在裁剪时必须包含裁剪偏移。

未说明变换时，程序不会把不同尺寸的图片直接拉伸叠加。

## 调整零位、范围和密度

```bash
PYTHONPATH=src /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.grid \
  --surface outputs/1920/surface/surface.npz \
  --axis outputs/1920/axis_annotate/axis.json \
  --zero-angle-deg 15 \
  --sections 7 \
  --sectors 16 \
  --output-dir outputs/1920/grid_rotated
```

`--zero-angle-deg` 绕轴旋转周向零位。还可以用 `--zero-pixel U V` 指定一个**处理图像**中有效表面点，让其径向方向成为零位；同时设置角度时，在这个参考方向上继续旋转。选择靠近轴线的点不能稳定定义周向角，程序会拒绝零向量。

其他参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--s-range MIN MAX` | 轴线文件范围 | 限制绘制的轴向区间，单位沿用预测 |
| `--visibility-tolerance` | 0.03 | 采样点相对拟合 z 深度的一致性阈值 |
| `--min-length-px` | 2 | 丢弃投影总长度过小的可见片段 |
| `--line-width` | 1 | 处理图像线宽；原图线宽按宽度比例放大 |

第一版不自动删掉远处过密的完整网格线；可通过减少分区数、缩小轴向范围控制密度，并保持网格线编号稳定。

## 数据接口

`coordinates.npz` 中 theta 单位为弧度，范围 [0,2π)，轴线上周向角无定义时保存 NaN。`parameter_valid` 只表示坐标可定义，实际使用还要结合 `inlier`。逐点数组与输入曲面点顺序相同。

`grid.npz` 中每条曲线由 `vertices[offsets[i]:offsets[i+1]]` 提取，`family_ids[i]` 为 0 表示横截面线、1 表示纵向线；`line_ids[i]` 为该线族内的零起始编号。同一条网格线断开后会有多个具有相同线号的片段，不能直接连接这些片段。

带 `visible_` 前缀的数组为可见性裁剪后的版本；`visible_pixels_uv` 是处理图像坐标。`closed` 标记闭合曲线，其末点会重复首点。原图映射矩阵保存在 `report.json`。

## 测试

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_grid.py tests/test_axis.py tests/test_surface.py tests/test_pointcloud.py
```

覆盖已知圆柱上的截面闭合和半平面纵向线、零位旋转、公共边去重、拓扑分离表面不误连、多半径分支、共面退化、孔洞/前景深度裁剪、透视采样、原图变换以及端到端命令行导出。
