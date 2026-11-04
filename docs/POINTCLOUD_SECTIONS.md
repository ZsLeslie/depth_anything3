# 原始点云的 Z 轴截面线与图像投影

入口：`src/depth_anything_3/pointcloud_sections.py`。

直接读取第一步的像素关联点云，固定使用相机 Z 轴 `x=y=0`、方向 `[0,0,1]`。程序不读取拟合表面，不拟合平面、样条或圆柱，不平滑顶点和曲线。原始点之间只建立局部三角连接，再用 `Z=常数` 平面求交。

## 基本运行

在仓库根目录执行：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.pointcloud_sections \
  --pointcloud outputs/pointcloud/frame_0000.npz \
  --sections 5 \
  --original-image images/1280_000053.jpg --resize-only \
  --output-dir outputs/pointcloud_sections_z
```

默认按有效原始网格顶点 Z 值的 12%～88% 分位范围，等距放置 5 个平面。该范围取自原始点云，可能与拟合表面版本稍有不同。

只需处理图像预览时可省略 `--original-image` 和 `--resize-only`。已有点云无需 GPU，无需重新运行 DA3。输入必须是保留 `pixels_uv`、内参及有效性信息的 NPZ，不能仅传 PLY。

## 与拟合表面版本使用相同深度

为了单独比较原始几何与拟合几何，使用上一版相机 Z 轴输出中的截面深度：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.pointcloud_sections \
  --pointcloud outputs/pointcloud/frame_0000.npz \
  --depths-from-axis outputs/axis_z/axis.json \
  --original-image images/1280_000053.jpg --resize-only \
  --output-dir outputs/pointcloud_sections_z
```

程序只读取参考轴线的相机信息与绝对截面深度，不读取或使用其拟合表面。要求参考轴线为相机坐标、方向 `[0,0,1]`，处理尺寸和内参与点云一致。同尺寸和内参不能证明属于同一帧，仍需配对使用同一图像的结果。

也可以手动设置：

- `--z-range MIN MAX --sections N`：在指定的正深度区间等距放置平面。
- `--section-depths Z1 Z2 ...`：指定 2～100 个严格递增的正深度值，此时数量取自列表。

`--z-range`、`--section-depths`、`--depths-from-axis` 三者互斥。所有深度沿用预测单位，非米制点云不能按厘米或米解释。

## 计算流程

1. 校验点云的内参、像素对应、有效深度及颜色。世界坐标输入通过保存的 `world_to_camera` 转回相机坐标，再使用相机 Z 轴。
2. 仅连接原始处理图像中相邻像素对应的点。一个像素单元的四个观测必须完整，才生成两个候选三角面。
3. 三角形的 `(最大Z-最小Z)/最小Z` 超过默认 10% 时删除该三角面。孔洞与未采样像素不补点、不跨越。
4. 对每个 `Z=Z_i` 平面计算网格边交点，按共享网格边/顶点拼接曲线。交点是相邻原始顶点之间的线性插值，未将原始顶点移动到拟合表面。
5. 曲线按投影距离采样，用内参投影；检查原始观测支持与原始深度一致性。默认相对深度容差为 3%，投影总长小于 2 像素的片段丢弃。
6. 分别绘制各截面，保留分段。线宽覆盖区域也按有效观测掩码裁剪。

截平面与整个三角面共面时没有唯一截线，这类面片跳过并计入报告。多分支交点处分段，不按角度或空间距离任意拼接不同表面。

程序保留原始深度的起伏，因此截线可能比拟合版本更抖动。平滑但错误的 DA3 深度连接仍可能通过跳变检查；该方法不自动识别左右壁面、腹板和遮挡物。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--sections` | 5 | 自动或指定范围内的平面数量，2～100 |
| `--max-relative-jump` | 0.1 | 三角面原始深度跨度上限 |
| `--confidence-min` | 不设 | 可选原始置信度阈值；不是标定后的概率 |
| `--visibility-tolerance` | 0.03 | 曲线采样点与最近像素原始深度的一致性容差 |
| `--min-length-px` | 2 | 最短可见曲线投影长度 |
| `--line-width` | 1 | 处理图像上的线宽，原图按比例放大 |
| `--no-plots` | 关闭 | 跳过诊断图，保留叠加图和几何数据 |

建议使用第一步 `--stride 1` 的稠密点云；本程序不跨未采样像素连接，stride>1 输入可能没有三角面。

不同尺寸原图必须指定 `--resize-only`，或通过 `--processed-to-original transform.json` 提供 3×3 矩阵。矩阵方向是处理图像到原图，存在裁剪时必须包含偏移。请传入同一帧原图；不自动从无关目录选取图片。

## 输出

| 文件 | 内容 |
| --- | --- |
| `sections_overlay.png` | 处理图像上的可见截线，不同深度用不同颜色 |
| `sections_overlay_original.png` | 提供原图和映射后生成的原图叠加 |
| `sections_validation.png` | 叠加图、原始深度、三维网格与截线、诊断说明 |
| `raw_mesh.npz` | 原始相机坐标顶点、三角面、像素关联、有效性及内参 |
| `raw_mesh.ply` | 参与连接的原始顶点及三角面 |
| `sections.npz` | 所有几何交线和可见曲线、二维投影及深度编号 |
| `sections.obj` | 可见的有序分段曲线，不跨断口连接 |
| `section_points.ply` | 可见曲线的彩色三维采样点 |
| `axis.json` | 固定相机 Z 轴及截面位置 |
| `report.json` | 原始点和面数、各截面分段、过滤阈值、原图映射 |

`raw_mesh.npz` 保留输入点的顺序。相机坐标输入的顶点数值保持不变；世界坐标输入会先转成相机坐标。

`sections.npz` 中：

- `segments`、`section_ids`、`segments_uv` 为全部独立三维短线段、逐线段截面编号及投影，兼容查看截线段的方式。
- `vertices[offsets[i]:offsets[i+1]]` 为第 i 段有序几何曲线，`path_section_ids[i]` 表示其截面编号。
- `visible_vertices[visible_offsets[i]:visible_offsets[i+1]]` 为裁剪后的第 i 段曲线，`visible_section_ids[i]` 为其截面编号。
- `visible_pixels_uv` 是处理图像坐标，`section_depths` 是绝对 Z 深度，`closed` 表示末点重复首点的闭合曲线。

三维诊断图最多显示约 20,000 个三角面以控制绘图时间；实际求交及保存的网格使用全部通过检查的面。

## 当前运行结果与验证

当前点云来自 `images/1280_000053.jpg`，处理尺寸 504×280。实际运行使用 `outputs/axis_z/axis.json` 的 5 个截面深度：约 0.675325、0.847202、1.019080、1.190957、1.362835。

141,120 个原始点组成 280,267 个三角面；几何交线形成 12 段，可见性及长度裁剪后保留 10 段。分段数量是当前数据结果，不保证每个截面闭合。

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_pointcloud_sections.py tests/test_pointcloud.py tests/test_axis.py
```

测试覆盖原始顶点保持不变、恒定 Z 平面求交、孔洞/深度跳变保留、稀疏输入、外参和置信度、参考深度配对、原图映射及命令行导出。数值一致性不代表原始深度已获得真实内壁测量精度。
