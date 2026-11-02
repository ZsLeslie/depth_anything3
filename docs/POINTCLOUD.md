# 第一步：从 DA3 深度恢复点云

本步骤不需要分割掩码，恢复相机可见表面的点云，为后续表面拟合和网格投影保留像素对应关系。不补全不可见表面，也不进行平滑、曲面拟合或尺度归一化。

在仓库根目录执行以下命令。读取预测结果只需要 NumPy；从图像开始推理需要安装项目推理依赖及相应模型权重。

## 从 images 推理并导出

```bash
PYTHONPATH=src python -m depth_anything_3.pointcloud \
  --images images \
  --model-dir depth-anything/DA3-BASE \
  --output-dir outputs/pointcloud
```

`--images` 也可以传入单张图片路径。目录中的图片按文件名排序，一次共同推理，逐帧导出点云。`--model-dir` 可指定本地权重目录。推理默认使用 CUDA；`--device` 可以指定设备，例如 `cuda:0`。当前上游推理实现依赖 CUDA 检查，CPU 推理不属于本步骤验证范围。默认 `--process-res 504`。已有 NPZ 的点云转换不需要 GPU。

DA3-BASE 用于初步验证，输出不保证米制尺度。使用其他模型时必须确认其提供内参；本脚本不会自动猜测焦距。即使使用 metric 模型，也需要实际测量验证尺度精度。

## 从已有预测恢复

支持 DA3 的 `npz` 和 `mini_npz` 导出；前者包含处理后的 RGB，可生成彩色点云，后者没有 RGB 时生成无颜色点云。

```bash
PYTHONPATH=src python -m depth_anything_3.pointcloud \
  --prediction path/to/exports/npz/results.npz \
  --output-dir outputs/pointcloud
```

必需字段为 `depth`（N,H,W）和 `intrinsics`（N,3,3）；可选 `image`（N,H,W,3，uint8 RGB）、`conf`（N,H,W）、`extrinsics`（N,3,4 或 N,4,4）。单帧也支持去掉 N 维。所有字段必须来自同一处理分辨率；原图不能未经对应的缩放、裁剪就直接用于着色。

## 输出与坐标约定

- `frame_0000.ply`：可由 CloudCompare、MeshLab 等工具打开的点云。
- `frame_0000.npz`：`points`（M,3）、`pixels_uv`（M,2）、`valid_mask`（H,W）、`intrinsics`、`coordinates`；有颜色或置信度时额外保存 `colors`、`confidence`。世界坐标模式还保存 `world_to_camera`。
- `summary.json`：坐标系、点数、深度尺寸及过滤参数；`is_metric` 为 null 时表示输入 NPZ 没有尺度标记，不能据此认定单位为米。
- `prediction.npz`：仅从图像推理时保存，包含未进行点云过滤的预测数组、处理后的 RGB、源图像路径及尺度标记，可供后续重复处理。

默认每帧保留独立的 OpenCV 相机坐标：x 向右、y 向下、z 向前。计算公式为 `P_camera = depth * inverse(K) @ [u, v, 1]`。DA3 深度表示相机 z 值，不是沿单位射线的距离。

`pixels_uv` 对应模型处理后的图像，**不是原始 1920×1080 图像坐标**。后续回绘原图时需要反向应用预处理变换。本脚本从图像推理使用 `upper_bound_resize`；不同尺寸图片共同推理时还可能发生批次中心裁剪，因此保留处理后的图像作为准确对应基准。

使用 `--coordinates world` 时，按 `inverse(extrinsics)` 把点变换到 DA3 世界坐标，要求输入含外参。同一次多视图预测的各帧可处于共同世界坐标；独立推理的结果不能直接视为已配准。本步骤不合并或去重多帧点云。

## 可选过滤

默认只去掉非正、非有限深度，以及存在置信度图时的非有限置信度。不会默认删除低置信度区域，以便检查原始几何。

- `--confidence-min VALUE`：保留大于等于该值的原始置信度。它不是 0～1 的校准概率，需要查看当前预测分布后设置。
- `--max-depth VALUE`：限制相机 z 深度，单位与预测一致；非米制输出不能按米解释。
- `--stride 2`：每隔两个像素采样，点数约为四分之一；仍保留原处理图像的像素坐标与内参。

全被过滤时会输出空点云并提示，不自动放宽阈值。相同输出目录、相同文件名会覆盖已有结果；不同试验建议使用不同目录。

## 几何验证

```bash
PYTHONPATH=src python -m pytest -q tests/test_pointcloud.py
```

测试覆盖平面反投影及重投影、无效深度过滤、采样与颜色对应、外参方向、二进制 PLY、空点云、标准 NPZ/mini_npz 命令行导出和缺失内参错误。它们验证坐标转换与导出逻辑，不代表实际图像上的深度估计精度。

本机已用以下命令对 `images/1920_45_000001.jpg` 完成验证：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.pointcloud \
  --images images \
  --model-dir /home/nvidia/models/DA3-BASE \
  --device cuda \
  --output-dir outputs/pointcloud
```

处理尺寸为 504×280，默认不过滤低置信度点，共导出 141,120 点。通过 trimesh 独立读取 PLY 并核对坐标、颜色；点云重投影到处理图像的最大数值误差小于 1e-8 像素。这仅验证变换与数据对应正确，不能衡量 DA3 预测曲面的真实误差。输出为非米制尺度。
