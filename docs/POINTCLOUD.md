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

必需字段为 `depth`（N,H,W）和 `intrinsics`（N,3,3）；使用下述 `--intrinsics` 时，NPZ 可以不含内参。可选 `image`（N,H,W,3，uint8 RGB）、`conf`（N,H,W）、`extrinsics`（N,3,4 或 N,4,4）。单帧也支持去掉 N 维。所有字段必须来自同一处理分辨率；原图不能未经对应的缩放、裁剪就直接用于着色。

## 指定相机内参 K

通过 `--intrinsics camera_K.json` 或 `--intrinsics camera_K.npy` 传入内参，覆盖模型预测或输入 NPZ 的内参。文件支持单个 `3×3` 矩阵（所有帧共用），或 `N×3×3` 矩阵（逐帧对应输入顺序；图片目录按文件名排序）。JSON 也支持 `{"K": ...}` 或 `{"intrinsics": ...}`。

例如下面的 JSON 表示一组 **1920×1080 标定分辨率下的示例值，需替换为实际标定结果**：

```json
[
  [1400.0, 0.0, 960.0],
  [0.0, 1400.0, 540.0],
  [0.0, 0.0, 1.0]
]
```

```bash
PYTHONPATH=src python -m depth_anything_3.pointcloud \
  --images images/1920_45_000001.jpg \
  --model-dir ~/models/DA3-LARGE/ \
  --intrinsics camera_K.json \
  --intrinsics-size 1920 1080 \
  --output-dir outputs/1920/pointcloud_custom_K
```

- 从图像推理：未传 `--intrinsics-size` 时，K 按每张原图的像素坐标解释；传入该参数时，先从指定标定分辨率缩放到各原图尺寸。随后由 DA3 预处理同步执行图像和 K 的缩放及批次中心裁剪。最终 `prediction.npz` 和每帧点云 NPZ 保存实际用于反投影的处理后 K。
- 从已有预测恢复：未传 `--intrinsics-size` 时，K 按深度图的像素坐标解释；传入该参数时，按宽高比例缩放到深度图尺寸。例如仅有缩放的 1920×1080 → 504×280，使用 `K_depth = diag(504/1920, 280/1080, 1) @ K`。**已有预测若还包含裁剪，此参数不能恢复未知裁剪偏移；应先计算处理后 K，再不带 `--intrinsics-size` 传入。**

```bash
PYTHONPATH=src python -m depth_anything_3.pointcloud \
  --prediction outputs/1920/pointcloud/prediction.npz \
  --intrinsics camera_K.json \
  --intrinsics-size 1920 1080 \
  --output-dir outputs/1920/pointcloud_custom_K
```

内参须为有限、可逆的针孔矩阵，焦距为正，最后一行为 `[0, 0, 1]`。本接口不接收畸变系数；K 应与实际输入图像的成像几何一致。指定 K 不会自动把非米制深度转换为米制深度。

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

## 按深度置信度着色

添加 `--color-by confidence` 后，点的颜色由原始深度置信度决定：**范围下限为红色，中点为黄色，上限为绿色**。默认 `--color-by rgb` 继续使用 RGB 图像颜色。

已有预测可以直接重新导出置信度着色点云：

```bash
PYTHONPATH=src python -m depth_anything_3.pointcloud \
  --prediction outputs/pointcloud_K/prediction.npz \
  --color-by confidence \
  --output-dir outputs/pointcloud_K_confidence
```

默认颜色范围取整个输入批次中有效正深度、有限置信度像素的置信度最小值和最大值。所有帧共享此范围；后续点数采样、`--confidence-min` 和 `--max-depth` 不改变色标。若所有值相同，使用中间的黄色。缺少 `conf` 字段时明确报错。

不同批次需要直接比较时，可以固定颜色范围，例如添加 `--confidence-color-range 1 5.5`。低于下限的值显示红色，高于上限的值显示绿色。该参数控制颜色，不负责过滤点，且必须与 `--color-by confidence` 一起使用。DA3 原始置信度不是 0～1 的正确概率，颜色只反映指定范围内的分数高低。

- PLY 的顶点颜色为置信度颜色。
- 点云 NPZ 的 `colors` 保存相同的置信度颜色；`confidence` 保留原始分数，有 RGB 时额外以 `rgb_colors` 保存原始颜色。
- NPZ 保存 `color_by`、`confidence_color_range`；`summary.json` 保存实际使用的范围和 `red-yellow-green` 色标名称。没有任何有效深度/置信度像素时，自动范围为 null，仍可以导出空点云。
- 从图像开始推理时，保存的 `prediction.npz` 中 `image` 仍是处理后的 RGB，可继续用于其他着色方式。

## 几何验证

```bash
PYTHONPATH=src python -m pytest -q tests/test_pointcloud.py tests/test_api_intrinsics.py
```

测试覆盖平面反投影及重投影、无效深度过滤、采样与颜色对应、外参方向、二进制 PLY、空点云、标准 NPZ/mini_npz 命令行导出、缺失内参错误，以及自定义 K、标定尺寸缩放和批次裁剪后的内参一致性。API 测试需要图像推理依赖，缺少时跳过。它们验证坐标转换与导出逻辑，不代表实际图像上的深度估计精度。

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
