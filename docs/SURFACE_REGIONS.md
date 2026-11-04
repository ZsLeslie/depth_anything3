# 人工多边形分区域拟合

本功能针对全局拟合可能平滑左右壁面、腹板交界的问题：先人工指定目标表面区域，再为每个区域建立独立的透视逆深度 B 样条。无需分割模型，无需重新预测深度。

## 1. 在原图上标注并拟合

在有桌面显示的终端、仓库根目录执行：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.surface \
  --pointcloud outputs/1920/pointcloud/frame_0000.npz \
  --annotate-regions \
  --annotation-image images/1920_45_000001.jpg \
  --annotation-resize-only \
  --output-dir outputs/1920/surface_regions
```

`--annotation-resize-only` 明确确认原图到深度处理图只有缩放，没有裁剪。当前单张图片的处理方式可用此参数；有裁剪时改用 `--annotation-to-processed matrix.json`，文件顶层为从标注图到深度图的 3×3 矩阵。映射方向与 grid 的 `--processed-to-original` 相反。

省略 `--annotation-image` 时，窗口直接显示点云保存的处理图像 RGB，其坐标与深度一一对应，不需要尺寸变换。没有 RGB 时显示浅色背景，建议指定匹配的图像。

窗口默认提供 `left`、`right` 两个区域：

| 操作 | 功能 |
| --- | --- |
| `1` / `2` | 切换左、右区域；有未完成多边形时需先完成或撤销 |
| 鼠标左键 | 沿所选表面的边界添加顶点 |
| `Enter` | 闭合并校验当前多边形；同一区域可以继续添加多个多边形 |
| `H` | 切换区域内部的孔洞标注模式，再次按 H 返回外多边形模式 |
| `Backspace` | 撤销当前顶点；当前没有顶点时撤销上一个已完成多边形或孔洞 |
| 滚轮 | 围绕鼠标位置缩放 |
| 工具栏 | 平移和框选缩放；开启工具栏模式时不会误添加顶点 |
| `S` | 保存已闭合区域，关闭窗口并开始拟合 |
| `Esc` / 关闭窗口 | 取消，不开始拟合 |

不要将不同部件放进同一区域。边界不确定时可留出窄带；灯具、遮挡物和远处暗区可以不包含，或用孔洞排除。至少需要一个有效区域；未标注的默认区域不会保存。若某个区域太小、有效训练点不足，程序会明确报出区域名，不会退回全局拟合或自行删掉该区域。

窗口保存时会检查自交、区域重叠、尺寸变换和处理像素覆盖。不同区域不允许在处理像素中心处重叠，包括共同覆盖的边界像素；可保留小间隔解决。孔洞归属于当前区域，不影响其他区域。

交互窗口需要 Matplotlib GUI 后端和可用桌面。无显示环境请通过 JSON 输入；`--no-plots` 只跳过诊断图，不会关闭显式请求的标注窗口。

## 2. 复用或编辑已有标注

复用已保存 JSON，直接重新拟合：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.surface \
  --pointcloud outputs/1920/pointcloud/frame_0000.npz \
  --regions outputs/1920/surface_regions/regions.json \
  --output-dir outputs/1920/surface_regions_repeat
```

同时指定 `--regions` 与 `--annotate-regions` 可加载标注到窗口继续编辑；标注图尺寸必须一致。原来在原图上标注的文件，编辑时仍应提供同一张 `--annotation-image`。同名输出会覆盖，试验参数时使用独立目录。

## 3. JSON 格式

以下只是格式示例，矩形不是当前叶腔的真实人工边界；请使用实际标注：

```json
{
  "image_size": [504, 280],
  "regions": [
    {
      "id": 1,
      "name": "left",
      "polygons": [[[20, 10], [180, 10], [180, 260], [20, 260]]],
      "holes": [[[80, 100], [100, 100], [100, 120], [80, 120]]]
    },
    {
      "id": 2,
      "name": "right",
      "polygons": [[[300, 10], [490, 10], [490, 260], [300, 260]]],
      "holes": []
    }
  ],
  "exclude_polygons": []
}
```

- `image_size` 为标注图的 `[W,H]`，顶点为该图的 `[u,v]` 坐标；末点不必重复首点。
- `id` 是 1～65535 的唯一整数，0 保留给忽略区域。JSON 可定义更多区域，窗口数字键可切换前 9 个区域。
- `polygons` 是同一区域的多个外多边形的并集，`holes` 是从该区域扣除的多边形。
- `exclude_polygons` 可选，对所有区域统一扣除。
- 原图标注必须额外提供 `original_to_processed`，包含缩放、裁剪偏移等变换。不同尺寸且没有变换时会拒绝输入。

例如当前 1920×1080 原图映射到 504×280 处理图像，可保存：

```json
"original_to_processed": [
  [0.2625, 0, 0],
  [0, 0.25925925925925924, 0],
  [0, 0, 1]
]
```

标注通过变换后，在处理图像的像素中心计算多边形归属；不将浮点多边形提前取整。原图文件与点云是否来自同一帧仍需调用者确认。

## 4. 拟合与网格行为

1. 为每个点确定区域编号，未标注、孔洞及排除区域记为 0。
2. 每个区域仅用本区域的有效点训练；置信度筛选与原有设置相同。
3. 使用该区域像素的包围盒作为局部参数域，因此控制点分配到本区域，而非分配到整张图像。
4. 每个区域有独立的系数、深度数值归一化尺度、稳健权重和训练采样。修改一个区域的观测不会影响另一个区域的拟合。
5. 保留原来的图像空间块验证划分；验证点不进入训练。`--max-samples` 是每个区域的训练上限。
6. `--controls-u/v`、`--smoothness`、`--relative-threshold` 等设置应用于每个区域。默认仍是 16×10 控制点、0.05 平滑、5% 残差阈值。
7. 三角网格沿处理图像邻接关系建立，整个采样单元必须有效且不跨区域边界。区域交界处会留出窄的未连接带，不自动焊接，也不强制法向连续。

多个不相连的多边形若使用同一 ID，会共享一个区域模型；如果希望两块表面独立，请分配不同 ID。当前版本不自动细分区域或自动调整每个区域的控制点数。

区域 0 的拟合点和法向保存为 NaN，`eligible/evaluated/inlier/train/validation` 均为 false。多边形内也仍需通过深度有效性、置信度及拟合残差检查，不能把区域标注直接等同于可靠表面。

## 5. 输出

保留原有 `surface.npz`、表面 PLY、`model.json`、`report.json` 和诊断图接口，新增：

| 文件或字段 | 内容 |
| --- | --- |
| `regions.json` | 原始标注坐标与变换，可用于再次编辑 |
| `regions_processed.json` | 已映射到处理图像坐标的多边形 |
| `region_mask.png` | 16 位整数区域图，0 为忽略区域，其他值为区域 ID |
| `region_overlay.png` | 处理图像上的区域颜色与边界预览 |
| `surface.npz: region_ids` | 每个输入点的区域 ID，点顺序保持不变 |
| `surface.npz: region_mask` | 与处理图像同尺寸的区域图 |
| `surface.npz: face_region_ids` | 每个三角面的所属区域 |
| `model.json: regions` | 各区域独立模型；顶层类型为 `piecewise_perspective_inverse_depth_bspline` |
| `report.json: regions` | 各区域点数、训练/验证误差、收敛情况和网格面数 |

`region_overlay.png` 的颜色依次为青色、橙色及后续调色板；具体名称与 ID 以 JSON 为准。`--no-plots` 下仍保存区域预览和掩码，需要 Pillow。

再次输入或编辑时使用 `regions.json`。`regions_processed.json` 用于检查变换后的坐标；包含裁剪的映射可能使部分顶点位于处理图像之外，区域计算会按处理图像范围限制。

`evaluate_surface(model, pixels, K)` 同时兼容旧的全局模型与新的分区模型。分区模型会按多边形分派亚像素查询，区域外返回无效。即使查询有效，绘制时仍应使用保存的观测和 inlier 支持约束。

## 6. 继续估计轴线与绘制网格

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.axis \
  --surface outputs/1920/surface_regions/surface.npz \
  --output-dir outputs/1920/axis_regions

PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.grid \
  --surface outputs/1920/surface_regions/surface.npz \
  --axis outputs/1920/axis_regions/axis.json \
  --original-image images/1920_45_000001.jpg --resize-only \
  --output-dir outputs/1920/grid_regions
```

若只比较表面方案对截线的影响，可以临时复用同一帧、同内参和尺寸的已有轴线。正式结果仍应结合新的有效表面复查轴线。网格预览使用分区网格，保持区域之间断开，`coordinates.npz` 也会保留 `region_ids`。

## 7. 验证与限制

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_surface_regions.py tests/test_surface.py \
  tests/test_pointcloud.py tests/test_axis.py tests/test_grid.py
```

验证包含：两个不同平面独立恢复、改变右区数据不影响左区、区域外无效、局部参数域法向、孔洞和原图变换、自交/重叠拒绝、区域内小岛不被粗网格跨越、标注窗口事件保存，以及分区输出继续被轴线加载器和网格命令使用。

交互事件在无显示后端下模拟验证，实际桌面显示仍需本机 GUI 环境。分区能阻止跨部件的拟合影响，但无法修复 DA3 在单一区域内部的系统性深度错误。当前尚未使用真实人工叶腔多边形验证贴合效果，需要先完成标注，再检查区域图、残差、三维表面和截线。
