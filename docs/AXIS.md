# 第三步：轴线估计与可视化验证

本步骤读取第二步的 `surface.npz`，估计一条直结构轴线候选，输出二维叠加、三维验证、真实曲面截线和稳定性报告。不重新运行 DA3，不需要掩码或 GPU。

**自动结果始终是候选，不自动认定为真实物理轴线。** 圆柱/锥面用于估计轴线，最终试探截线取自第二步的有效曲面，而不是用理想圆环替代真实截面。

## 自动运行

在仓库根目录执行：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.axis \
  --surface outputs/surface/surface.npz \
  --output-dir outputs/axis
```

需要 NumPy、SciPy；生成静态图还需要 Matplotlib。`--no-plots` 可只输出数值和 PLY。默认使用 CPU。

## 自动估计方法

1. 读取第二步通过残差检查的点，以原始 DA3 三维点拟合候选，以拟合曲面的法向提供较弱的方向约束。保留第二步的空间块验证划分。
2. 用相机前向、远处点方向、法向变化方向和若干倾斜方向初始化。这里没有自动将 RGB 边线认定为平行线，也没有将整片点云 PCA 第一主方向直接认定为轴向。
3. 分别稳健拟合圆柱及圆锥假设。各模型的轴线写作 `P(s) = c + s*a`，圆锥半径写作 `r(s) = r0 + k*s`。
4. 根据验证点的截断相对几何残差排序；圆锥相比圆柱改善不足 5% 时，优先选择圆柱。此验证集用于选择候选，报告不是独立测试集成绩。
5. 对选定模型进行 12 次有放回重采样，检查轴线方向、参考位置的变化，并比较残差相近的其他候选。
6. 用垂直于候选轴线的 5 个平面与有效三角网格求交。截线不跨越已有孔洞；再检查截面圆拟合是否有足够角度支持、中心是否偏离候选轴线。

自动方向采用 `[tx,ty,1]` 归一化，`tx,ty` 限制在 [-2,2]，适用于朝结构内部拍摄的视角；达到边界会提示可能退化。它不适合近乎横向观察的结构。三维曲线轴、椭圆截面轴及扭转中心线尚不在此版本范围内。

## 固定使用相机 Z 轴查看截面

可跳过轴线估计，直接使用相机光轴 `x=y=0`、方向 `[0,0,1]`：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/da3-matplotlib OPENBLAS_NUM_THREADS=1 \
  /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.axis \
  --surface outputs/surface/surface.npz \
  --camera-z-axis \
  --output-dir outputs/axis_z
```

此模式不拟合圆柱/圆锥轴线，也不进行轴线重采样。轴上参考点使用 `[0,0,有效网格顶点的中位Z]`，它始终位于相机 Z 轴上；截平面是 `Z=常数`，位置仍取网格轴向坐标的 12%～88% 分位范围，再等距放置默认 5 个平面。`--sections` 可调整数量。

`--camera-z-axis` 同时固定方向与位置；原有 `--direction 0 0 1` 只固定方向，仍会估计轴线位置。Z 轴模式不能与其他轴线参数或参考线标注组合。

相机 Z 轴在二维图像中投影到主点 `[cx,cy]`，因此以叉号显示，不会出现一条有长度的轴线投影。三维图仍会显示轴线。截线继续由拟合表面网格求交得到，保留原有孔洞及分区断开。

输出继续兼容后续 grid：`axis.json` 增加 `axis_source: "camera_z"` 和绝对深度 `section_depths`；`report.json` 的轴线模型验证残差为 null，重采样次数为 0。截面形状诊断仍会执行。该模式用于比较查看方向的影响，不自动确认相机 Z 轴就是结构主轴。

如需直接对原始点云切片，使用独立入口 `depth_anything_3.pointcloud_sections`，详见 [原始点云 Z 轴截面与投影](POINTCLOUD_SECTIONS.md)。该入口不读取 `surface.npz`，可通过 `--depths-from-axis` 复用相同截面深度。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `axis_overlay.png` | 在处理后 RGB 上叠加紫色虚线轴、青色方向消失点、彩色截线 |
| `axis_validation.png` | 图像叠加、三维曲面及轴线、横截面形状、诊断说明 |
| `axis.json` | 相机坐标系中的轴上参考点、单位方向、截面位置、模型及状态 |
| `report.json` | 候选比较、残差、重采样、各截面支持及需要检查的问题 |
| `sections.npz` | 三维截线段、处理图像坐标、截面编号、轴线采样、重采样轴线 |
| `axis_points.ply` / `section_points.ply` | 可导入点云查看器，与表面 PLY 同坐标叠加检查 |
| `annotations.example.json` | 空的参考线标注模板，不包含虚构参考点 |

`sections.npz` 中 `segments` 的形状为 `(N,2,3)`，每一项是一个独立三角面交线段；`section_ids` 对应 `levels` 的索引。`segments_uv` 为同一线段在模型处理图像上的投影。`bootstrap_axes` 每行依次为 `[cx,cy,cz,ax,ay,az]`。

默认参考点 c 位于训练点中位 z 所在的相机横截面。s=0 只是这条轴线的参考原点，并非叶片根部。所有长度沿用 DA3 的预测单位，不自动变成米。

**青色叉号是由候选方向推导的消失点，不是识别到的空腔中心。** 对于渐缩结构，锥面边线的汇聚位置可能是锥顶投影，不能与轴向平行线的消失点混用。紫色轴线穿过空腔，用于验证位置，因此不会像最终表面网格一样做表面遮挡裁剪。

## 如何判断结果

先打开 `axis_validation.png`：

- 检查方向是否符合真实结构延伸趋势，而不只看颜色截线是否平滑。
- 检查横截面中心是否沿轴线排列，截面是否由足够宽的可见弧段支持。
- 检查远处低置信度区、左侧板面与右侧曲面是否影响了候选模型。
- 比较候选及重采样差异；变化小只意味着对当前数据较稳定，不能排除 DA3 的系统性形变。

当前诊断启发式包括：模型中位/P90 残差超过估计半径的 10%/25%、重采样方向 P95 超过 5°、位置 P95 超过半径的 15%、相近残差候选方向相差超过 10°、或超过一半的试探截面缺少可靠圆形支持等。触发时状态为 `needs_review`。

`circle_reliable` 仅表示该截面通过当前圆拟合检查（至少 120° 角度覆盖、较低残差、非退化拟合），不表示它是物理上正确的圆截面。`candidate_consistent` 同样只表示没有触发这些启发式。`accepted_as_physical_axis` 始终为 false。

## 人工修正方法一：直接指定三维参数

只固定方向，让程序重新估计轴线位置：

```bash
PYTHONPATH=src /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.axis \
  --direction 0 0 1 \
  --output-dir outputs/axis_direction_trial
```

这里 `0 0 1` 只是演示相机前向，不是对当前图像的正确方向断言。

使用 `--axis-point X Y Z` 可固定轴线上一个三维点；不同时指定方向时会继续估计方向。位置必须是相机坐标中的空腔轴心参考点，不能直接读取“二维空腔中心像素”的壁面深度作为轴心深度。

也可以复制 `axis.json`，修改 `point` / `direction`，再重新生成验证结果：

```bash
PYTHONPATH=src /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.axis \
  --axis-json outputs/axis_adjusted.json \
  --output-dir outputs/axis_adjusted
```

`--axis-json` 将方向和位置同时固定；`--direction` / `--axis-point` 可覆盖文件对应值。剩余的圆柱/锥面半径参数仍会拟合，以提供残差诊断。固定参数不会因拟合自行漂移，报告也会注明重采样不能验证人工约束是否正确。

## 人工修正方法二：点击参考线

有桌面显示的本机终端可运行：

```bash
PYTHONPATH=src /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.axis \
  --annotate \
  --output-dir outputs/axis_annotated
```

窗口中：

1. 沿一条**真实空间中平直、且平行于结构轴向**的接缝点击 2～5 点，按 Enter 完成该线。
2. 标注至少另一条空间位置不同的平行线。推荐三条，以便检查方向一致性。
3. Backspace 撤销；S 保存并开始估计；Esc 取消。

标注保存为 `annotations.json`，可再次使用：

```bash
PYTHONPATH=src /home/nvidia/venvs/da3_env/bin/python -m depth_anything_3.axis \
  --annotations outputs/axis_annotated/annotations.json \
  --output-dir outputs/axis_annotated_repeat
```

标注得到的方向作为固定约束，位置继续由点云估计。**锥面上逐渐汇聚的母线、弯曲接缝、光斑边缘不能作为平行线标注。** 如果无法确认平行性，应采用三维参数修正，或引入已知尺寸/额外视角。

默认标注坐标与处理图像一致，本例为 504×280。JSON 结构为 `image_size: [W,H]` 和 `parallel_lines: [[[u,v], ...], ...]`。从原始图像标注时，必须另外提供准确的 `original_to_processed` 3×3 变换矩阵，包含缩放与裁剪；尺寸不匹配且未提供变换时会拒绝处理。

点击窗口需要可用的 Matplotlib GUI 后端。静态输出及 JSON 标注输入适用于无显示环境；本次自动验证运行没有打开桌面窗口。

## 当前图像的运行记录

当前自动选择圆锥假设，候选方向约为 `[-0.16199, -0.01361, 0.98670]`，参考点约为 `[0.04073, -0.04998, 0.92445]`。

12 次重采样全部成功，方向变化 P95 约 1.19°，参考位置变化 P95 约为半径的 1.82%。验证点的模型距离中位数约为半径的 7.21%，P90 约为 16.90%。这些指标与第二步的“相对 z 深度残差”定义不同。

5 个试探截面中，只有 2 个通过圆形支持检查，因此最终标记为 `needs_review`。这表示自动计算和可视化已完成，但在据此生成结构化网格前，需要检查圆形截面假设是否符合实际结构。

## 测试

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_axis.py tests/test_surface.py tests/test_pointcloud.py
```

覆盖已知倾斜圆柱/圆锥轴线恢复、模型选择、人工参数固定、平行线方向及坐标变换、退化标注拒绝、网格交线及孔洞、截面圆形支持和命令行导出。测试与自动运行不代表交互窗口已在当前桌面环境中验证。
