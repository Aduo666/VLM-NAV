# 近场通行证据诊断

在仓库根目录、已安装 Isaac Sim 的 `vlmnav` 环境中运行：

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src:. python -m constraint_traverse.diagnostics.run_nearfield \
  --episode-id kujiale_0067_bedroom_001 \
  --reference-report debug/local/fine_failure_0002_10145c27.json
```

默认使用任务清单中的起始位置，加载主导航使用的场景、AGX 资产、前视相机和底盘下视相机，并读取现有的导航配置。诊断期间保持机器人静止。`--frames` 可以调整同步 RGB-D 帧组数，默认采集两组；`--gpu` 指定显卡，`--show-window` 显示 Isaac Sim 窗口。`--reference-report` 用于检查一次已保存的 `forward_2` 拒绝记录；省略时检查本次采集出现的未知栅格。

主导航启动时从 USD 重新计算全部 23 个碰撞体的保守范围，统一转换到 `dummy_base_link` 坐标系。四个轮胎分别建立部件，转向支撑、车身外壳、升降结构和机械臂等碰撞体也分别建立部件。核查报告保存到 `constraint_traverse/data/diagnostics/agx_footprint_<编号>.json`，导航日志记录报告路径。缺少碰撞几何或者核查数量不符时，轮廓构建会报告错误并终止启动。

几何范围使用 `UsdGeom.Boundable.ComputeExtentFromPlugins` 从实际形状参数和 Mesh 顶点计算，逐项记录已有 `extent` 是否与实际形状一致。全部规划部件使用保守包围体，并标记 `CONSERVATIVE_USD_BOUND_VERIFIED`。车轮与转向支撑采用完整旋转的解析包络，覆盖转向控制范围及转向过渡。当前占用证据仅使用转向部件内部始终被占据的中心范围，转向包络其他区域仍需要地图证据。

物理轮廓和额外 3 厘米安全余量分别计算。逐动作报告包含 `physical_unknown_count`、`safety_margin_unknown_count`、`physical_collision`、`safety_margin_collision`。两层中的障碍、地图边界和新增未知区域仍按硬约束处理。`comparison` 使用 USD 轮廓；`configured_footprint_comparison` 使用导航配置中的轮廓作同帧对照。

每次运行都会在 `constraint_traverse/data/diagnostics/` 创建独立目录：

- `report.json`：三项实验的逐栅格结果、车体几何和 PhysX 查询结果。
- `raw_rgbd.npz`：两路相机的原始 RGB、深度、内参和采集位姿。
- `map_stride_10.npz`、`map_stride_5.npz`：同一批 RGB-D 帧按两种深度采样间隔生成的体素证据和二维规划地图。
- `comparison.png`：两种规划地图并排显示；粉色标出车体模型新增覆盖的未知栅格，橙色标出仅安全余量覆盖的未知栅格，蓝色标出机器人位置。
- `configured_footprint_comparison.png`：同一批相机数据使用配置轮廓的对照图。

实验 A 查询固定位置前进 0.1 米时的 PhysX 环境几何。`physics.planning_swept_envelope` 报告二维规划足迹扫过的每个栅格在离地 0.05—1.50 米范围内的环境碰撞体命中情况；`physics.cell_overlap` 给出目标栅格的车轮高度和车身高度检查。`physics.status=PARTIAL_ROBOT_SWEEP` 表示有机器人碰撞体无法参与逐形状扫掠，此时不能把零命中解释为完整车体已经通过物理验证。场景加载时的物理错误也应结合 Isaac Sim 日志检查。

实验 B 用完全相同的 RGB-D 帧分别按 `pixel_stride=10` 与 `5` 建图。`reference_cells` 或 `comparison.cells` 包含逐栅格地图状态、相机来源位和相机投影。`BEHIND_CAMERA` 表示该高度的点位于相机后方；`OUTSIDE_IMAGE` 表示点位于图像边界外；`NEARER_SURFACE` 表示深度图在该像素处记录了更近的表面。

实验 C 从 AGX 的驱动关节找到四个车轮碰撞体，比较 USD 几何、配置中的车轮中心和二维车体模型。`wheel_geometry.reference_cell_coverage` 给出目标栅格与实际车轮碰撞几何沿前进方向扫过的二维包围盒距离。`wheel_geometry.status=PARTIAL_BOUNDS` 表示部分机器人碰撞体无法从 USD 取得有效包围盒；此时车轮比较仍可用，完整车身的覆盖结论仍不完整。

`usd_footprint_audit.status=COMPLETE_CONSERVATIVE_BOUNDS` 表示全部 23 个碰撞体已取得有效的保守几何范围。这个状态与 PhysX 的逐形状扫掠查询状态分别记录。

运行自动化检查：

```bash
PYTHONPATH=src:. python -m unittest discover -s constraint_traverse/diagnostics -p 'test_*.py' -v
```

其中 USD 资产核查测试需要加载 Isaac Sim 的 `pxr` 环境。普通 Python 环境会跳过这组测试；运行时应检查测试输出中的跳过数量。
