独立实验入口提供键盘驾驶、地图选点、Smac State Lattice 路径搜索、PyTorch MPPI 闭环控制、同步 RGB-D 记录和试验状态恢复。

自动导航使用已安装的 Nav2 1.1.20 原生 State Lattice 搜索库，通过 Python 3.11 绑定调用，不需要启动 ROS 节点。编译依赖为 ROS 2 Humble 的 `nav2_smac_planner`、CMake、C++ 编译器和 pybind11；控制依赖为现有 `pytorch-mppi==0.9.1`。在 `vlmnav` 环境中安装绑定依赖并编译：

```bash
python -m pip install -r constraint_traverse/active_passage/traversal/requirements.txt
python -m constraint_traverse.active_passage.traversal.build
```

编译生成的 `.so` 用于当前 Python 3.11 环境，构建临时文件自动清理。换用其他环境后需要重新编译。

在已经安装 Isaac Sim 的 `vlmnav` 环境中启动空白地面实验：

```bash
python -m constraint_traverse.active_passage run
```

可通过 `--gpu 1` 使用第二张显卡。从远程终端启动且没有设置 `DISPLAY` 时，本机图形显示为 `:1`，可以在启动命令前添加 `DISPLAY=:1`。

加载现有室内场景，明确指定初始世界坐标和朝向角度：

```bash
python -m constraint_traverse.active_passage run \
  --usd /Data/qiang/interiornav_data/scene_data/kujiale_0004/kujiale_0004.usda \
  --start 5.4009 -1.4496 0.32 179.18
```

`--start` 的四个数依次为世界坐标 X、Y、Z（米）以及朝向（角度）。界面启动时物理仿真暂停，可以使用 USD 移动工具调整机器人位置，点击 `CONFIRM PLACEMENT` 恢复物理。再次点击 `PLACE ROBOT` 可以重新调整位置。

操作顺序：

1. 确认放置后，使用方向键或 WASD 驾驶，前进与转向可以同时按下。
2. 在地图中点击目标，目标记录为世界坐标 XYZ，仅在地图中显示标记。
3. `SAVE START` 保存当前机器人、场景、控制器、地图、传感器记录和随机状态。
4. `START RECORDING` 从当前状态开始记录，并将该状态保存为本次试验起点；目标在记录期间保持固定。
   `START NAVIGATION` 从当前状态启动自动导航并自动记录。两种开始操作互相独立；记录期间目标保持固定。
5. `STOP` 停止记录和驱动命令；保持按键时需要释放后重新按下才能继续驾驶。
6. `RESET TRIAL` 结束当前记录，恢复已保存起点及目标，并检查实际恢复误差。
7. `RESET MAP` 清除当前显示地图；该操作与试验状态恢复独立。

面板包括 `Active Passage Controls`、`Active Passage Map` 和 `Active Passage RGB`，窗口可以移动。地图使用直接 RGB-D 二维建图，默认融合 `front` 和底盘下视 `down`，同时用于显示、选点、路径搜索和控制代价。三路相机均通过 `ObservationMapFusion` 提交真实观测。

点击地图后，目标保存为固定世界坐标。Smac 根据已观测地图搜索包含朝向、前进、后退和原地旋转的参考路径；用户无需指定终点朝向。MPPI 沿参考路径滚动优化四轮控制，每个物理周期读取实测关节和底盘状态。手动驾驶与自动导航共用运动执行模型，积分周期读取 Isaac Sim 实际物理周期。配置位于 `constraint_traverse/configs/active_passage_navigation.yaml`，其中控制参数复用 `benchmark.yaml`。

UNKNOWN 保持禁行。起点完整车体、参考路径扫掠区域、准备执行的短动作及其制动范围都需要已知空闲证据，已知障碍和地图边界同样参与检查。观测不足时显示 `OBSERVATION_REQUIRED` 并停止前进，可以停止试验后通过手动观测补充地图。程序不会根据完整场景几何填补未知区域。

地图中的紫色线表示 Smac 参考路径，绿色线表示 MPPI 预测轨迹，蓝色线表示实际轨迹。自动导航在距离目标不超过 0.30 m、实测线速度不超过 0.025 m/s、角速度不超过 0.04 rad/s 并持续 0.30 s 后记录 `SUCCESS`。该结果表示目标到达；宽门验收另外检查车体尾部越过出口平面。物理接触报告覆盖全部机器人部件，正常轮胎地面支撑单独识别；其他环境碰撞终止试验并记录 `COLLISION`。

相机分工在 `cameras.py` 定义，分辨率、频率和前视视场读取 `configs/navigation.yaml`。每次同步渲染期间保持物理时间不变，保存 RGB、原始浮点深度、内参、实际相机世界位姿、底盘位姿和全部关节状态。相机位姿通过实际所属刚体计算。

建图会过滤机器人自身的深度回波：使用机器人可见三角表面及各刚体实际位姿，识别距离自身表面不超过 3 mm 的深度点。过滤标记单独保存为 `robot_mask`，原始 RGB-D 保持完整。被标记的像素不提交障碍终点，也不用于产生空闲射线。正常运行只使用机器人自身模型，场景物体标签仅在仿真验收中用于核对过滤结果。

| 相机标识 | 实际相机 | 用途 |
| --- | --- | --- |
| `front` | 云台 `PanCamera` | 前视导航观测与地图显示 |
| `arm` | 机械臂 `link6/Camera` | 可选地图融合、机械臂主动感知的数据来源 |
| `down` | 底盘 `dummy_base_link/DownCamera` | 近场地面观测与地图显示 |

机械臂末端下视相机暂不初始化。当前入口保存三路真实观测；机械臂移动与主动观察调度仍待接入。

控制面板的 `Map fusion` 提供 `front`、`arm`、`down` 三个独立勾选项，记录期间也可以切换。其下方显示地图版本和各路相机曾观测过的栅格数量，来源覆盖可以重叠。

启动时启用三路融合：

```bash
python -m constraint_traverse.active_passage run --map-sources front arm down
```

`--map-sources` 可以选择任意组合，仅写该选项并省略名称会关闭全部地图融合。相机采集与地图融合分别控制，未参与融合的图像仍会记录。

开关控制后续新观测，已有地图证据继续保留。关闭机械臂融合后，其已经提供的环境信息仍可使用。比较常规导航与主动感知时，通过 `RESET TRIAL` 恢复相同初始地图，或者使用 `RESET MAP` 清除地图。

融合接口按每帧实际相机位姿反投影，保留采集时间与来源。重复帧、倒序帧以及早于新来源启用时间的缓存帧不会再次提交。`RESET TRIAL` 同时恢复融合开关、去重历史和启用时间。

记录默认保存到 `constraint_traverse/data/active_passage/`，可通过 `--output` 指定位置。每条记录包含：

| 文件 | 用途 |
| --- | --- |
| `meta.json` | 配置、目标、代码版本及源码校验值、记录数量、恢复误差和终止原因 |
| `records.jsonl` | 物理状态、控制命令、图像索引、融合开关事件与地图更新来源 |
| `frames/*.npz` | 同时刻原始 RGB-D、自体像素过滤标记、相机内外参及完整机器人状态 |
| `plans/*.npz` | Smac 参考路径、方向、MPPI 控制序列、预测轨迹、候选轨迹及代价 |
| `initial_state.pkl` | 本程序产生的数值状态快照，供后续实验复现使用 |

停止、超时、主动重置、异常和退出的记录都会保留。手动记录的结束状态不自动表示完成通行。`timestamp_sec` 是恢复后的试验逻辑时间；状态记录中的 `physical_time_sec` 是 Isaac Sim 当前物理时间，两者分别保存。

`map_sources` 保存本条试验的初始融合条件；记录期间的变化写入 `MAP_SOURCES_CHANGED`。`MAP_UPDATE` 保存地图版本、实际融合的观测标识符、相机名称、采集时间与来源更新数量。记录检查会核实这些观测已经保存，且相机来源和时间一致。

检查记录完整性：

```bash
python -m constraint_traverse.active_passage inspect /完整路径/某条试验记录
```

运行 10 次物理恢复与记录验收；测试数据默认使用临时目录并自动清理：

```bash
python -m constraint_traverse.active_passage check-sim --gpu 1
```

增加 `--gui` 显示验收过程，增加 `--output 路径` 保留验收记录。指定室内场景进行无界面验收时，需要同时提供 `--usd` 和 `--start`。验收包含实际行驶、动态刚体及静态场景变换恢复、地图与随机状态恢复、同步图像检查、记录完整性和超时结果保存。PhysX 接触求解器缓存与 GPU 渲染缓存无法直接保存，其影响通过重复执行误差检查。

已完成的验收数据及检查范围记录在 [VALIDATION.md](VALIDATION.md)。

每个文件的职责：

| 文件或目录 | 用途 |
| --- | --- |
| `__main__.py` | 命令行入口与程序资源关闭 |
| `experiment.py` | 开始、停止、超时和恢复试验的流程 |
| `recording.py` | 保存与检查试验记录 |
| `state.py` | 数值状态快照、随机状态及误差标准 |
| `cameras.py` | 相机实际路径与用途分工 |
| `mapping.py` | 三路真实观测的统一融合、来源开关和去重历史 |
| `simulation/runtime.py` | Isaac Sim 机器人加载、键盘驾驶和地图显示 |
| `simulation/sensors.py` | 同步 RGB-D 读取与实际相机位姿 |
| `simulation/robot_filter.py` | 根据机器人自身表面与实际姿态过滤深度回波 |
| `simulation/snapshot.py` | 场景与机器人状态的采集、恢复及检查 |
| `simulation/ui.py` | 控制面板、目标地图和相机画面 |
| `simulation/verify.py` | 物理验收流程 |
| `simulation/verify_navigation.py` | 实际运动预测、空地及宽门重复通行验收 |
| `traversal/` | 原生 Smac 绑定、路径搜索、MPPI 调用与执行范围检查 |
| `simulation/contacts.py` | 轮胎支撑识别与实际障碍碰撞记录 |
| `simulation/overlay.py` | 参考路径、预测轨迹和实际轨迹显示 |

控制面板的标签、状态、地图统计和操作消息均为英文。记录开始后显示试验编号，完整保存路径输出到终端。

`contracts.py` 定义任务、环境信念、通行规划、机械臂观察和评价五个模块的公共接口。数据对象只携带数值、标识符和不可写数组，规划接口使用信念快照和机器人状态。

信息来源：

- `RealObservation` 包含真实相机帧及同一采集时刻的机器人状态。
- `PredictedObservation` 包含环境假设标识符及模拟相机帧。
- `GroundTruthGeometry` 表示供仿真和离线评价使用的完整几何。
- `ObservedGeometry` 必须记录对应真实观测标识符。
- `Belief.update()` 只接受真实观测；融合器的输出必须为 `ObservedGeometry`。

当前 `Belief` 实现真实观测历史、版本及来源检查。三维融合通过 `RealGeometryIntegrator` 接口接入；未提供融合器时，`observed_geometry` 保持 `None`。几何生成、未来观测预测和机械臂实际安全检查尚未实现。

`ArmObservationPlan` 检查伸出和收回轨迹的数据连续性、返回收纳姿态及安全检查结果类型。`safety_passed` 是实际安全检查模块填写的结果；该数据类型自身不证明运动安全。

`evaluation.py` 使用 NumPy 和公共数据类型，可以通过普通 Python 单元测试运行。

执行损失采用：

\[
L=c_{collision}\mathbf1[collision]+c_{incomplete}\mathbf1[incomplete]
 +c_{time}T+c_{clearance}\left[\max\left(0,\frac{d_{target}-d_{min}}{d_{target}}\right)\right]^2.
\]

所有权重和间距目标由调用者明确提供。`COMPLETED` 表示到达目标并完成通行，碰撞独立计数；无碰撞成功需要同时满足 `completed` 和 `not collided`。停止、超时、规划失败和碰撞终止均计入未完成损失。

`traversal_duration_sec` 记录通行阶段时间，观察成本通过 `observation_cost` 独立输入，记录端需要保证时间成本没有重复计价。没有执行运动的失败结果可使用 `minimum_clearance_m=None`；完成或执行过的通行过程在启用间距损失时必须提供最小间距。没有障碍时可使用正无穷间距。

风险定义为：

\[
\rho(Z)=\mathbb E[Z]+\beta\operatorname{CVaR}_{\alpha}(Z),\qquad 0\le\alpha<1.
\]

`alpha` 使用置信水平定义：计算最高损失侧 `1-alpha` 的概率质量。离散分位点处使用部分样本权重，不将整个分位点样本一并加入尾部。`alpha=0` 时 CVaR 等于期望损失。

观察价值定义为 `risk_before - risk_after - observation_cost`。`LossSamples` 记录每条损失对应的环境与执行误差标识符；观测分支可以扩展样本数量，但每个 `(environment_id, execution_noise_id)` 的总权重必须与观察前一致。环境假设的后验权重不能替代这项事前执行评价权重。

运行测试：

```bash
python -m unittest discover -s constraint_traverse/active_passage/tests -v
```

公共接口、来源隔离和风险评价可以独立测试。Smac 与执行检查测试需要完成上述原生绑定编译。旧功能迁移状态记录在 [MIGRATION.md](MIGRATION.md)。机械臂伸出及收纳、NBV 和成对隐藏障碍实验仍待实现。

运行自动导航验收：

```bash
python -m constraint_traverse.active_passage check-navigation --gpu 1 --repeats 2 --timeout 45
```

验收包含前进、倒退、连续转弯、原地旋转和制动的 0.2、1、4 秒预测误差，以及空地和宽门的重复通行。地图由测试中的多次人工放置与真实 RGB-D 采集建立，UNKNOWN 持续禁行。宽门默认总几何余量 0.40 m，通过 `--door-clearance` 指定。验收场景完整几何仅用于创建场景、离线间距评价及车体尾部判定，规划器只接收实际观测地图。临时记录自动删除，指定 `--output` 可保留。

`--checks motion` 单独检查运动与预测，`--checks navigation` 检查闭环通行及实际碰撞记录，`--checks contacts` 单独检查墙体接触记录，默认 `all` 包含全部检查。运动检查要求约 0.2 秒的位置误差不超过 2 cm、朝向误差不超过 3°，并检查实际运动方向和制动停稳；1 秒、4 秒误差完整报告。长时间开环预测仍有转向误差，不能据此声称四秒预测准确。
