交付项 1 提供独立实验入口、键盘驾驶、地图选点、同步 RGB-D 记录和试验状态恢复。该入口记录手动驾驶过程，自动通行规划、NBV 和机械臂主动运动尚未接入。

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
5. `STOP` 停止记录和驱动命令；保持按键时需要释放后重新按下才能继续驾驶。
6. `RESET TRIAL` 结束当前记录，恢复已保存起点及目标，并检查实际恢复误差。
7. `RESET MAP` 清除当前显示地图；该操作与试验状态恢复独立。

面板包括 `Active Passage Controls`、`Active Passage Map` 和 `Active Passage RGB`，窗口可以移动。地图复用现有直接 RGB-D 二维地图，默认融合 `front` 和底盘下视 `down`，用途为显示与目标选择。三路相机均通过 `ObservationMapFusion` 提交真实观测。后续三维环境表示和规划方法仍待确定。

相机分工在 `cameras.py` 定义，分辨率、频率和前视视场读取 `configs/navigation.yaml`。每次同步渲染期间保持物理时间不变，保存 RGB、原始浮点深度、内参、实际相机世界位姿、底盘位姿和全部关节状态。相机位姿通过实际所属刚体计算。

| 相机标识 | 实际相机 | 用途 |
| --- | --- | --- |
| `front` | 云台 `PanCamera` | 前视导航观测与地图显示 |
| `arm` | 机械臂 `link6/Camera` | 可选地图融合、机械臂主动感知的数据来源 |
| `down` | 底盘 `dummy_base_link/DownCamera` | 近场地面观测与地图显示 |

机械臂末端下视相机暂不初始化。当前入口保存三路真实观测；机械臂移动与主动观察调度仍待接入。

控制面板的“地图融合”提供 `front`、`arm`、`down` 三个独立勾选项，记录期间也可以切换。其下方显示地图版本和各路相机曾观测过的栅格数量，来源覆盖可以重叠。

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
| `frames/*.npz` | 同时刻 RGB-D、相机内外参及完整机器人状态 |
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
| `simulation/snapshot.py` | 场景与机器人状态的采集、恢复及检查 |
| `simulation/ui.py` | 控制面板、目标地图和相机画面 |
| `simulation/verify.py` | 物理验收流程 |

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

公共接口、来源隔离和风险评价可以独立测试。旧功能迁移状态记录在 [MIGRATION.md](MIGRATION.md)。机械臂伸出及收纳、成对通行场景和自动通行执行仍待实现。
