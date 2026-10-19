# Constraint Traverse 数据采集

该目录用于采集 AGX 在窄门、窄通道、家具间隙和窄转角中的人工连续控制示范。目前只实现数据采集、完整性校验和离线回放，采集程序不会加载主导航、NavDP、训练或强化学习模块。

## 采集内容

每条成功示范保存：

- 前视 RGB 和深度；
- 下视深度；
- robot-centric 三通道 BEV：`occupied/free/unknown`；
- 世界位姿、实测速度、人工控制量和实际执行控制量；
- 转向关节与车轮速度；
- 相机内参和当前外参；
- 人工确认成功时，最后一组同步观测对应的二维世界终点坐标；
- 采集开始时的完整体素地图状态，以及本次采集使用的地图融合帧索引；
- 每个物理步的状态、人工控制量、执行控制量及各自的时间与物理步编号。

## 启动

在仓库根目录运行：

```bash
conda activate vlmnav
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python -m constraint_traverse.collect.collect \
  --usd /Data/qiang/interiornav_data/scene_data/kujiale_0067/kujiale_0067.usda \
  --output /Data/qiang/data/constraint_traverse
```

场景加载完成后，程序暂停物理计算并等待手动设置机器人出生位置。

## 操作

| 按键 | 动作 |
| --- | --- |
| `W` 或 `↑` | 前进 |
| `S` 或 `↓` | 后退 |
| `A` 或 `←` | 左转 |
| `D` 或 `→` | 右转 |

按住 `↑+←/→` 或 `W+A/D` 可以边前进边转弯；按住 `↓+←/→` 或 `S+A/D` 可以边后退边转弯；单独按 `←/→` 或 `A/D` 原地转向。

采集流程：

```text
点击 BORN
→ 使用 Isaac Sim 移动和旋转工具放置机器人
→ 再次点击 BORN
→ 等待 Goal Map 建立可通行区域
→ 可在 Goal Map 的已知自由区域点击示意点
→ START DEMO
→ 人工穿越受限空间
→ 在终点停稳
→ SUCCESS
```

- `BORN`：第一次点击暂停物理并选中机器人，第二次点击恢复物理并确认出生位姿。
- `Goal Map`：`IDLE` 状态持续使用前视、下视 RGB-D、`LocalVoxelMap` 和 `render_bev_goal_map` 建图。显示范围、分辨率和刷新周期读取主导航 `map_selector`，当前为 `10 m` 视野、`1.5 m` 后方范围和 `420×420` 像素。点击已知自由区域后，红色圆点表示驾驶示意点。点击位置使用绘制这张地图时的机器人位姿转换为世界坐标。
- `RESET MAP`：在 `IDLE` 状态清空当前局部地图。
- `START DEMO`：停稳后记录实际起点并开始采集，沿用当前地图，同时保存初始地图状态。示意点可以根据采集需要选取。
- `SUCCESS`：由操作者确认完成穿越。停稳后点击，程序保持停止指令，等待一组新的同步观测，将该组状态中的实际 XY 记录为训练目标并保存示范。
- `DISCARD`：删除当前示范。
- `RESET TO START`：删除当前未完成示范并回到最近一次起点。

`Collector Controls` 面板显示 `State`、`Episode`、`Samples`、`Time`、当前操作步骤和键盘说明，提供 `BORN`、`RESET MAP`、`START DEMO`、`SUCCESS`、`DISCARD` 和 `RESET TO START` 按钮。

目标在结束采集时写入 `meta.json` 的 `goal_world_xy: [x, y]`。终点航向作为车辆状态保存。训练和回放根据 `goal_world_xy` 与每帧 `pose_world_xyyaw` 计算机器人坐标系中的相对目标。整条示范使用同一个实际终点作为训练目标。

`success` 保存操作者的人工成功标注，`valid` 表示数据完整性和时间关联检查的结果。关闭采集程序或采集超时时，已经写入的批次保留为未完成记录。

## 数据目录

```text
<output>/<scene_name>/episode_000001/
├── meta.json
├── state.npz
├── controls.npz
├── initial_map.npz
├── chunks/
├── front_rgb/
├── front_depth/
├── down_depth/
└── bev/
```

新采集记录的 `schema_version` 为 `2`。`state.npz` 保存每组同步观测对应的状态与策略指令，`controls.npz` 保存连续物理步的状态和控制。`sample_time_sec` 与 `state_timestamp_sec` 对应相机渲染时刻，`action_timestamp_sec` 对应读取观测后指令的生效时刻，两者通过物理步编号关联。

人工速度指令在同步观测到达时更新，并保持到下一次观测。底层每个物理步继续执行速度适配器。`human_action_vw` 是适配器的输入，`applied_action_vw` 是适配器处理后的执行指令，模仿学习与强化学习应使用相同的动作周期和适配器。

地图融合使用与 PNG 保存精度一致的毫米深度。`map_integrated` 标记实际参与融合的观测；结合 `initial_map.npz`、相机变换、地图配置和保存的深度，可以逐帧重建 BEV。

深度保存为 uint16 毫米 PNG：

```text
0      无效深度
65535  无返回值
其他值 深度毫米数
```

## 校验

每条 episode 保存后会自动检查时间关联、连续物理步记录、数组与文件数量、数据形状、深度有效比例、动作范围、首尾位姿和 BEV 格式。检查失败的数据会保存为 `valid: false` 并记录具体错误。保存时与离线检查共用同一个校验函数。

手动检查整个数据目录：

```bash
python -m constraint_traverse.collect.validate \
  /Data/qiang/data/constraint_traverse
```

重建并核对一条记录的全部 BEV：

```bash
python -m constraint_traverse.collect.validate \
  /Data/qiang/data/constraint_traverse/<scene_name>/episode_000001 \
  --rebuild-bev
```

采集开始时会立即写入元数据与初始地图。传感器文件与状态按批次写入，批次完成后原子提交 `chunks/*.npz`。未完成记录保存在 `.episode_XXXXXX.tmp`；校验命令会报告这些目录。异常中断后，可以恢复已经完整提交的批次：

```bash
python -m constraint_traverse.collect.validate \
  /Data/qiang/data/constraint_traverse/<scene_name>/.episode_000001.tmp \
  --recover
```

恢复操作重建 `state.npz` 和 `controls.npz`，并保留 `status: incomplete`、`success: false`。尚未完成写入的批次可能丢失。`DISCARD` 删除当前示范，`RESET TO START` 删除当前未完成示范并恢复起点。

如果成功记录已经完整写入，程序在目录重命名期间中断，恢复操作会核验这份完整记录并完成目录重命名，保留已经保存的人工成功标注。

## 回放

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse/<scene_name>/episode_000001
```

回放窗口同步显示前视 RGB、两路深度、BEV、世界位姿、相对目标、人工控制量和实际执行控制量，默认按保存的时间间隔回放。`--fps` 可以指定回放速度。

当 OpenCV 缺少 GUI 后端时，回放器自动在场景数据目录生成 `<episode_name>_replay.mp4`。也可以指定输出位置：

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse/<scene_name>/episode_000001 \
  --fps 10 \
  --output /tmp/episode_000001_replay.mp4
```

## 配置

采集配置位于 [`configs/collect.yaml`](configs/collect.yaml)，包括仿真设备、手动放置、BEV、连续控制速度、episode 时长和数据质量阈值。前视与下视相机路径读取 `configs/robots/agx.yaml`，相机分辨率、频率、帧同步阈值、地图融合周期和 `LocalVoxelMap` 参数读取 `configs/navigation.yaml`，与主导航保持一致。
