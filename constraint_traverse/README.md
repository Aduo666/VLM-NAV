# 手动驾驶采集说明

## 安装到 Python 环境

在已有的 `vlmnav` 环境中，把当前仓库安装一次：

```bash
conda activate vlmnav
cd /home/Lab/qiangsibin/VLM-NAV-active
python -m pip install -e .
```

安装命令必须在仓库根目录运行，这个目录里面有 `pyproject.toml`。

`.` 表示当前仓库，`-e` 表示直接使用仓库里的源码。以后修改代码，下一次启动程序就会使用修改后的代码。更换 conda 环境或移动仓库目录后，需要重新执行安装命令。

安装完成后，下面的采集、检查和回放命令都可以直接运行。每次打开新终端时，先执行 `conda activate vlmnav`。

采集使用已有 `vlmnav` 环境中的 Isaac Sim。上面的命令会安装本仓库和图像处理等通用依赖。

完整仿真环境的依赖版本记录在根目录的 [requirements.txt](../requirements.txt)。如果正在准备新的 Python 3.11 环境，可以在仓库根目录安装这些依赖，再执行上面的仓库安装命令：

```bash
python -m pip install -r requirements.txt
```

## 开始采集

下面这条命令会启动 Isaac Sim，加载 `kujiale_0067` 场景，并准备录制手动驾驶数据：

```bash
python -m constraint_traverse.collect.collect \
  --usd /Data/qiang/interiornav_data/scene_data/kujiale_0067/kujiale_0067.usda \
  --output /Data/qiang/data/constraint_traverse
```

- `--usd`：要加载的场景文件。换场景时修改这个路径。
- `--output`：保存采集数据的文件夹。程序会在里面按场景名称和记录编号建立子目录。
- `--config`：可以指定采集配置文件，默认使用 [configs/collect.yaml](configs/collect.yaml)。

场景加载完成后，按下面的步骤操作：

1. 点击 `BORN`，使用 Isaac Sim 的移动、旋转工具摆放小车。
2. 再次点击 `BORN`，确认小车位置，等待地图显示附近环境。
3. 可以在地图上点击一个示意点，帮助自己确定驾驶方向。
4. 小车停稳后，点击 `START DEMO`，开始录制。
5. 手动驾驶小车穿过门口或通道。
6. 到达你认为合适的位置并停稳后，点击 `SUCCESS`。

`START DEMO` 会保留当前地图，并保存开始采集时的地图状态。你可以在开始采集前使用 `RESET MAP` 决定是否清空地图。

启动时，小车车体中心的 Z 坐标为 0.32 米，场景地面位于 Z 约为 0 米的位置。确认放置前保持物理暂停；点击 `BORN` 后会关闭拖动吸附，可以连续移动小车。

点击 `SUCCESS` 表示你人工确认完成了这次穿越。程序会保持停止指令，取得最后一组图像和车辆状态，把其中的实际位置保存为这条示范的训练目标。

## 如何驾驶小车

| 按键 | 小车动作 |
| --- | --- |
| `W` 或 `↑` | 前进 |
| `S` 或 `↓` | 后退 |
| `A` 或 `←` | 原地左转 |
| `D` 或 `→` | 原地右转 |
| 同时按住前进键和左转、右转键 | 前进时转弯 |
| 同时按住后退键和左转、右转键 | 后退时转弯 |

按键顺序没有要求，行驶过程中按住方向键就会同时移动和转弯。松开方向键后逐渐恢复直行；松开前进或后退键后逐渐转为原地转向；全部松开后逐渐停止，四个车轮自动回正，朝向车体前方。

停车回正时，转向关节的最大力矩默认提高到 50 牛米，重新驾驶时恢复模型原有设置。这个上限由采集配置中的 `collection.steering_centering_max_effort_nm` 控制。

驾驶时，方向键和 WASD 由小车控制程序处理。进入 `BORN` 放置或暂停仿真后，这些按键交回 Isaac Sim 编辑工具。

低速转弯时，程序会根据底盘允许的车轮转向角调整转弯速度，保留前进或后退速度。

## 窗口和按钮的用途

| 名称 | 用途 |
| --- | --- |
| `Goal Map` | 显示小车附近的障碍物和可通行区域。点击可通行区域会显示一个红色示意点。 |
| `BORN` | 进入或结束小车位置调整。 |
| `RESET MAP` | 在尚未录制时清空当前地图。 |
| `START DEMO` | 保存起点并开始录制。 |
| `SUCCESS` | 人工确认成功，取得实际终点并保存记录。 |
| `DISCARD` | 删除正在录制的这一条记录。 |
| `RESET TO START` | 删除当前未完成记录，回到最近一次起点，并重新建立地图。 |

`Collector Controls` 窗口显示当前状态、记录编号、已经保存的帧数和录制时间。

## 查看已经录制的数据

这条命令用于查看第一条记录中的图像、深度、周围地图、行驶轨迹和控制指令：

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse/kujiale_0067/episode_000001
```

查看第二条时，把末尾目录名改成 `episode_000002`。这里需要填写一条记录的完整目录。

OpenCV 支持显示窗口时，程序会打开回放窗口：空格暂停或继续，`Q` 或 `Esc` 退出。如果 OpenCV 没有窗口支持，程序会在场景目录生成 `episode_000001_replay.mp4`。

## 导出回放视频

这条命令把第一条记录导出为指定的视频文件：

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse/kujiale_0067/episode_000001 \
  --output /tmp/episode_000001_replay.mp4
```

回放命令中的 `--output` 是视频文件路径。默认使用记录中的时间间隔确定回放速度。

如果希望按每秒 10 帧播放，可以加上 `--fps 10`：

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse/kujiale_0067/episode_000001 \
  --fps 10 \
  --output /tmp/episode_000001_replay.mp4
```

`--fps` 控制回放速度。这批数据约为每秒 14.75 帧，使用 `--fps 10` 时播放得更慢。

## 批量生成回放视频

下面的命令会查找数据总目录中的所有已保存记录，为每条记录生成一个 H.264 编码的 MP4 视频，可以直接在 VS Code 中打开：

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse \
  --batch \
  --output-dir /Data/qiang/data/constraint_traverse_videos
```

只生成某个场景时，填写该场景目录，输出目录可以继续使用同一个位置：

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse/kujiale_0067 \
  --batch \
  --output-dir /Data/qiang/data/constraint_traverse_videos
```

输出按场景和记录编号组织，例如：

```text
constraint_traverse_videos/
  kujiale_0067/
    episode_000001.mp4
    episode_000002.mp4
```

- `--batch`：启用按目录批量生成，逐条显示进度。
- `--output-dir`：批量视频的输出目录，必须填写。
- `--overwrite`：重新生成已有视频。
- `--fps 10`：统一按每秒 10 帧生成；默认分别读取每条记录的采样帧率。

程序会检查已有视频的编码和完整帧数。与当前记录帧数一致的完整 H.264 视频会自动跳过；损坏、帧数不一致或编码不兼容的视频会重新生成。调整已有视频的播放帧率时，加上 `--overwrite`。

正在录制的临时目录会跳过。成功和失败的已保存记录都可以生成回放。某条记录生成失败后，终端会显示原因并继续处理后续记录，最后统计生成、跳过和失败数量。

按 `Ctrl+C` 可以中断。已经完成的视频会保留，正在生成的临时视频会清理，重新执行同一条命令即可继续。

批量模式需要系统能够执行 `ffmpeg` 和 `ffprobe`，并且 FFmpeg 支持 `libx264` 编码器。单条生成继续使用上面的原有命令。

## 检查数据是否完整

保存每条记录时，程序会自动检查数据。下面这条命令可以检查整个采集目录：

```bash
python -m constraint_traverse.collect.validate \
  /Data/qiang/data/constraint_traverse
```

它会检查图像和状态数量、文件能否读取、时间对应关系、控制步是否连续、深度是否有效、终点是否正确保存等。

- `[OK]`：这条记录通过检查。
- `[INVALID]`：发现问题，下面会列出具体原因。

这条命令只读取数据。

## 检查保存的地图能否重新生成

在普通检查的基础上，加上 `--rebuild-bev`，可以检查一条记录中的全部周围地图：

```bash
python -m constraint_traverse.collect.validate \
  /Data/qiang/data/constraint_traverse/kujiale_0067/episode_000001 \
  --rebuild-bev
```

程序会使用初始地图、保存的深度和相机位置重新计算每一帧地图，再和原来保存的 BEV 比较。这项检查比较耗时，只读取数据。

## 程序意外退出后恢复数据

未完成记录的目录名类似 `.episode_000003.tmp`。如果程序在录制期间意外退出，可以恢复已经完整写入的部分。

下面的目录名需要替换成实际存在的未完成目录：

```bash
python -m constraint_traverse.collect.validate \
  /Data/qiang/data/constraint_traverse/kujiale_0067/.episode_000003.tmp \
  --recover
```

这条命令会写入恢复后的状态文件、控制文件和元数据。恢复出来的未完成记录会保留未完成标记。尚未写完的部分可能丢失。

如果人工成功标注和全部数据已经写完，程序在最后重命名目录时中断，恢复命令会检查这份完整记录并完成保存。

## 每条记录保存哪些文件

```text
/Data/qiang/data/constraint_traverse/kujiale_0067/episode_000001/
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

| 文件或目录 | 内容 |
| --- | --- |
| `meta.json` | 场景、起点、实际终点、时长、成功标注和检查结果。 |
| `state.npz` | 每组观测对应的小车位置、朝向、速度、控制指令和相机信息。 |
| `controls.npz` | 每个物理计算步骤的小车状态和控制指令。 |
| `initial_map.npz` | 点击开始时已经建立的完整地图。 |
| `chunks/` | 采集期间分批保存的状态和控制记录，供中断恢复使用。 |
| `front_rgb/` | 前视彩色图像。 |
| `front_depth/` | 前视深度图。 |
| `down_depth/` | 底盘下视深度图。 |
| `bev/` | 小车周围的俯视地图，分别表示障碍、空闲和未知区域。 |

新数据使用格式版本 `schema_version: 2`。

一次图像采集称为一帧。两帧之间，仿真仍会继续计算小车运动，这些更细的计算步骤保存在 `controls.npz` 中。

`success: true` 表示你点击了成功按钮并完成保存；`valid: true` 表示程序的数据检查通过。完整保存的记录为 `status: complete`，中途中断的记录为 `status: incomplete`。

训练目标保存在 `goal_world_xy` 中，是最后一组车辆状态的实际位置。训练和回放根据每一帧小车的位置、朝向计算目标在小车前后、左右的相对位置。

前视、下视相机分别保存各自的拍摄时间和对应车辆状态。程序在配置允许的时间差内配对两路图像。控制指令按固定的仿真时间周期更新，每个物理步骤都会保存执行记录。

`human_action_vw` 保存人工给出的线速度和角速度，`applied_action_vw` 保存底盘控制器处理后的执行指令。

线速度表示前进、后退的速度，角速度表示左右转向的速度。

深度图保存为毫米数：`1000` 表示一米，`0` 表示无效值，`65535` 表示没有测到返回距离。

## 配置文件在哪里

- [configs/collect.yaml](configs/collect.yaml)：采集时的速度、加减速、录制时长和数据检查要求。
- [configs/navigation.yaml](../configs/navigation.yaml)：相机图像尺寸、更新频率、地图尺寸和地图更新周期。
- [configs/robots/agx.yaml](../configs/robots/agx.yaml)：小车模型、相机位置和轮组参数。

## 使用 Isaac Lab 训练

训练工具位于 `constraint_traverse/isaaclab_training/`。使用 Isaac Sim 5.1.0、Isaac Lab 官方 v2.3.2、RSL-RL 3.1.2 和 PyTorch 2.7.0。BC 学习人工给出的目标速度，输出继续经过原有轮组计算和底盘约束。

现有 57 条示范全部参加训练。验证和测试使用另外采集的数据。策略读取 BEV、目标相对位置、车辆速度、关节状态、上一动作和地图年龄；主模型包含 CNN、MLP 和 LSTM。

以下命令在项目根目录运行。专用训练环境和 Isaac Lab 源码位于 `/Data/qiang/constraint_traverse_training`。

```bash
cd /home/Lab/qiangsibin/VLM-NAV-active
conda activate /Data/qiang/constraint_traverse_training/env
export ISAACLAB_PATH=/Data/qiang/constraint_traverse_training/IsaacLab
```

检查数据并生成缓存：

```bash
python -m constraint_traverse.isaaclab_training prepare \
  --dataset /Data/qiang/data/constraint_traverse
```

这条命令检查数据完整性、控制周期、动作接口、地图时间及 BEV 重建。缓存保存在专用训练目录的 `cache` 中，原始采集数据保留。

运行模型和 PPO 接口测试：

```bash
python -m constraint_traverse.isaaclab_training check
```

启动 Isaac Lab 仿真前，加载已有 Isaac Sim 的运行环境：

```bash
source /Data/qiang/isaacsim5.1.0/setup_conda_env.sh
```

使用真实房间、机器人和相机执行仿真验收：

```bash
python -m constraint_traverse.isaaclab_training check-sim \
  --dataset /Data/qiang/data/constraint_traverse/kujiale_0067/episode_000001
```

这条命令执行初始状态恢复、相机和地图检查、临时 BC 更新、检查点恢复、Actor 权重迁移，以及一次完整 PPO 采样和更新。还会检查停稳到达、超时价值补偿和真实障碍物碰撞。检查使用的临时模型自动清理。

相机通过 Replicator 同步取得图像，取图期间保持物理时间不变。地图在物理步中处理新帧，策略读取决策前已经完成融合的地图。碰撞通过官方 PhysX 接触视图读取，并按接触对象区分地面和障碍物。

正式启动 BC 训练：

```bash
python -m constraint_traverse.isaaclab_training train \
  --model lstm --device cuda:0 --seed 0 --epochs 200 \
  --output /Data/qiang/constraint_traverse_training/runs/bc_lstm_seed0
```

`--epochs` 表示完整使用训练数据的次数，`--batch-size` 默认同时处理 8 条示范。`last.pt` 保存最近完成的一轮，每 10 轮另存一个检查点。检查点包含模型、归一化参数、数据清单、优化器和随机状态。

恢复训练时，`--epochs` 填写希望最终完成的总轮数：

```bash
python -m constraint_traverse.isaaclab_training train \
  --model lstm --device cuda:0 --epochs 200 \
  --resume /Data/qiang/constraint_traverse_training/runs/bc_lstm_seed0/last.pt \
  --output /Data/qiang/constraint_traverse_training/runs/bc_lstm_seed0
```

运行 CNN＋MLP 对照模型时，将 `--model` 改为 `mlp`，并指定另外的输出目录。两个模型使用相同的数据、随机种子和评估任务。

闭环评估命令中的数据目录填写另外采集的验证集或测试集：

```bash
python -m constraint_traverse.isaaclab_training evaluate \
  --checkpoint /Data/qiang/constraint_traverse_training/runs/bc_lstm_seed0/last.pt \
  --dataset /Data/qiang/data/constraint_traverse_validation \
  --perturbations 3 --seed 0
```

评估执行原始起点和扰动起点，扰动位置半径为 0.1 米、朝向范围为正负 5 度，并检查起点附近的障碍物。每个回合恢复对应的历史地图。结果显示成功率、碰撞率、超时率、停滞率、耗时、终点误差和轨迹长度。添加 `--gui` 可以显示仿真窗口；添加 `--output 文件路径.json` 可以保存结果。

到达要求距离目标不超过 0.15 米、实际线速度不超过 0.05 m/s、角速度不超过 0.08 rad/s，连续保持 0.3 秒。有效障碍物碰撞结束回合，60 秒超时记录为截断。碰撞、成功和超时使用明确的判定优先级，终止统计在自动重置前保存。

严格统计指定门洞的通过率时，通过 `--doors 标注文件.json` 提供门洞标注。JSON 的键是 `场景名称/episode_编号`，值是该轨迹需要经过的门洞列表；每个门洞包含 `center_world_xy`、`normal_world_xy`、`width_m`、`thickness_m`。方向指向门洞出口，坐标和尺寸使用米。结果记录标注数量和整车通过数量。

PPO 接入验收复用 BC 的 Actor。正式 PPO 训练的奖励权重和训练预算在 BC 闭环评估后确定。
