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
- SUCCESS 时确定的世界目标和每帧相对目标。

## 启动

在仓库根目录运行：

```bash
conda activate vlmnav
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python -m constraint_traverse.collect.collect \
  --usd /Data/qiang/interiornav_data/scene_data/kujiale_0067/kujiale_0067.usda \
  --output /Data/qiang/data/constraint_traverse
```

程序会通过 PhysX 地面查询自动寻找无碰撞出生位置。指定位置时使用：

```bash
python -m constraint_traverse.collect.collect \
  --usd /absolute/path/to/scene.usda \
  --output /Data/qiang/data/constraint_traverse \
  --spawn-x 1.5 \
  --spawn-y 4.0 \
  --spawn-yaw -90
```

`--spawn-x` 和 `--spawn-y` 必须同时提供。`--spawn-yaw` 可以单独使用。

## 操作

| 按键 | 动作 |
| --- | --- |
| `W` | 前进 |
| `S` | 后退 |
| `A` | 左转 |
| `D` | 右转 |

组合按键可以生成前进转弯和后退转弯。

采集流程：

```text
IDLE 下驾驶到起点
→ 停稳
→ START DEMO
→ 地图预热
→ 人工穿越受限空间
→ 在终点停稳
→ SUCCESS
```

- `START DEMO`：记录起点、清空 episode 地图并开始采集。
- `SUCCESS`：记录当前静止位姿作为目标并保存成功示范。
- `FAILURE`：保存失败示范。
- `DISCARD`：删除当前示范。
- `RESET TO START`：删除当前未完成示范并回到最近一次起点。

## 数据目录

```text
<output>/<scene_name>/episode_000001/
├── meta.json
├── state.npz
├── front_rgb/
├── front_depth/
├── down_depth/
└── bev/
```

深度保存为 uint16 毫米 PNG：

```text
0      无效深度
65535  无返回值
其他值 深度毫米数
```

## 校验

每条 episode 保存后会自动检查时间同步、数组与文件数量、数据形状、深度有效比例、动作范围、首尾位姿、BEV 格式和相对目标。失败的数据仍会保存，但 `meta.json` 会记录 `valid: false` 和具体错误。

手动检查整个数据目录：

```bash
python -m constraint_traverse.collect.validate \
  /Data/qiang/data/constraint_traverse
```

## 回放

```bash
python -m constraint_traverse.collect.replay \
  /Data/qiang/data/constraint_traverse/<scene_name>/episode_000001
```

回放窗口同步显示前视 RGB、两路深度、BEV、世界位姿、相对目标、人工控制量和实际执行控制量。

## 配置

采集配置位于 [`configs/collect.yaml`](configs/collect.yaml)，包括仿真设备、自动出生、相机同步、体素地图、BEV、连续控制速度、episode 时长和数据质量阈值。
