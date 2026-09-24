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
- 采集开始前确定的世界目标和每帧相对目标。

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

组合按键可以生成前进转弯和后退转弯。

采集流程：

```text
点击 BORN
→ 使用 Isaac Sim 移动和旋转工具放置机器人
→ 再次点击 BORN
→ 等待 Goal Map 建立可通行区域
→ 在 Goal Map 的已知自由区域点击目标
→ START DEMO
→ 地图预热
→ 人工穿越受限空间
→ 在终点停稳
→ SUCCESS
```

- `BORN`：第一次点击暂停物理并选中机器人，第二次点击恢复物理并确认出生位姿。
- `Goal Map`：`IDLE` 状态持续使用主导航的前视、下视 RGB-D 和 `LocalVoxelMap` 建图。点击已知自由区域后，红色圆点表示目标；场景中不创建任何目标几何或标记。
- `RESET MAP`：在 `IDLE` 状态清空当前局部地图。
- `START DEMO`：锁定已确认目标、记录起点、清空 episode 地图并开始采集。
- `SUCCESS`：保存成功示范，保存的目标保持为采集开始前确认的目标。
- `DISCARD`：删除当前示范。
- `RESET TO START`：删除当前未完成示范并回到最近一次起点。

`Collector Controls` 面板显示 `State`、`Episode`、`Samples`、`Time`、当前操作步骤和键盘说明，提供 `BORN`、`RESET MAP`、`START DEMO`、`SUCCESS`、`DISCARD` 和 `RESET TO START` 按钮。

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

采集配置位于 [`configs/collect.yaml`](configs/collect.yaml)，包括仿真设备、手动放置、BEV、连续控制速度、episode 时长和数据质量阈值。前视与下视相机路径读取 `configs/robots/agx.yaml`，相机分辨率、频率、帧同步阈值、地图融合周期和 `LocalVoxelMap` 参数读取 `configs/navigation.yaml`，与主导航保持一致。
