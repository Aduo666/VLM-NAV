# 受限空间局部规划器实验

独立入口加载现有 AGX USD 和参数化直门。当前支持真实障碍地图、MPPI/KMPPI、目标直达与共享参考路径两种条件。新增程序和配置位于 `constraint_traverse/`；主导航和人工示范采集入口保持原有用法。

## 启动

在仓库根目录使用已有 Isaac Sim 环境：

```bash
conda activate vlmnav
python -m pip install -r constraint_traverse/benchmark/requirements.txt
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
python -m constraint_traverse.benchmark.run
```

默认门洞总余量为 0.40 m。机器人在门前的固定位置开始，机械臂保持加载时姿态。门宽等于核查后的保守车体宽度加指定总余量，当前保守宽度约为 0.593 m。该宽度包含车轮完整转向包络；日志中的间距也是保守包围轮廓间距。

配置位于 `constraint_traverse/configs/benchmark.yaml`。物理计算采用 60 Hz，规划周期为 0.20 s，预测时域为 4 s；轮组几何和硬件上限读取 `configs/robots/agx.yaml`。MPPI/KMPPI 使用同一运动模型、物理代价、安全余量和执行接口。

## 操作

`Passage Controls` 和 `Passage Map` 是可移动窗口。

- 方向键或 WASD 手动驾驶，组合方向键可以连续转弯。
- `BORN` 暂停物理并选中机器人，使用 Isaac Sim 移动、旋转工具放置，再次点击确认。
- 在地图点击门后空闲位置选择固定世界目标。场景中不会创建目标几何。
- 选择 `MPPI` 或 `KMPPI`；选择 `Goal` 或 `Reference Path`。
- 停稳后点击 `START`，程序记录初始状态并自动通行。
- `STOP` 结束本次试验并制动。
- `RESET TRIAL` 结束正在进行的试验，恢复已保存的底盘、全部关节、执行目标、控制器、目标、地图版本及种子。下一次 START 创建新的规划器，清空历史控制序列。
- `RESET MAP` 重建真实地图显示，不改变试验起点。真实地图由固定墙体参数生成，重建后的障碍布局保持一致。

蓝色曲线表示实际运动，绿色曲线表示最终预测，浅灰绿色曲线表示候选。红色目标圆点仅出现在地图中。参考路径条件额外显示紫色路线。

当前交互试验要求静止起点。运动模型包含当前轮组转角、轮速、速度变化及转向响应。恢复采用底盘和全部关节状态重建 PhysX 运动学缓存，并核对各刚体状态。接触求解器内部缓存通过 World.reset 重建。

## 单个条件重复试验

```bash
python -m constraint_traverse.benchmark.run \
  --headless --method MPPI --goal 1.6 0 \
  --door-clearance 0.20 --repeat 2 --seed 0
```

可用参数包括 `--start-y`、`--start-yaw-deg`、`--navigation-condition goal/reference` 和 `--timeout`。`--repeat` 在同一状态、同一种子下重复该条件，用于检验重置。正式随机种子评测需要分别指定种子；这里的重复次数不代表独立随机试验数量。

## 成功与失败

成功需要完整保守车体越过门洞出口平面、全程无门框或墙体接触、进入目标 0.30 m 范围并停稳至少 0.30 s。机器人中心越过门洞不会单独触发成功。轮胎与地面的正常接触不计为障碍碰撞。

真实物理碰撞使用 PhysX 接触报告。优化代价使用核查了全部 23 个 CollisionAPI 部件的保守轮廓和障碍余量。程序执行优化器输出的首个控制指令，并根据底盘速度、加减速和转向能力生成轮组指令。发生物理障碍碰撞时，记录 `COLLISION` 并结束试验。优化输出含无效数值时，记录 `INVALID_OUTPUT`。

结果还包含 `COLLISION`、`TIMEOUT`、`STOP_BY_USER`、`RESET_BY_USER` 和 `APPLICATION_CLOSED`。单个失败结果保留逐周期规划原因，不能单凭停止就认定机器人或目标不可通行。

## 数据和回放

每次试验保存在 `constraint_traverse/data/benchmark/<时间_编号>/`：

- `meta.json`：配置、起始状态、刚体与控制器状态、几何核查、代码与依赖版本、终止原因和指标。
- `state.npz`：仿真时间、实际位姿与运动状态、请求控制、限速控制和保守轮廓间距。
- `plans/*.npz`：全部候选、候选控制与代价、优化控制序列和最终预测。
- `events.json`：规划耗时、代价模型估计间距及障碍接触事件。

规划耗时包含优化、预测轨迹计算和数据转换，记录均值、P95 和超过规划周期的次数。当前仿真采用同步规划，推理期间不推进物理时间；这些结果不能直接声称满足实际机器人实时控制要求。保存候选数据的耗时没有包含在 `planning_sec` 中。

```bash
python -m constraint_traverse.benchmark.replay \
  constraint_traverse/data/benchmark/<时间_编号>
```

回放直接生成该试验目录下的 `replay.mp4`，不需要 OpenCV GUI。

## 验收

```bash
python -m unittest constraint_traverse.benchmark.test_benchmark -v
python -m constraint_traverse.benchmark.verify
```

第二条命令启动真实 Isaac Sim，检查状态恢复后的重复运动、单物理周期预测误差、面板 START 接口，并通过已知障碍接触验证碰撞回调。验证报告位于 `constraint_traverse/data/benchmark_validation/`。

已完成的功能验收与窄门通行数据见 [VALIDATION.md](VALIDATION.md)。

iCEM、RGB-D 地图、动态起点和参数组合批量评测尚未接入。RGB-D 条件将沿用主导航的前视与底盘下视建图设置，UNKNOWN 继续独立记录和禁行。
