# CAPTAINav

当前分支在 Isaac Sim 中运行 AGX 视觉语言导航。远程 InternVLA-N1 System 2 根据指令与前视历史输出 pixel-goal 或离散动作；pixel-goal 转换成 metric goal 后交给现有粗粒度或细粒度导航。用户仍可在 RGB-D 局部地图上手动选择目标。

## 导航流程

```text
语言指令 + 前视历史
          |
          v
远程 InternVLA-N1 S2
          |
          +---- pixel-goal ----> RGB-D 几何转换 ----> metric goal
          |
          +---- FORWARD / LEFT / RIGHT ----> 局部位置或航向目标
          |
          +---- LOOKDOWN ----> 下视相机续问
          |
          +---- STOP ----> 静止复核
                                  |
            COARSE / FINE / MPPI / KMPPI
                                  |
                           AGX 四轮连续控制
```

底层支持 `COARSE`（NavDP）、`FINE`（Hybrid A*）、`MPPI` 和 `KMPPI`。默认模式由 `configs/navigation.yaml` 的 `execution.default_mode` 设置，当前为 `COARSE`。Isaac Sim 的 `Navigation Controls` 窗口提供四种模式的下拉选择；切换时清除旧轨迹和规划历史，保留当前局部目标。

## 环境与运行

- 在 conda 环境中安装 Isaac Sim 5.1 和 [requirements.txt](requirements.txt) 中的依赖。
- S2 地址和 pixel-goal 转换参数位于 [导航配置](configs/navigation.yaml) 的 `internnav_s2` 段。
- 场景文件、初始位姿和导航指令位于 [任务清单](configs/episodes.json)。
- `COARSE` 模式需要启动 NavDP 服务，其地址由 [导航配置](configs/navigation.yaml) 的 `server.base_url` 指定。
- AGX 的 USD、相机和轮组参数位于 [AGX 配置](configs/robots/agx.yaml)。

```bash
conda activate vlmnav
bash scripts/run_nav.sh
```

可以通过启动参数直接选择底层：

```bash
bash scripts/run_nav.sh --local-planner MPPI
bash scripts/run_nav.sh --local-planner KMPPI
```

启动参数只影响本次运行。运行过程中也可以在 `Navigation Controls` 下拉框切换回 `COARSE` 或 `FINE`。

启动后，终端会列出 `episodes.json` 中的任务。输入任务编号后，程序才会启动 Isaac Sim，并使用该任务的场景、初始位姿和指令。输入无效编号时会要求重新输入，不会自动选择任务。

完整导航使用远程 S2 服务。`COARSE` 另外需要 NavDP 服务；`FINE`、`MPPI`、`KMPPI` 在本机进行底层规划。主程序获得有效前视帧后自动调用 `/reset` 创建会话，随后通过 `/infer` 执行所选任务的指令。S2 输出 `LOOKDOWN` 时，程序停车并将固定下视相机的新图像发回同一会话。

MPPI/KMPPI 依赖通过 `python -m pip install -r constraint_traverse/benchmark/requirements.txt` 安装，参数由 `sampling_planner.config_path` 指向的配置读取，当前为 `constraint_traverse/configs/benchmark.yaml` 中的运动模型、代价、采样和障碍余量设置。输入包括前视和底盘下视 RGB-D 直接建立的二维规划地图、S2 局部目标及实测轮组状态。每个周期执行优化器输出的首个控制指令，底盘控制器限制速度、加减速与转向。障碍和 UNKNOWN 参与优化代价。日志记录 `[SamplingPlan]`、优化耗时及估计间距。

主导航使用二维占用栅格：深度像素根据相机内外参转换到世界坐标，按地面相对高度识别障碍和地面；平面射线更新空闲栅格，障碍终点更新占用栅格。地图随机器人平移并保留重叠区域的证据，Goal Map 与规划器使用同一份二维地图。启动日志显示 `[Mapping] type=RGBD_2D`。地图尺寸、栅格分辨率、采样间隔及融合周期读取 `mapping.map`，障碍高度范围读取 `fine_planner.obstacle_min_height_m/obstacle_max_height_m`。

前视相机水平视角由 `configs/navigation.yaml` 的 `camera.horizontal_fov_deg` 设置，当前为 `80°`。S2 返回连续同向转向时，程序最多执行 `30°`，停稳后用新的前视画面重新请求决策。

Isaac Sim 中包含 `S2 Decision`、`Goal Map` 和 `Navigation Controls` 窗口。任务开始、局部目标切换和最终停止均由主导航自动处理。点击 Goal Map 中的已知空闲位置仍可执行手动点目标导航，方向键可以手动控制 AGX。

新增任务时，在 [任务清单](configs/episodes.json) 中增加唯一 `id`、名称、USD 路径、初始 XY、初始朝向和指令。XY 和朝向使用场景世界坐标，初始高度由 AGX 配置决定。[导航配置](configs/navigation.yaml) 中的 `scene` 段只保存场景加载公共参数；`add_indoor_lighting` 控制运行时室内灯光，原始场景 USD 不会被修改。

## 代码与输出

| 文件 | 用途 |
| --- | --- |
| [navigation.py](src/vln_nav/app/navigation.py) | 场景加载、目标选择与四种底层模式 |
| [constraint_traverse/navigation.py](constraint_traverse/navigation.py) | MPPI/KMPPI 规划地图适配与轮组执行 |
| [navigation.yaml](configs/navigation.yaml) | 场景、相机、地图、规划和跟踪参数 |
| [episodes.json](configs/episodes.json) | 可选择任务的场景、初始位姿和指令 |
| [episode.py](src/vln_nav/runtime/episode.py) | 任务清单校验与终端选择 |
| [navdp_client.py](src/vln_nav/navigation/navdp_client.py) | NavDP 请求与轨迹校验 |
| [internnav_s2_client.py](src/vln_nav/navigation/internnav_s2_client.py) | 远程 S2 请求与输出校验 |
| [s2_goal_adapter.py](src/vln_nav/navigation/s2_goal_adapter.py) | pixel-goal 深度反投影与 metric goal 转换 |
| [rgbd_map.py](constraint_traverse/rgbd_map.py) | RGB-D 直接二维建图 |
| [fine_local_planner.py](src/vln_nav/navigation/fine_local_planner.py) | 细粒度局部轨迹搜索 |
| [trajectory.py](src/vln_nav/control/trajectory.py) | AGX 连续轨迹跟踪与轮组速度转换 |
| [scene_lighting.py](src/vln_nav/runtime/scene_lighting.py) | 仿真 Stage 中的室内灯光 |

程序覆盖写入 `logs/nav_latest.log`。S2 pixel-goal、NavDP 候选轨迹和细粒度几何图片覆盖写入 `debug/local/`。

Fine 模式对完整路径和增量路径统一检查车体扫掠区域：新增覆盖的二维未知栅格或已知障碍会阻止执行。增量路径还检查停车距离与近期实际位姿，避免反复接收没有实际进展的短路径。日志中的 `FULL_PATH`、`SAFE_PARTIAL_PATH`、`NO_SAFE_PROGRESS` 和 `SEARCH_INCOMPLETE` 表示规划结果；首次失败时，`debug/local/fine_failure_*.json` 记录各动作与车体部件的未知覆盖，配套 `.npz` 保存相同地图版本的规划栅格及三维观测，`.png` 显示当前车体、膨胀余量、直行扫掠及转向未知区域。固定地图测试命令为 `PYTHONPATH=src python3 -m unittest discover -s scripts -p 'test_fine_incremental.py'`。
