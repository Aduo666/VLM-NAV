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
               +------------------+------------------+
               |                                     |
          COARSE: NavDP                         FINE: Hybrid A*
          滚动生成轨迹                         使用局部地图规划轨迹
               |                                     |
               +------------------+------------------+
                                  |
                           AGX 连续轨迹跟踪
```

当前固定使用 `COARSE`，通过独立运行的 NavDP point-goal 服务生成轨迹。细粒度规划代码仍保留，界面暂时不显示模式选择按钮和 Fine Geometry 窗口。

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

启动后，终端会列出 `episodes.json` 中的任务。输入任务编号后，程序才会启动 Isaac Sim，并使用该任务的场景、初始位姿和指令。输入无效编号时会要求重新输入，不会自动选择任务。

完整导航需要先启动远程 S2 服务和 NavDP 服务，再运行主程序。主程序获得有效前视帧后自动调用 `/reset` 创建会话，随后通过 `/infer` 执行所选任务的指令。S2 输出 `LOOKDOWN` 时，程序停车并将固定下视相机的新图像发回同一会话。

前视相机水平视角由 `configs/navigation.yaml` 的 `camera.horizontal_fov_deg` 设置，当前为 `80°`。S2 返回连续同向转向时，程序最多执行 `30°`，停稳后用新的前视画面重新请求决策。

Isaac Sim 中只保留 `S2 Decision` 和 `Goal Map` 两个导航窗口。任务开始、局部目标切换和最终停止均由主导航自动处理。点击 Goal Map 中的已知空闲位置仍可执行手动点目标导航，方向键可以手动控制 AGX。

新增任务时，在 [任务清单](configs/episodes.json) 中增加唯一 `id`、名称、USD 路径、初始 XY、初始朝向和指令。XY 和朝向使用场景世界坐标，初始高度由 AGX 配置决定。[导航配置](configs/navigation.yaml) 中的 `scene` 段只保存场景加载公共参数；`add_indoor_lighting` 控制运行时室内灯光，原始场景 USD 不会被修改。

## 代码与输出

| 文件 | 用途 |
| --- | --- |
| [navigation.py](src/vln_nav/app/navigation.py) | 场景加载、目标选择与两种导航模式 |
| [navigation.yaml](configs/navigation.yaml) | 场景、相机、地图、规划和跟踪参数 |
| [episodes.json](configs/episodes.json) | 可选择任务的场景、初始位姿和指令 |
| [episode.py](src/vln_nav/runtime/episode.py) | 任务清单校验与终端选择 |
| [navdp_client.py](src/vln_nav/navigation/navdp_client.py) | NavDP 请求与轨迹校验 |
| [internnav_s2_client.py](src/vln_nav/navigation/internnav_s2_client.py) | 远程 S2 请求与输出校验 |
| [s2_goal_adapter.py](src/vln_nav/navigation/s2_goal_adapter.py) | pixel-goal 深度反投影与 metric goal 转换 |
| [local_geometry.py](src/vln_nav/active_perception/local_geometry.py) | RGB-D 局部体素地图 |
| [fine_local_planner.py](src/vln_nav/navigation/fine_local_planner.py) | 细粒度局部轨迹搜索 |
| [trajectory.py](src/vln_nav/control/trajectory.py) | AGX 连续轨迹跟踪与轮组速度转换 |
| [scene_lighting.py](src/vln_nav/runtime/scene_lighting.py) | 仿真 Stage 中的室内灯光 |

程序覆盖写入 `logs/nav_latest.log`。S2 pixel-goal、NavDP 候选轨迹和细粒度几何图片覆盖写入 `debug/local/`。

Fine 模式对完整路径和增量路径统一检查车体扫掠区域：新增覆盖的二维未知栅格或已知障碍会阻止执行。增量路径还检查停车距离与近期实际位姿，避免反复接收没有实际进展的短路径。日志中的 `FULL_PATH`、`SAFE_PARTIAL_PATH`、`NO_SAFE_PROGRESS` 和 `SEARCH_INCOMPLETE` 表示规划结果；首次失败时，`debug/local/fine_failure_*.json` 记录各动作与车体部件的未知覆盖，配套 `.npz` 保存相同地图版本的规划栅格及三维观测，`.png` 显示当前车体、膨胀余量、直行扫掠及转向未知区域。固定地图测试命令为 `PYTHONPATH=src python3 -m unittest discover -s scripts -p 'test_fine_incremental.py'`。
