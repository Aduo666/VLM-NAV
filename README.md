# CAPTAINav

当前分支验证 Isaac Sim 中的 AGX 点目标导航。用户在 RGB-D 局部地图上点击已观测的空闲位置，机器人按选定模式滚动规划并跟踪轨迹。语言指令导航尚未接入当前主入口。

## 导航流程

```text
前视 RGB-D + 近场 RGB-D
          |
          v
     滚动局部体素地图 ----> 可点击的目标地图
                                  |
                            世界坐标目标点
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

`COARSE` 使用独立运行的 NavDP point-goal 服务；`FINE` 使用局部 RGB-D 几何地图、AGX 轮组运动约束和碰撞检查。模式由用户手动选择，当前只支持 AGX。

## 环境与运行

- 在 conda 环境中安装 Isaac Sim 5.1 和 [requirements.txt](requirements.txt) 中的依赖。
- `COARSE` 模式需要启动 NavDP 服务，其地址由 [导航配置](configs/navigation.yaml) 的 `server.base_url` 指定。
- AGX 的 USD、相机和轮组参数位于 [AGX 配置](configs/robots/agx.yaml)。

```bash
conda activate vlmnav
bash scripts/run_nav.sh
```

在 Isaac Sim 界面选择 `COARSE` 或 `FINE`，再点击目标地图中的已知空闲位置。`STOP` 停止当前导航；方向键可手动控制 AGX。

切换场景时，在 [导航配置](configs/navigation.yaml) 的 `scene` 段一起修改 `usd_path`、`start_xy_m` 与 `start_yaw_deg`。XY 和朝向使用场景世界坐标，初始高度由 AGX 配置决定。`add_indoor_lighting` 控制运行时室内灯光；原始场景 USD 不会被修改。

## 代码与输出

| 文件 | 用途 |
| --- | --- |
| [navigation.py](src/vln_nav/app/navigation.py) | 场景加载、目标选择与两种导航模式 |
| [navigation.yaml](configs/navigation.yaml) | 场景、相机、地图、规划和跟踪参数 |
| [navdp_client.py](src/vln_nav/navigation/navdp_client.py) | NavDP 请求与轨迹校验 |
| [local_geometry.py](src/vln_nav/active_perception/local_geometry.py) | RGB-D 局部体素地图 |
| [fine_local_planner.py](src/vln_nav/navigation/fine_local_planner.py) | 细粒度局部轨迹搜索 |
| [trajectory.py](src/vln_nav/control/trajectory.py) | AGX 连续轨迹跟踪与轮组速度转换 |
| [scene_lighting.py](src/vln_nav/runtime/scene_lighting.py) | 仿真 Stage 中的室内灯光 |

程序覆盖写入 `logs/nav_latest.log`。候选轨迹和细粒度几何图片覆盖写入 `debug/local/`，路径见导航配置的 `debug` 段。
