# Isaac Sim Open-Nav VLN

这是一个基于 Isaac Sim 5.1 的 Open-Nav 风格 VLN 工程。当前版本只保留一条主链路：

```text
Isaac Sim observation -> Open-Nav candidate/action representation -> VLM reasoner -> Ackermann joint controller
```

旧的手动控制测试、startup scan、contact sheet、BEV/语义地图实验链路已经从主工程删除。

## 目录结构

```text
.
├── assets/
│   └── README.md
├── configs/
│   └── default.yaml
├── scripts/
│   ├── setup_grounded_sam_local.sh
│   ├── run_nav.sh
│   └── test_llm_call.py
├── src/
│   ├── ros2_car_model_ws/
│   └── vln_nav/
│       ├── app/
│       │   └── nav.py
│       ├── control/
│       │   └── controller.py
│       ├── navigation/
│       │   ├── candidate_waypoint_predictor.py
│       │   ├── depth_safety.py
│       │   ├── instruction_manager.py
│       │   ├── grounding_dino_detector.py
│       │   ├── landmark_map.py
│       │   ├── landmark_tracker.py
│       │   ├── lidar_sensor.py
│       │   ├── observation_adapter.py
│       │   ├── open_nav_client.py
│       │   ├── progress_manager.py
│       │   ├── spatial_transform.py
│       │   ├── target_grounding.py
│       │   ├── target_memory.py
│       │   ├── vln_policy.py
│       │   └── vln_reasoner.py
│       └── runtime/
│           ├── config.py
│           └── isaac_runtime_utils.py
└── usd/
    └── hospital.usd
├── third_party/
│   └── GroundingDINO/
└── weights/
    └── grounded_sam/
```

## 运行环境

使用已有 conda 环境：

```bash
conda activate env_isaaclab
```

如果需要安装 Isaac Sim 5.1，请在该环境内按 NVIDIA 官方方式安装。当前运行脚本默认使用 `env_isaaclab`。

## 本地 GroundingDINO 部署

当前工程默认从工程内部读取 GroundingDINO：

```text
third_party/GroundingDINO/
weights/grounded_sam/GroundingDINO_SwinT_OGC.py
weights/grounded_sam/groundingdino_swint_ogc.pth
```

部署代码和依赖：

```bash
bash scripts/setup_grounded_sam_local.sh
```

如果你已经有 CA-Nav 的权重，可以直接软链接，不需要复制 3GB 文件：

```bash
ln -sf /home/Lab/qiangsibin/CA-Nav-code/data/grounded_sam/GroundingDINO_SwinT_OGC.py weights/grounded_sam/GroundingDINO_SwinT_OGC.py
ln -sf /home/Lab/qiangsibin/CA-Nav-code/data/grounded_sam/groundingdino_swint_ogc.pth weights/grounded_sam/groundingdino_swint_ogc.pth
```

如果后续要启用 SAM mask，再放：

```text
weights/grounded_sam/sam_vit_h_4b8939.pth
weights/grounded_sam/repvit_sam.pt
```

代码里已经预留 `sam_vit_h_4b8939.pth` 接口，但当前默认配置 `target_grounding.sam_enabled: false`，实际运行不会使用 SAM。如果你已有 CA-Nav 权重且想打开 SAM，再软链接：

```bash
ln -sf /home/Lab/qiangsibin/CA-Nav-code/data/grounded_sam/sam_vit_h_4b8939.pth weights/grounded_sam/sam_vit_h_4b8939.pth
```

## 配置

主配置文件是 `configs/default.yaml`。

常改字段：

```yaml
llm:
  api_key: ""
  api_base_url: "https://api.siliconflow.cn/v1"
  model_name: "Qwen/Qwen3-VL-30B-A3B-Instruct"

opennav:
  instruction: "<your navigation instruction>"
  landmark_arrival_distance_m: 1.0
  landmark_arrival_confirmations: 2
  progress_stall_distance_m: 0.08
  recovery_enabled: true
  recovery_trigger_stalled_steps: 3
  debug_image_update_period_sec: 1.0

map_candidates:
  enabled: true
  max_candidates: 6
  semantic_enabled: true
  frontier_enabled: true
  escape_enabled: true
  min_clearance_m: 0.34

target_grounding:
  enabled: true
  backend: "auto"   # auto | grounding_dino | qwen_vl
  call_period_sec: 1.5
  reuse_last_observation_sec: 2.5
  coordinate_mode: "auto"
  allow_normalized_bbox: true
  depth_region: "bbox_depth_mask"
  depth_statistic: "median"
  min_depth_valid_ratio: 0.12
  grounding_dino_repo_path: "third_party/GroundingDINO"
  grounding_dino_config_path: "weights/grounded_sam/GroundingDINO_SwinT_OGC.py"
  grounding_dino_checkpoint_path: "weights/grounded_sam/groundingdino_swint_ogc.pth"
  grounding_dino_device: "cuda:1"
  sam_checkpoint_path: "weights/grounded_sam/sam_vit_h_4b8939.pth"
  sam_model_type: "vit_h"
  sam_device: "cuda:1"

landmark_tracker:
  arrival_visible_depth_m: 1.00
  arrival_anchor_radius_m: 1.00
  min_visible_count: 2
  arrival_confirmations: 2
  landmark_map_enabled: true
  landmark_map_anchor_ttl_sec: 120.0
  reacquire_guard_enabled: true
  reacquire_guard_recent_visible_sec: 20.0

camera:
  front_camera_prim_path: "link6/Camera"
  observer_camera_prim_path: "dummy_base_link/observecamera"
  obstacle_stop_distance_m: 0.65
  near_field_memory_enabled: true
  near_field_memory_sec: 2.0

control:
  wheel_drive_velocity: 14.0
  turn_drive_velocity: 6.0
  use_discrete_action_primitives: true
  stuck_detection_enabled: true
  stuck_window_sec: 2.0
  stuck_recovery_backward_sec: 0.75
  stuck_recovery_turn_sec: 0.35
  action_forward_duration_sec: 0.65
  action_turn_duration_sec: 0.30
  action_turn_angle_deg: 15.0

lidar:
  enabled: true
  prim_path: "dummy_base_link/base_link/Lidar_link"
  child_name: "lidar_sensor"
  horizontal_fov_deg: 360.0
  min_range_m: 0.20
  max_range_m: 6.0
  yaw_offset_deg: 0.0

scene:
  usd_path: "usd/hospital.usd"

robot:
  usd_path: "src/ros2_car_model_ws/src/1557/urdf/1557_all/1557_all.usd"
```

建议用环境变量传 API key，不要写进配置文件：

```bash
export SILICONFLOW_API_KEY="your_key"
```

如果没有设置 API key，程序会在启动 Isaac Sim 前直接退出。当前版本没有本地导航 fallback，不能在无 VLM 的情况下移动。

## 运行导航

```bash
bash scripts/run_nav.sh
```

指定配置：

```bash
CONFIG_PATH=/abs/path/to/default.yaml bash scripts/run_nav.sh
```

无界面运行：

```bash
bash scripts/run_nav.sh --headless
```

非 headless 模式需要当前终端有可用显示环境，并且 Isaac Sim 的 renderer GPU 必须是连接显示器的 GPU。配置里 `sim.active_gpu` 默认是 `auto`，程序会用 `nvidia-smi display_active` 自动选择显示 GPU；也可以手动覆盖：

```bash
ACTIVE_GPU=1 bash scripts/run_nav.sh
```

如果日志出现 `Failed to find a graphics and/or presenting queue`、`createSwapchain failed` 或 `backbuffers are not initialized`，通常说明 `ACTIVE_GPU` 指到了没有显示输出的 GPU，或者当前终端没有 `DISPLAY/WAYLAND_DISPLAY`。这种情况下从桌面/NoMachine/VNC 终端启动，或改用 `--headless`。

## 调试输出

运行时会生成：

```text
logs/nav_latest.log
debug/vln_debug.png
debug/videos/vln_debug_YYYYMMDD_HHMMSS.mp4
debug/llm_request.png
logs/last_candidate_choice_request.json
```

`debug/vln_debug.png` 是给人看的组合调试图：左侧是第一人称 target/动作覆盖图，会直接画出 grounding/tracker bbox、可选 SAM mask、depth 或 lost 原因；右侧是 north-up occupancy/navigation map，包含轨迹、landmark、当前目标和 occupancy。

`debug/videos/vln_debug_YYYYMMDD_HHMMSS.mp4` 会把运行过程中的组合调试图低频记录成视频，每次运行自动加日期后缀，格式为 VSCode 可直接打开的 mp4。

`debug/llm_request.png` 是真正发送给 VLM 的候选点请求图，只在重规划时覆盖更新。单独的 `local_map/global_map/lidar_raw_scan/depth_debug` 默认不再周期性落盘，避免 IO 和观察干扰。

## 目标识别与到达判定

当前目标感知不是每帧重新跑 GroundingDINO。默认流程是：GroundingDINO 给出候选 bbox，bbox/depth 一致区域用于估计距离，随后 VLM 会对标记区域做语义复核；通过后初始化 OpenCV tracker。原子动作执行期间默认暂停新的 GroundingDINO/Qwen grounding，只用 tracker 更新 bbox，并继续通过 bbox + depth 估计 bearing/depth/world anchor。只有 `can_confirm=1` 且连续观测稳定的目标才会进入 confirmed anchor；低置信或 VLM 复核失败的结果只作为 weak hint，不允许触发 landmark 到达。SAM 是可选增强，当前默认关闭。

默认关键阈值在 `configs/default.yaml`：

```yaml
target_grounding:
  sam_enabled: false
  semantic_verification_enabled: true
  semantic_verification_min_confidence: 0.70
  semantic_verification_auto_accept_confidence: 1.01
  pause_during_active_primitive: true
  tracker_enabled: true
  tracker_type: "CSRT"
landmark_tracker:
  confirm_confidence_threshold: 0.65
  arrival_recent_visible_sec: 3.0
```

候选请求图 `debug/llm_request.png` 的第一人称画面只保留小号编号点，避免遮挡目标和环境；候选动作、距离、clearance、free ratio 仍在 prompt 文本里。

`scripts/test_llm_call.py` 可以复用最近一次导航真正发给模型的 prompt：

```bash
python scripts/test_llm_call.py
```

也可以手动测试：

```bash
python scripts/test_llm_call.py --mode manual --image debug/llm_request.png
```

## 当前主链路

1. `nav.py` 启动 Isaac Sim，加载 `usd/hospital.usd` 和机器人 USD。
2. `observation_adapter.py` 整理 RGB、Depth、机器人位姿和上一帧执行反馈。
3. `lidar_sensor.py` 从 `dummy_base_link/base_link/Lidar_link` 读取或创建 PhysX LiDAR，并用 LiDAR scan 更新 occupancy grid；只有关闭 LiDAR 时才退回前视深度建图。
4. `candidate_waypoint_predictor.py` 从深度图生成固定槽位候选动作。
5. `map_candidate_generator.py` 基于 occupancy grid、当前 landmark anchor、安全上下文和进展状态补充 `semantic_approach/frontier/escape_turn` 候选，并统一成 `WaypointCandidate` 供 VLM 选择。
6. `target_grounding.py` 只在搜索/重规划/跟踪丢失时低频获取当前 landmark 的 bbox；`backend: auto` 时优先尝试本地 GroundingDINO，缺少 checkpoint 或不可用时自动回退到 Qwen VLM bbox。
7. `target_tracker.py` 在 confirmed grounding 后接管 bbox 连续跟踪。原子动作执行期间不再新发 GroundingDINO/Qwen grounding，而是 tracker bbox + depth 继续更新目标 bearing/depth/world anchor。
8. `landmark_tracker.py` 只在连续一致观测后确认 landmark anchor，并判断当前 landmark 是否到达。
9. `landmark_map.py` 保存 confirmed landmark anchor；目标暂时离开视野后，主链仍能基于已确认世界坐标判断距离和给 VLM 提供记忆。
10. `vln_reasoner.py` 将指令、历史、候选点图、LandmarkMap 和安全上下文发给 VLM。
11. `vln_policy.py` 将 VLM 选中的候选槽位映射成固定离散动作原语，并在 tracker/map 判定到达后切换到下一个 landmark；如果目标刚可见后因转向丢失，`ReacquireGuard` 会禁止继续同向转。
12. `controller.py` 执行 `FORWARD/BACKWARD/TURN_LEFT/TURN_RIGHT/STOP`，每个原语按固定时间或固定转角结束，然后停车回正并重规划。

当前版本没有本地启发式强行覆盖 VLM 选择；安全模块只在近距离碰撞风险时 STOP，并把原因传回下一轮 VLM 决策。

当前本地语义感知主链是 GroundingDINO bbox + bbox/depth 估计 + VLM 语义复核。GroundingDINO 缺失时会回退到 Qwen bbox；SAM 只有在 `target_grounding.sam_enabled: true` 时才参与。
