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

后续如果接 SAM mask，再放：

```text
weights/grounded_sam/sam_vit_h_4b8939.pth
weights/grounded_sam/repvit_sam.pt
```

当前工程已经接入 `sam_vit_h_4b8939.pth`。如果你已有 CA-Nav 权重，建议直接软链接：

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

## 调试输出

运行时会生成：

```text
logs/nav_latest.log
assets/opennav_debug.png
assets/opennav_debug.mp4
assets/opennav_request.png
assets/occupancy_grid.png
assets/depth_debug.png
logs/last_candidate_choice_request.json
```

`assets/opennav_debug.png` 是给人看的组合调试图：左侧是第一人称 target/动作覆盖图，会直接画出 grounding bbox、SAM mask、depth 或 lost 原因；右侧是 navigation map，包含轨迹、landmark、当前目标和 occupancy。`assets/occupancy_grid.png` 是单独的局部占用图，红色为深度障碍，橙色为安全膨胀边界。

`assets/opennav_debug.mp4` 会把运行过程中的组合调试图低频记录成视频，便于事后看完整决策过程。

`assets/opennav_request.png` 是真正发送给 VLM 的候选点请求图，只在重规划时更新。

`scripts/test_llm_call.py` 可以复用最近一次导航真正发给模型的 prompt：

```bash
python scripts/test_llm_call.py
```

也可以手动测试：

```bash
python scripts/test_llm_call.py --mode manual --image assets/opennav_request.png
```

## 当前主链路

1. `nav.py` 启动 Isaac Sim，加载 `usd/hospital.usd` 和机器人 USD。
2. `observation_adapter.py` 整理 RGB、Depth、机器人位姿和上一帧执行反馈。
3. `candidate_waypoint_predictor.py` 从深度图生成固定槽位候选动作。
4. `target_grounding.py` 低频获取当前 landmark 的 bbox；`backend: auto` 时优先尝试本地 GroundingDINO，缺少 checkpoint 或不可用时自动回退到 Qwen VLM bbox。bbox 会映射回原始 RGB/depth 坐标，然后优先用 SAM 在 bbox 内生成目标 mask，并用 mask 内深度估计目标距离和世界坐标候选 anchor；SAM 不可用时才退回 bbox 内深度一致区域。
5. `landmark_tracker.py` 只在连续一致观测后确认 landmark anchor，并判断当前 landmark 是否到达。
6. `landmark_map.py` 保存 confirmed landmark anchor；目标暂时离开视野后，主链仍能基于已确认世界坐标判断距离和给 VLM 提供记忆。
7. `vln_reasoner.py` 将指令、历史、候选点图、LandmarkMap 和安全上下文发给 VLM。
8. `vln_policy.py` 将 VLM 选中的候选槽位映射成固定离散动作原语，并在 tracker/map 判定到达后切换到下一个 landmark；如果目标刚可见后因转向丢失，`ReacquireGuard` 会禁止继续同向转。
9. `controller.py` 执行 `FORWARD/BACKWARD/TURN_LEFT/TURN_RIGHT/STOP`，每个原语按固定时间或固定转角结束，然后停车回正并重规划。

当前版本没有本地启发式强行覆盖 VLM 选择；安全模块只在近距离碰撞风险时 STOP，并把原因传回下一轮 VLM 决策。

当前本地语义感知主链是 GroundingDINO bbox + SAM mask。GroundingDINO 缺失时会回退到 Qwen bbox；SAM 缺失时会明确输出 `[SAM] unavailable`，并退回 bbox 深度估计。
