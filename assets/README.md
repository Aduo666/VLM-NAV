# Assets

这个目录只保留说明文件。运行时调试图会自动生成，并被 `.gitignore` 忽略。

当前生成物：

- `opennav_debug.png`: 给人看的运行调试图，固定为左侧局部俯视图、右侧第一人称图，低频刷新。
- `opennav_request.png`: 真正发送给 VLM 的候选点请求图，只在重规划时更新。
- `depth_debug.png`: 最近一次前向深度图的灰度可视化。

场景入口：

- `../usd/hospital.usd`

机器人模型：

- `../src/ros2_car_model_ws/src/1557/urdf/1557_all/1557_all.usd`
