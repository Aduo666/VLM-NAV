`contracts.py` 定义任务、环境信念、通行规划、机械臂观察和评价五个模块的公共接口。数据对象只携带数值、标识符和不可写数组，规划接口使用信念快照和机器人状态。

信息来源：

- `RealObservation` 包含真实相机帧及同一采集时刻的机器人状态。
- `PredictedObservation` 包含环境假设标识符及模拟相机帧。
- `GroundTruthGeometry` 表示供仿真和离线评价使用的完整几何。
- `ObservedGeometry` 必须记录对应真实观测标识符。
- `Belief.update()` 只接受真实观测；融合器的输出必须为 `ObservedGeometry`。

当前 `Belief` 实现真实观测历史、版本及来源检查。三维融合通过 `RealGeometryIntegrator` 接口接入；未提供融合器时，`observed_geometry` 保持 `None`。几何生成、未来观测预测和机械臂实际安全检查尚未实现。

`ArmObservationPlan` 检查伸出和收回轨迹的数据连续性、返回收纳姿态及安全检查结果类型。`safety_passed` 是实际安全检查模块填写的结果；该数据类型自身不证明运动安全。

`evaluation.py` 使用 NumPy 和公共数据类型，可以通过普通 Python 单元测试运行。

执行损失采用：

\[
L=c_{collision}\mathbf1[collision]+c_{incomplete}\mathbf1[incomplete]
 +c_{time}T+c_{clearance}\left[\max\left(0,\frac{d_{target}-d_{min}}{d_{target}}\right)\right]^2.
\]

所有权重和间距目标由调用者明确提供。`COMPLETED` 表示到达目标并完成通行，碰撞独立计数；无碰撞成功需要同时满足 `completed` 和 `not collided`。停止、超时、规划失败和碰撞终止均计入未完成损失。

`traversal_duration_sec` 记录通行阶段时间，观察成本通过 `observation_cost` 独立输入，记录端需要保证时间成本没有重复计价。没有执行运动的失败结果可使用 `minimum_clearance_m=None`；完成或执行过的通行过程在启用间距损失时必须提供最小间距。没有障碍时可使用正无穷间距。

风险定义为：

\[
\rho(Z)=\mathbb E[Z]+\beta\operatorname{CVaR}_{\alpha}(Z),\qquad 0\le\alpha<1.
\]

`alpha` 使用置信水平定义：计算最高损失侧 `1-alpha` 的概率质量。离散分位点处使用部分样本权重，不将整个分位点样本一并加入尾部。`alpha=0` 时 CVaR 等于期望损失。

观察价值定义为 `risk_before - risk_after - observation_cost`。`LossSamples` 记录每条损失对应的环境与执行误差标识符；观测分支可以扩展样本数量，但每个 `(environment_id, execution_noise_id)` 的总权重必须与观察前一致。环境假设的后验权重不能替代这项事前执行评价权重。

运行测试：

```bash
python -m unittest discover -s constraint_traverse/active_passage/tests -v
```

本轮实现公共接口、来源隔离和风险评价。旧功能迁移状态记录在 [MIGRATION.md](MIGRATION.md)。机械臂伸出及收纳、成对场景、物理状态恢复和执行验收仍待实现。
