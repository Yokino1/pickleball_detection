# 追踪约束与调参

当前参考配置默认只输出评分最高的一个运动球轨迹。内部仍可维护多个候选，以便真实球短时丢失后从误检目标切回；将 `max_output_tracks` 设为 `0` 可恢复全部有效运动轨迹输出。检测框去重、相机运动补偿、运动确认、静止抑制和物理位移约束共同决定哪些候选能够成为轨迹。

## 处理顺序

1. 估计相邻帧的全局画面平移，用背景特征补偿相机抖动和小幅平移。
2. YOLO 输出置信度不低于 `detector.low_conf` 的候选框。
3. 合并同一个球的重叠或嵌套检测框，只保留最高置信度框。
4. temporal/physics 版本使用连续帧运动过滤器检查局部运动证据；主线跳过该步骤。
5. 高置信度检测可以创建轨迹；低置信度检测只能恢复已有轨迹。
6. 新轨迹必须同时产生原始画面位移和相机补偿后的位移，才会成为可见运动轨迹。
7. 正常关联使用真实帧时间差更新卡尔曼预测，并按 `px/s` 速度、漏检毫秒数、球框尺寸和飞行方向计算代价。
8. 普通飞行中的大幅方向突变会被拒绝。击球、反弹或遮挡恢复只有在至少经历配置的短时漏检后，才能在有限范围内接回原 ID。
9. 漏检时只显示短时预测。普通球最多显示 `max_prediction_ms`，高速球最多显示 `fast_max_prediction_ms`，置信度按实际经过时间衰减，越界预测不输出。
10. 已激活轨迹长时间不动后休眠；再次产生真实运动时使用原 ID 唤醒。
11. `max_output_tracks: 1` 只输出评分最高的主轨迹；设为 `0` 时输出全部有效运动轨迹。

## 当前参考参数

- `detector.low_conf: 0.15`：更低的检测不能参与关联。
- `tracker.high_conf: 0.25`：达到此阈值的检测可以创建轨迹或触发急变恢复。
- `motion_threshold_px: 12`：运动确认的累计位移门槛，以 1280 像素宽画面为基准。
- `min_motion_speed_px_per_second: 16`：允许持续缓慢滚动的球激活，同时抑制检测框漂移。
- `max_stationary_ms: 1500`：确认轨迹连续静止超过该时间后停止输出。
- `max_speed_px_per_second: 2400`：以 1280 像素宽画面为基准的速度上限，其他宽度按比例缩放。
- `impact_recovery_gate_px: 260`：击球、反弹和遮挡恢复的基础半径。
- `impact_recovery_min_missing_ms: 20`：跨普通门控或大幅变向恢复前，至少经历一次短时漏检。
- `impact_recovery_max_missing_ms: 120`：人体接触只允许恢复短时漏检；更早的旧轨迹不得借人体框接到远处候选。
- `max_flight_direction_change_deg: 75`：非恢复状态下的最大飞行转向角。
- `direction_gate_min_speed_px_per_second: 250`：低于该速度时不启用方向门控，避免低速方向噪声。
- `max_prediction_ms: 120`：普通轨迹最长可见预测时间。
- `fast_prediction_speed_px_per_second: 800`、`fast_max_prediction_ms: 60`：高速球只做极短预测，减少沿旧方向飞出画面的假轨迹。
- `max_missing_ms: 350`：轨迹在内部允许等待重新检测的最长时间，超过后释放 ID。
- `prediction_velocity_decay: 0.90`：漏检期间按实际经过时间降低预测速度。
- `max_output_tracks: 1`：默认单目标输出；设为 `0` 可恢复多目标输出。
- `output.trail_length: 10`：画面只保留最近 10 个输出位置，减少历史误检形成的长拖尾。

像素距离和 `px/s` 速度阈值以 `reference_frame_width: 1280` 为基准，普通完整画面按宽度自动缩放。裁切不会改变保留区域内的像素密度，因此不能用裁切后的宽度重新推导门控尺度。双摄入口可通过 `runtime.dual_camera_streams.<side>.frame_scale_override` 为每路画面显式指定尺度；该尺度同时用于追踪物理门控、快速运动候选和跨摄交接速度判断。时间门限使用视频时间戳，与 20、30、60 FPS 等帧率解耦；缺失或异常时间戳使用 `default_fps` 回退。正式部署前仍必须使用最终摄像头、分辨率、视野和帧率重新验收。

## 相机运动

`runtime.camera_motion` 使用背景角点光流和 RANSAC 估计全局平移。估计结果只在内点数量、置信度和最大位移约束通过时应用。JSONL 中的 `diagnostics.camera_motion` 会记录每帧的 `dx`、`dy`、内点数和是否应用。

运动确认同时检查两个坐标参考：

- 相机补偿坐标：抑制随镜头一起移动的固定背景。
- 原始画面坐标：抑制固定在屏幕位置的转播台标和记分图层。

该策略面向固定机位和轻微抖动。如果摄像机会持续主动跟随球，应单独建立该机位的验收集并重新调整规则。

## 连续帧运动过滤

`runtime.temporal_motion` 不改变 YOLO 模型，而是在检测后使用连续两帧灰度图提供轻量运动证据：

1. 将上一帧按相机运动估计平移到当前帧。
2. 对两帧做降采样、模糊、差分、阈值化和膨胀。
3. 在每个检测候选周围统计运动像素比例。
4. 低于 `min_motion_fraction` 的候选不进入追踪器。
5. 第一帧没有历史时全部放行；全画面变化超过 `max_global_motion_fraction` 时失效开放，防止切镜或相机运动估计失败造成整帧漏检。

JSONL 的 `diagnostics.temporal_motion` 记录全局运动比例、接受/拒绝数量和每个候选的局部运动比例。该方法主要抑制灯光、墙面标牌和固定 overlay，不会解决检测模型完全漏掉高速球的问题。

该功能在 `configs/tracking_temporal.yaml` 和 `configs/tracking_physics.yaml` 中开启，
在主线 `configs/tracking.yaml` 中关闭。

## 物理约束实验配置

`configs/tracking_physics.yaml` 在当前时序过滤基础上启用六状态常加速度 KF：

- `motion_model: constant_acceleration`：状态为 `[x, y, vx, vy, ax, ay]`。
- `constant_acceleration_min_observations: 4`：至少四次可靠观测后才允许加速度参与预测。
- `max_acceleration_px_per_second2: 12000`：限制滤波状态中的归一化图像加速度。
- `max_observed_acceleration_px_per_second2: 16000`：拒绝普通飞行中的异常观测加速度。
- `use_nis_gate: true`、`nis_gate_threshold: 13.82`：使用预测协方差和测量残差进行创新门控。
- `continuous_prediction_horizon: true`：预测时长随速度从 120ms 连续缩短至 60ms。

impact recovery 会绕过普通飞行的 NIS、加速度和方向异常，但必须满足高置信度、恢复距离、速度和最短漏检时间要求。恢复成功后旧加速度清零。

这些参数仍是二维像素空间经验值，不代表 `m/s`、`m/s^2` 或真实重力。运行 `scripts/run_physics_tracking.cmd` 会将结果写入 `outputs/experiments/desktop_ab/physics`，用于和默认版本做 A/B 对比。

## 诊断字段

JSONL 的 `diagnostics.tracker` 包含：

- `impact_recoveries`：本帧通过急变或遮挡恢复接回原 ID 的数量。
- `physical_gate_rejections`：因不合理位移被拒绝的关联数量。
- `direction_gate_rejections`：普通飞行中因方向突变被拒绝的关联数量。
- `nis_gate_rejections`：physics 版本中因创新过大被拒绝的关联数量。
- `acceleration_gate_rejections`：physics 版本中因观测加速度过大被拒绝的关联数量。
- `motion_unconfirmed_tracks`：尚未形成真实运动的内部候选数量。
- `stationary_suppressed_tracks`：当前因静止而停止输出的轨迹数量。
- `track_states`：内部轨迹的 ID、预测中心、速度、加速度、漏检帧数和运动确认状态。
- `frame_dt_ms`：本帧用于 Kalman 预测的真实时间间隔。
- `missing_time_ms`：每条内部轨迹距离上次真实检测经过的时间。

## 已知边界

纯速度模型无法在没有任何新检测时知道球已经被球拍击回。因此高速漏检只做很短预测；一旦检测重新出现，再通过急变恢复重置速度方向。要进一步判断击球瞬间，需要增加球拍/人体关键点或专门的事件模型，不能靠无限延长直线预测解决。

## 双摄 handoff 和高速运动候选

`tracking_person_contact.yaml` 的双摄实验增加两层保守辅助：

1. `runtime.dual_camera_handoff` 根据当前侧球是否靠近网口、是否朝网口运动，
   为另一侧开启毫秒级接球预警；接收侧仍先做全图 YOLO，漏检后才做网口 ROI
   二次 YOLO。
2. `runtime.fast_motion` 只在 handoff ROI 或已有高速预测 ROI 内工作。候选必须
   尺寸受限、速度在配置范围内并连续出现至少 `min_streak` 帧，才以
   `source=fast_motion` 进入追踪器。

运动候选不能覆盖同帧 YOLO 检测，也不能在全图无约束地创建球。JSONL 中的
`roi_retry_used`、`fast_motion_proposal_count` 和对应 diagnostics 用于区分
YOLO 找回、运动辅助和追踪预测。

人体接触只解除有限范围内的方向、NIS 和加速度异常，不解除最高速度约束。接触证据要求上一观测点、当前观测点或当前预测点实际进入 eligible player 的扩展框；仅仅因为一条很长的连接线穿过人体框，不构成接触证据。

双摄全局协调器还会对最终输出位置逐帧执行与本地追踪器相同的速度上限。即使本地轨迹 ID 发生变化，只要全局位置跳跃超过按真实时间和流尺度计算的最大位移，该帧就输出 `temporarily_lost`，不会把两个不连续候选连接成同一条轨迹。`global_tracking.jsonl` 的 `coordinator.continuity_gate_rejections` 记录此类拒绝次数。
