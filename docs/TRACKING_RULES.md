# 追踪约束与调参

本文参数以 `configs/tracking.yaml` revision 9 为唯一依据。当前正式配置默认只输出
评分最高的一个运动球轨迹。内部仍可维护多个候选，以便真实球短时丢失后从误检
目标切回；将 `max_output_tracks` 设为 `0` 可恢复全部有效运动轨迹输出。

“模型观测优先”指已经通过前处理并被本地 tracker 接受、确认的 YOLO/ONNX
`observed` 轨迹优先于预测。它不表示每一个原始 YOLO 框都绕过检测阈值、
时序运动过滤、轨迹确认和物理关联。JSONL 必须同时检查
`raw_detection_count`、`temporal_motion` 和 `tracker` 诊断，不能只根据最终
是否画出轨迹判断模型有没有检测到球。

## 处理顺序

1. 估计相邻帧的全局画面平移，用背景特征补偿相机抖动和小幅平移。
2. YOLO 输出置信度不低于 `detector.low_conf` 的候选框。
3. 合并同一个球的重叠或嵌套检测框，只保留最高置信度框。
4. 当前 revision 9 使用连续帧运动过滤器检查局部运动证据；该步骤可能拒绝
   缺少局部帧差证据的原始模型候选。
5. 高置信度检测可以创建轨迹；低置信度检测只能恢复已有轨迹。
6. 新轨迹必须同时产生原始画面位移和相机补偿后的位移，才会成为可见运动轨迹。
   尚未完成运动确认的 tentative 轨迹如果经历过漏帧，下一次匹配只作为新的
   确认种子：跨漏帧位移不能直接完成运动确认，旧滤波速度和加速度同时清零，
   必须再由连续观测证明真实运动。
7. 正常关联使用真实帧时间差更新卡尔曼预测，并按 `px/s` 速度、漏检毫秒数、球框尺寸和飞行方向计算代价。
8. 普通飞行中的大幅方向突变会被拒绝。击球和遮挡恢复只有在至少经历配置的
   短时漏检后，才能在有限范围内接回原 ID。落地反弹另走更窄的邻近恢复规则：
   只有主模型观测紧邻最后落地点并形成“先向下、后向上”时才立即重置运动状态。
   若连续两帧都是邻近且原始方向一致的主模型观测，仅 Kalman/NIS 滞后产生冲突，
   则使用真实观测纠正滤波状态，不创建新的碎片 ID。
9. 漏检时只显示短时预测。普通球最多显示 `max_prediction_ms`，高速球最多显示 `fast_max_prediction_ms`，置信度按实际经过时间衰减，越界预测不输出。
10. 已激活轨迹长时间不动后休眠；再次产生真实运动时使用原 ID 唤醒。
11. 本地输出先按 `observed`、`predicted` 分层，再在同层内评分；预测分数不能
    压过已经接受的模型观测。
12. `max_output_tracks: 1` 只输出评分最高的主轨迹；设为 `0` 时输出全部有效运动轨迹。

## 当前参考参数

- `detector.low_conf: 0.12`：更低的检测不会进入后续 pipeline。
- `tracker.low_conf: 0.12`：达到该阈值的检测可参与已有轨迹恢复。
- `tracker.high_conf: 0.20`：达到该阈值的检测可以创建轨迹或触发急变恢复。
- `motion_threshold_px: 12`：运动确认的累计位移门槛，以 1280 像素宽画面为基准。
- `min_motion_speed_px_per_second: 16`：允许持续缓慢滚动的球激活，同时抑制检测框漂移。
- `max_stationary_ms: 1500`：确认轨迹连续静止超过该时间后停止输出。
- `max_speed_px_per_second: 3200`：以 1280 像素参考宽度为基准的速度上限。
  1280 原视频的成对裁切尺度为 1.0，因此每侧仍是 3200 px/s；3840 原视频裁成
  两个 1920 半场时尺度为 3.0，因此每侧上限为 9600 px/s。
- `impact_recovery_gate_px: 260`：击球、反弹和遮挡恢复的基础半径。
- `impact_recovery_min_missing_ms: 20`：跨普通门控或大幅变向恢复前，至少经历一次短时漏检。
- `impact_recovery_max_missing_ms: 120`：人体接触只允许恢复短时漏检；更早的旧轨迹不得借人体框接到远处候选。
- `bounce_recovery_max_displacement_px: 35`：落地反弹的新主模型观测必须位于
  最后观测点 35 px 的邻域内（1280 参考宽度，并随既有帧尺度缩放）。
- `bounce_recovery_min_downward_speed_px_per_second: 100`、
  `bounce_recovery_min_upward_speed_px_per_second: 40`：只接受明确的图像纵向
  下落到上升反转；横向速度达到 40 px/s 时不得反向。
- `bounce_recovery_max_missing_ms: 80`：只允许最近落地点参与反弹恢复，并为
  同一反弹保留一个不续期的 80 ms 上升观测稳定窗口。
- `primary_continuity_gate_px: 55`：连续 YOLO/ONNX 观测纠正滤波滞后时，
  当前点与上一真实观测点的最大距离；仍按 1280 参考宽度缩放。
- `max_flight_direction_change_deg: 60`：非恢复状态下的最大飞行转向角。
- `direction_gate_min_speed_px_per_second: 250`：低于该速度时不启用方向门控，避免低速方向噪声。
- `direction_gate_min_hits: 4`：方向门控从第 4 次已有观测后启用，与 CA 状态成熟
  门槛对齐，避免刚进入新摄像头的轨迹被两三个点形成的临时速度误拒。
- `max_prediction_ms: 120`：普通轨迹最长可见预测时间。
- `fast_prediction_speed_px_per_second: 800`、`fast_max_prediction_ms: 60`：高速球只做极短预测，减少沿旧方向飞出画面的假轨迹。
- `max_missing_ms: 350`：轨迹在内部允许等待重新检测的最长时间，超过后释放 ID。
- `prediction_velocity_decay: 0.90`：漏检期间按实际经过时间降低预测速度。
- `max_output_tracks: 1`：默认单目标输出；设为 `0` 可恢复多目标输出。
- `observation_first_output: true`：本地单球仲裁中 observed 优先于 predicted。
- `output.trail_length: 10`：画面只保留最近 10 个输出位置，减少历史误检形成的长拖尾。

像素距离和 `px/s` 速度阈值以 `reference_frame_width: 1280` 为基准，普通完整画面按宽度自动缩放。裁切不会改变保留区域内的像素密度，因此不能只用某一个裁切后画面的宽度推导门控尺度。revision 9 的正式离线双半场入口使用
`frame_scale_mode: paired_crop_total_width`，按
`(left_width + right_width) / reference_frame_width` 给左右两侧设置同一尺度：
640+640 和 618+662 得到 1.0，1920+1920 得到 3.0。真实独立双摄不属于同源裁切，
必须通过 `runtime.dual_camera_streams.<side>.frame_scale_override` 逐路标定。
该尺度同时用于追踪物理门控、快速运动候选和跨摄交接速度判断。时间门限使用视频
时间戳，与 20、30、60 FPS 等帧率解耦；缺失或异常时间戳使用 `default_fps`
回退。正式部署前仍必须使用最终摄像头、分辨率、视野和帧率重新验收。

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

该功能已在当前正式 `configs/tracking.yaml` revision 9 中开启。旧 temporal 和
physics 配置仅保存在 `legacy/ball_tracking_handoff/configs/maintained_history/`
用于历史回归。

## 当前物理约束

`configs/tracking.yaml` 在时序过滤基础上启用六状态常加速度 KF：

- `motion_model: constant_acceleration`：状态为 `[x, y, vx, vy, ax, ay]`。
- `constant_acceleration_min_observations: 4`：至少四次可靠观测后才允许加速度参与预测。
- `max_acceleration_px_per_second2: 12000`：限制滤波状态中的归一化图像加速度。
- `max_observed_acceleration_px_per_second2: 16000`：拒绝普通飞行中的异常观测加速度。
- `use_nis_gate: true`、`nis_gate_threshold: 10.75`：使用预测协方差和测量残差进行创新门控。
- `continuous_prediction_horizon: true`：预测时长随速度从 120ms 连续缩短至 60ms。

impact recovery 会绕过普通飞行的 NIS、加速度和方向异常，但必须满足高置信度、恢复距离、速度和最短漏检时间要求。恢复成功后旧加速度清零。

落地反弹恢复不使用人体接触的宽恢复半径。它只接受
`yolo`/`onnxruntime` 高置信度观测，并同时要求：旧观测正在向图像下方运动、
新观测转为向上、两点距离不超过 35 px 标定邻域、显著横向运动不反向、
漏检不超过 80 ms。eligible-player 接触区不再否决这条更严格的几何反弹规则，
因为真实落地点可能靠近球员脚部；它仍先经过原有
`max_speed_px_per_second` 上限。任一条件不满足就完全回到原有物理门控路径。
恢复成功后直接使用两次真实观测计算出的向上速度，并清零旧加速度，避免 CA
预测继续向图像下方延伸。同一固定 80 ms 窗口内，后续邻近且继续向上的主模型
观测可以继续重置速度，以吸收反弹初始几帧的加速度变化；该窗口不会因每次匹配
而延长。

连续主模型观测恢复只处理“真实观测序列一致、滤波状态滞后”的情况。上一帧和
当前帧都必须来自 `yolo`/`onnxruntime`，中间没有漏检，两点距离不超过 55 px，
两段原始观测速度的转角不超过 60°，并继续受全局速度上限约束。它不接受
`fast_motion`，也不能把原始观测本身的 90° 转向解释为滤波滞后。命中时使用
当前两次真实观测重置速度和加速度。

这些参数仍是二维像素空间经验值，不代表 `m/s`、`m/s^2` 或真实重力。
当前正式运行由 `apps/track_dual_halves.py` 对两个半场视频成对处理。单路
`apps/track_video.py` 只用于问题定位。若要复查旧 physics r1，必须显式使用
legacy 中的历史配置，并创建新的历史回归 run ID。

## 诊断字段

JSONL 的 `diagnostics.tracker` 包含：

- `impact_recoveries`：本帧通过急变或遮挡恢复接回原 ID 的数量。
- `bounce_recoveries`：本帧通过受限邻近落地反弹规则接回原 ID 的数量。
- `primary_continuity_recoveries`：本帧使用连续主模型观测纠正滞后滤波状态的数量。
- `physical_gate_rejections`：因不合理位移被拒绝的关联数量。
- `direction_gate_rejections`：普通飞行中因方向突变被拒绝的关联数量。
- `nis_gate_rejections`：因创新过大被拒绝的关联数量。
- `acceleration_gate_rejections`：因观测加速度过大被拒绝的关联数量。
- `motion_unconfirmed_tracks`：尚未形成真实运动的内部候选数量。
- `unconfirmed_gap_reseeds`：本帧有多少未运动确认轨迹在漏帧后被重新设为
  连续运动确认种子；这些匹配不会把跨漏帧跳跃变成速度或加速度。
- `stationary_suppressed_tracks`：当前因静止而停止输出的轨迹数量。
- `track_states`：内部轨迹的 ID、预测中心、速度、加速度、漏检帧数和运动确认状态。
- `frame_dt_ms`：本帧用于 Kalman 预测的真实时间间隔。
- `missing_time_ms`：每条内部轨迹距离上次真实检测经过的时间。

## 已知边界

纯运动模型无法在没有任何新检测时知道球已经被球拍击回。因此高速漏检只做很短预测；一旦检测重新出现，再通过急变恢复重置速度方向。要进一步判断击球瞬间，需要增加球拍/人体关键点或专门的事件模型，不能靠无限延长预测解决。

## 双摄 handoff 和高速运动候选

当前 `configs/tracking.yaml` 的双摄流程增加两层保守辅助：

1. `runtime.dual_camera_handoff` 根据当前侧球是否靠近网口、是否朝网口运动，
   为另一侧开启毫秒级接球预警；接收侧仍先做全图 YOLO，漏检后才做网口 ROI
   二次 YOLO。
2. `runtime.fast_motion` 只在 handoff ROI 或已有高速预测 ROI 内工作。候选必须
   尺寸受限、速度在配置范围内并连续出现至少 `min_streak` 帧，才以
   `source=fast_motion` 进入追踪器。

运动候选不能覆盖同帧 YOLO 检测，也不能在全图无约束地创建球。JSONL 中的
`roi_retry_used`、`fast_motion_proposal_count` 和对应 diagnostics 用于区分
YOLO 找回、运动辅助和追踪预测。

人体接触只解除有限范围内的方向、NIS 和加速度异常，不解除最高速度和时间上限。
接触证据要求上一观测点、当前观测点或当前预测点实际进入 eligible player
的扩展框；仅仅因为一条很长的连接线穿过人体框，不构成接触证据。

双摄全局协调器还会对预测、辅助候选和普通重关联执行速度连续性检查，防止本地
ID 变化形成超速长线。revision 9 中，已经确认的主模型观测不会再被协调器这层
重复速度门控否决；物理合法性由本地 tracker 的检测关联负责。
`global_tracking.jsonl` 的 `coordinator.continuity_gate_rejections` 记录协调层
拒绝次数。

模型观测优先不代表可视化必须把所有观测点连成一条线。revision 9 在以下情况
清空对应侧的绘制尾迹，但仍保留当前模型观测点：

- 全局活动摄像头切换；
- 同一侧被选中的本地 `track_id` 改变；
- 同一本地 ID 的相邻显示位置突破按真实时间和当前尺度计算的速度上限。

逐帧原因记录在 `global_tracking.jsonl` 的
`rendering.trail_reset_reason`，运行汇总记录 `trail_reset_frames`。这样不会用
平滑或旧预测遮挡正常检测，也不会用全局固定 `ID 1` 画出虚假的跨轨迹直线。

`pickleball_tracking` revision 9 将“可靠模型观测优先于预测”作为最高层仲裁规则：

1. 当前活动侧存在可靠 YOLO/ONNX `observed` 轨迹时，继续使用当前侧观测；
2. 当前侧只有 `predicted`，但同侧本地 tracker 已输出另一个确认的
   YOLO/ONNX `observed` 轨迹时，同侧真实观测立即替代旧预测并切断旧尾迹；
3. 当前侧只有 `predicted` 或已经丢失，而另一侧本地 tracker 已输出确认的
   YOLO/ONNX `observed` 轨迹时，真实观测立即抢占预测，不要求 handoff 提醒；
4. `fast_motion` 等辅助候选不能使用观测优先通道，仍必须通过严格 handoff；
5. 本地单球输出同样先按 `observed/predicted` 分层，再在同层内评分，预测分数
   无论多高都不能压过已经被 tracker 接受的真实观测；
6. 跨侧切换后在 `switch_lock_ms` 内禁止立即反向切换。

这里的 observation-first 是当前离线 revision 9 的召回优先策略：同侧确认的
主模型观测优先于同侧旧预测；另一侧确认的
主模型观测可以在旧侧只剩预测或丢失时直接抢占，不要求 handoff 已预警。它不等于
真实机器人已经完成严格跨摄状态机。生产运行时仍需把“真实模型观测不能输给旧预测”
与“跨摄切换必须满足时空状态”同时纳入明确状态机。

严格 handoff 继续负责模型漏检后的辅助恢复：源侧必须靠近球网并朝球网运动，
接收候选必须位于入口 ROI、同一本地轨迹连续达到
`receiver_confirmation_hits`，且处于有效时间窗内。未激活 handoff、ROI 外
`fast_motion`、单帧辅助干扰和超时候选均不能接管全局 ID。

物理、方向、NIS 和接触规则仍用于本地检测关联和 ID 继承；它们不能让另一侧
已经确认的模型观测输给旧侧预测。`coordinator.observation_preemptions` 记录
跨侧模型观测抢占预测/丢失侧的次数，
`coordinator.same_side_observation_preemptions` 记录同侧确认主模型观测替代
旧预测的次数。

严格模式还按每路配置的 `left_net_edge` / `right_net_edge` 检查源侧预测。
预测中心一旦越过本侧球网出口边界，该预测仍可保留在本地 tracker 内部，
但不再进入全局输出或视频尾迹；全局状态只能等待接收侧完成上述确认。
切换成功时左右两侧的绘制尾迹同时清空，禁止把两个独立像素坐标系中的点连成一条线。
`coordinator.prediction_boundary_rejections` 记录被球网出口边界抑制的预测数量。
