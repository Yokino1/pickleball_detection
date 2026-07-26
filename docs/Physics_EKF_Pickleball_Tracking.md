# Pickleball 物理约束追踪方案

## 1. 目标与适用范围

本方案用于增强当前 Pickleball 检测追踪主流程，重点解决：

- 检测点在固定背景、灯光、标牌之间跳变；
- 高速球短时漏检后的错误外推；
- 球飞行过程中出现不合理的速度、加速度或方向突变；
- 击球、反弹和遮挡恢复后难以保持原 ID；
- 不同帧率、分辨率下参数含义不一致。

当前阶段仍使用单目视频中的二维图像坐标，不恢复球场投影坐标，也不假设已经完成相机标定。因此，本方案采用**图像空间软物理约束**，不能直接把真实世界的重力 `9.81 m/s^2` 代入像素坐标。

物理模型的主要作用是：

1. 约束检测候选和轨迹关联；
2. 拒绝明显不合理的跳点；
3. 在极短漏检期间提供保守预测；
4. 在击球或反弹发生时允许速度突变并快速重新关联。

物理模型不能替代检测器。连续多帧没有可靠检测时，追踪器必须停止输出预测。

---

## 2. 当前主流程

当前主流程保持不变：

```text
Video frame
    |
    v
Ultralytics ball detector
    |
    v
Detection filtering and deduplication
    |
    v
Camera motion compensation
    |
    v
MultiBallTracker
    |-- Kalman prediction
    |-- physical gating
    |-- detection association
    |-- impact recovery
    |-- stationary suppression
    |-- short missing-detection prediction
    |
    v
Tracked MP4 and JSONL
```

实现应继续位于当前主代码中：

- `src/tracking/ball_detector.py`
- `src/tracking/camera_motion.py`
- `src/tracking/multi_ball_tracker.py`
- `src/tracking/ball_pipeline.py`
- `configs/tracking.yaml`

不要为本方案建立一套与当前主流程平行的 detector、tracker 或 main，也不要依赖 `legacy/handoff_projection`。

---

## 3. 坐标、时间和单位

### 3.1 使用实际时间

所有速度和预测必须使用视频时间戳或 FPS 计算的 `dt`：

```text
position: px
velocity: px/s
acceleration: px/s^2
prediction duration: s or ms
```

不能用“每帧位移”代替速度，否则同一段运动在 30 FPS 和 60 FPS 视频中会得到不同结果。

当时间戳不可用或异常时，回退到：

```text
dt = 1 / default_fps
```

并对异常大的 `dt` 做上限保护，避免解码停顿导致一次预测跨越过远。

### 3.2 按分辨率归一化

像素距离参数以参考宽度为基准：

```text
scale = frame_width / reference_frame_width
normalized_distance = pixel_distance / scale
normalized_speed = pixel_speed / scale
normalized_acceleration = pixel_acceleration / scale
```

门控半径、速度阈值、加速度阈值和静止阈值必须使用同一套缩放规则。

这种归一化只能解决分辨率变化，不能消除透视和景深影响。

### 3.3 不直接使用真实重力

当前状态是图像坐标，真实重力的单位是 `m/s^2`，两者不能直接相加。

在完成相机内外参标定、地面坐标定义和三维定位前：

- 不设置 `ay = 9.81`；
- 不把图像下方固定视为严格重力方向；
- 不使用真实球场高度、落地点或弹跳系数作为硬约束；
- 只估计有边界的图像空间加速度。

---

## 4. 运动模型选择

### 4.1 基线模型：常速度 Kalman Filter

当前模型状态为：

$$
\mathbf{x} =
\begin{bmatrix}
x & y & v_x & v_y
\end{bmatrix}^{T}
$$

状态转移为：

$$
\mathbf{x}_{t+1} =
\begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\mathbf{x}_{t}
$$

该模型的均值按常速度传播，过程噪声用于表达未建模的加速度。它计算量小、稳定，继续作为默认模型和回退模型。

### 4.2 可选模型：常加速度 Kalman Filter

连续获得足够数量的可靠检测后，可启用六状态模型：

$$
\mathbf{x} =
\begin{bmatrix}
x & y & v_x & v_y & a_x & a_y
\end{bmatrix}^{T}
$$

状态转移为：

$$
\mathbf{F} =
\begin{bmatrix}
1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 & 0 \\
0 & 1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

观测仍然只有位置：

$$
\mathbf{z} =
\begin{bmatrix}
x & y
\end{bmatrix}^{T}
$$

该模型是线性的，应使用普通 Kalman Filter，不需要 EKF。

为了防止检测抖动被解释成加速度，必须满足以下条件：

- 至少有 3 至 5 个连续可靠观测后才使用加速度预测；
- `ax`、`ay` 使用归一化上限裁剪；
- 漏检期间对加速度和速度进行衰减；
- 轨迹刚创建、刚恢复或刚发生撞击时回退到常速度模型；
- 加速度模型没有通过回归测试前，不替换默认常速度模型。

---

## 5. 运动模式

单一平滑模型无法同时描述正常飞行和击球瞬间。追踪器应维护轻量状态模式，而不是强迫所有运动符合一条抛物线。

### 5.1 `tentative`

新检测候选尚未确认：

- 不使用强物理结论；
- 需要连续命中和运动确认；
- 不允许长时间预测；
- 固定背景候选继续由静止抑制逻辑淘汰。

### 5.2 `flight`

轨迹连续、残差较小：

- 使用常速度模型，或在条件满足时使用常加速度模型；
- 启用速度、加速度、转向角和创新门控；
- 允许极短漏检预测。

### 5.3 `impact_recovery`

可能发生击球、反弹或遮挡后重现：

- 允许一次超出普通门控半径的高置信度检测；
- 要求候选仍在恢复门控范围内；
- 使用观测间隔重新估计速度；
- 重置或增大速度、加速度协方差；
- 加速度清零或回退到常速度模型；
- 恢复后重新进入 `flight`。

当前代码已有 impact recovery，新增物理约束时必须保留该能力，不能因为转向角过大而直接拒绝真实击球后的检测。

### 5.4 `missing`

当前帧没有可靠检测：

- 只使用滤波器预测；
- 置信度按实际经过时间衰减；
- 速度和加速度逐步衰减；
- 高速球使用更短预测时间；
- 超过预测时间后停止可视化；
- 超过轨迹保留时间后删除轨迹。

---

## 6. 检测关联与软物理约束

物理约束应优先用于判断“这个检测是否可能属于当前轨迹”，而不是用于延长漏检预测。

### 6.1 基础距离门控

基础门控继续考虑：

```text
base gate
+ predicted displacement
+ missing-time growth
```

所有距离按参考分辨率缩放，并设置最大门控半径。

### 6.2 最大速度门控

根据最后可靠观测计算：

$$
v_{obs} =
\frac{\left\|\mathbf{z}_t-\mathbf{z}_{last}\right\|}
{\Delta t}
$$

若归一化速度超过最大允许值，候选应被拒绝。该阈值是图像空间的经验上限，不代表真实球速。

### 6.3 创新门控

Kalman 预测残差为：

$$
\mathbf{r} = \mathbf{z} - \mathbf{H}\hat{\mathbf{x}}
$$

创新协方差为：

$$
\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^{T}+\mathbf{R}
$$

使用归一化创新平方：

$$
\mathrm{NIS} = \mathbf{r}^{T}\mathbf{S}^{-1}\mathbf{r}
$$

NIS 比纯像素距离更合理，因为它会考虑预测不确定度。超过阈值的候选通常拒绝，但 `impact_recovery` 模式可以使用单独、更宽的阈值。

### 6.4 加速度门控

有至少三个可靠观测时，估计观测加速度：

$$
\mathbf{a}_{obs} =
\frac{\mathbf{v}_{obs}-\mathbf{v}_{prev}}
{\Delta t}
$$

加速度约束必须是软约束：

- 小幅超过阈值时增加关联成本；
- 明显超过上限时拒绝；
- 检测中心抖动较大或时间间隔过短时不使用；
- 击球恢复模式不使用普通飞行加速度上限。

### 6.5 转向角门控

飞行模式下，可比较预测速度和观测速度的夹角：

$$
\theta =
\cos^{-1}
\left(
\frac{\mathbf{v}_{pred}\cdot\mathbf{v}_{obs}}
{\|\mathbf{v}_{pred}\|\|\mathbf{v}_{obs}\|}
\right)
$$

转向角只作为关联成本的一部分：

- 低速时不使用，因为方向不稳定；
- 缺少足够历史观测时不使用；
- 高速且残差小的平滑转向可以接受；
- 击球或反弹恢复时允许大幅变向。

### 6.6 尺寸和置信度

YOLO 置信度可以影响测量噪声，但不能单独决定测量是否可靠。测量成本还应考虑：

- 检测框尺寸与历史尺寸差异；
- 中心位置创新；
- 连续帧运动一致性；
- 是否位于排除区域；
- 是否长期静止；
- 是否触发恢复模式。

高置信度固定背景误检仍可能存在，因此测量噪声 `R` 必须设置上下限，不能随 confidence 无限减小。

---

## 7. 短时预测策略

预测时间继续保持保守：

```yaml
max_prediction_ms: 120.0
fast_prediction_speed_px_per_second: 800.0
fast_max_prediction_ms: 60.0
prediction_velocity_decay: 0.90
```

建议进一步支持按速度连续缩短，而不是只有普通和高速两个档位：

```text
prediction_ms =
    clamp(
        base_prediction_ms
        - speed_ratio * reduction_ms,
        min_prediction_ms,
        base_prediction_ms
    )
```

其中 `speed_ratio` 使用归一化速度计算并限制在 `[0, 1]`。

预测输出还必须满足：

- 预测中心仍在合理画面范围内；
- 预测置信度高于最低阈值；
- 没有被静止抑制；
- 没有超过最大缺失时间；
- 不因常加速度模型产生突然增大的位移；
- 预测轨迹使用实线，但应通过状态颜色或标签区分 observed 和 predicted。

更复杂的物理模型不能成为延长预测时间的理由。

---

## 8. 相机运动补偿

当前相机运动模块使用背景角点光流和 RANSAC 估计全局平移。滤波预测和运动判断必须在补偿相机运动后进行。

需要注意：

- 平移补偿不能处理明显旋转、缩放和透视变化；
- 相机运动估计置信度不足时，不应用补偿；
- 相机运动异常帧应增大过程噪声或放宽一次关联门控；
- 不能把相机抖动残差解释成球的真实加速度。

未来如果手持拍摄中旋转和缩放较明显，可以评估仿射变换或单应性补偿，但应先用回归视频验证稳定性和板端开销。

---

## 9. 轻量连续帧检测

当前检测器逐帧独立推理，追踪器虽然使用历史帧状态，但检测模型本身没有直接输入连续帧。

第一阶段不建议直接替换成大型时序网络。可优先尝试轻量方案：

1. 保留当前单帧 YOLO；
2. 从最近 2 至 3 帧构造帧差或运动区域；
3. 使用运动区域对 YOLO 候选重新评分或过滤；
4. 在预测位置附近设置动态 ROI；
5. ROI 检测失败时回退到全帧检测；
6. 保持周期性全帧扫描，避免轨迹丢失后无法恢复。

该方案需要特别防止相机抖动产生整幅帧差，因此帧差必须在相机运动补偿后计算。

训练真正的连续帧模型可以作为后续独立实验，例如输入相邻帧堆叠、轻量时序特征融合或 TrackNet 类热图模型。是否采用必须根据板端速度、量化精度和真实回归数据决定。

---

## 10. 配置建议

建议新增参数时保持默认关闭，逐项完成消融测试：

```yaml
tracker:
  motion_model: constant_velocity

  physics_gating:
    enabled: false
    use_nis_gate: true
    nis_gate_threshold: 9.21
    use_acceleration_gate: true
    max_acceleration_px_per_second2: 12000.0
    acceleration_cost_weight: 0.15
    use_turn_angle_gate: true
    min_turn_speed_px_per_second: 250.0
    max_flight_turn_angle_deg: 65.0
    turn_angle_cost_weight: 0.15

  constant_acceleration:
    enabled: false
    min_observations: 4
    acceleration_decay: 0.80
```

以上数值只是初始实验值，均以 `reference_frame_width` 为基准，不能在缺少回归数据的情况下直接认定为最终参数。

不建议一次同时启用所有约束。推荐顺序：

1. NIS 门控；
2. 转向角软成本；
3. 加速度软成本；
4. 常加速度状态模型；
5. 多模型或 IMM。

---

## 11. 当前实验实现

当前代码已经提供一版可选物理约束实验，通过
`configs/tracking_physics.yaml` 启用，不改变默认
`configs/tracking.yaml`：

- 六状态常加速度 KF：`[x, y, vx, vy, ax, ay]`；
- 连续可靠观测达到 `constant_acceleration_min_observations` 后才使用加速度；
- 加速度按参考分辨率归一化并受 `max_acceleration_px_per_second2` 限制；
- 漏检期间同时衰减速度和加速度；
- impact recovery 后使用观测重估速度，并将旧加速度清零；
- 使用 NIS、观测加速度、速度和转向角共同约束关联；
- 高速球的预测时长在 `max_prediction_ms` 和
  `fast_max_prediction_ms` 之间连续缩短；
- JSONL 输出 `motion_model`、`acceleration`、NIS 拒绝数量和加速度拒绝数量。

该版本仍然是**图像空间物理约束实验**，加速度单位是
`px/s^2`，没有使用真实重力、空气阻力、三维球场坐标或
地面碰撞方程。

运行入口：

```bat
scripts\run_physics_tracking.cmd
```

结果写入：

```text
outputs/experiments/desktop_ab/physics
```

## 12. 实施阶段

### 阶段 A：增强现有常速度模型

- 保持当前四状态 Kalman Filter；
- 增加 NIS 诊断和可选门控；
- 增加归一化转向角、加速度诊断；
- 将物理量写入 JSONL；
- 先只作为关联成本，不立即硬拒绝；
- 使用固定回归视频调参。

这是当前最优先、风险最低的阶段。

### 阶段 B：事件感知和常加速度实验

- 明确 `flight`、`impact_recovery`、`missing` 状态；
- 加入六状态常加速度 KF 实验开关；
- 在击球恢复后重置加速度；
- 比较 CV、CA 和切换模型；
- 验证不同 FPS、分辨率和球速。

### 阶段 C：轻量连续帧检测

- 帧差运动区域辅助候选过滤；
- 预测 ROI 与全帧回退；
- 对相机抖动和固定 overlay 做专项测试；
- 评估板端推理速度和量化损失。

### 阶段 D：标定后的三维物理模型

只有完成相机标定和三维定位后，才考虑状态：

$$
\mathbf{x}_{3D} =
\begin{bmatrix}
X & Y & Z & V_X & V_Y & V_Z
\end{bmatrix}^{T}
$$

这时才可以合理加入：

- `9.81 m/s^2` 重力；
- 二次空气阻力；
- Magnus 力；
- 地面碰撞和恢复系数；
- 球网、地面和球场边界；
- EKF 或 UKF 非线性观测。

该阶段不属于当前二维检测追踪主流程的近期范围。

---

## 13. 回归测试与评价指标

不能只通过轨迹看起来更平滑来判断模型有效。至少记录：

- 检测召回率；
- 误检进入主轨迹的次数；
- ID switch 数量；
- 击球后保持原 ID 的成功率；
- 1、2、3 帧漏检预测的平均和 95 分位像素误差；
- 不合理速度、加速度和转向的数量；
- 真实检测被物理门控错误拒绝的数量；
- 每帧处理时间、显存和内存；
- FP32、FP16、INT8 下的检测与追踪差异。

回归样本应覆盖：

- 近距离和远距离球；
- 30 FPS、60 FPS 及更高 FPS；
- 不同分辨率；
- 高速扣杀、慢速吊球、击球变向和落地反弹；
- 相机固定、轻微抖动和明显移动；
- 墙灯、标牌、固定 overlay 等典型误检；
- 连续 1 至 5 帧漏检。

每次只改变一组参数，并保留对应 MP4、JSONL 和指标结果。

---

## 14. 最终原则

1. 当前二维图像坐标中不硬编码真实重力。
2. 先用物理约束提高关联正确率，再考虑更复杂的预测模型。
3. 正常飞行使用平滑约束，击球和反弹必须允许速度突变。
4. 高速球预测时间比低速球更短。
5. 预测只用于短时补洞，不能代替连续检测。
6. 所有阈值使用实际时间和参考分辨率归一化。
7. 默认继续使用轻量常速度 KF，新增能力通过配置逐项启用。
8. 是否升级到 CA、IMM、EKF 或连续帧模型，由固定回归数据决定。
