# 人体接触门控与透视补偿追踪方案

## 1. 功能目标

本功能研究一个更可靠的击球恢复条件：

```text
只有球接近球员的有效接触区域，并同时出现可信的运动突变时，
才允许临时放宽方向、加速度、NIS 和关联距离约束。
```

它用于减少当前 `impact_recovery` 把灯光、标牌、鞋子或其他错误检测接回原 ID 的情况。

该功能最初作为独立实验实现，现已合入唯一正式配置
`configs/tracking.yaml` revision 9。人体支路只为 impact recovery 提供接触门控，
不能创建球，也不能覆盖可靠球模型观测。

### 当前实现状态

```text
配置：configs/tracking.yaml
正式入口：apps/track_dual_halves.py
辅助单路诊断：apps/track_video.py
输出：outputs/experiments/current
人体模型：artifacts/models/yolo11n.pt（COCO person 类，本地权重不提交 Git）
```

已经实现：

- 在桌面和后续板端配置中固定每 5 帧运行一次人体检测；
- 检测间隔内每帧延续人体框和稳定人体 ID；
- 使用脚点比赛区域、观众排除区、持续帧数和最大人数做比赛人员初筛；
- 只有球的观测线段进入 `eligible_player` 扩展框，才允许大角度恢复；
- JSONL 保留全部人体框、筛选分数及门控诊断；当前正式 MP4 默认
  `draw_players: false`，需要调试时才显式显示人体框。

尚未实现：

- 手腕关键点和球拍检测；
- 基于球员长期运动/击球历史的自动观众分类；
- 标定后的场地多边形和真实三维深度；
- 落地反弹与人体击球的独立事件分类。

正式 YAML 的通用 `max_players` 为 4；离线双摄 CLI 默认通过
`--max-players-per-half 2` 将每个半场限制为 2 名 eligible player。真实双摄
仍需按单打/双打和每路视野配置，不能把这个 CLI 覆盖值误认为相机标定结果。

因此当前二维实现仍需人工配置比赛区域或 `spectator_exclusion_regions`。它只提供较保守的二维接触证据，
不能把球与人体框重叠解释为真实击球。

---

## 2. 为什么远近和透视必须考虑

### 2.1 像素速度不等于真实速度

针孔相机中，三维点投影到图像近似为：

$$
u = f_x \frac{X}{Z} + c_x
$$

$$
v = f_y \frac{Y}{Z} + c_y
$$

其中 `Z` 是目标到相机的深度。同样的真实位移，在近处产生更大的像素位移，在远处产生更小的像素位移。

因此可能出现：

- 球真实速度不变，但接近相机时 `px/s` 增大；
- 球真实直线运动，但二维投影看起来弯曲；
- 球沿深度方向运动时，图像速度和加速度明显变化；
- 相同 `max_speed_px_per_second` 对近处过严、对远处过宽。

当前按视频时间计算 `px/s` 解决了不同 FPS 的单位问题，按画面宽度归一化解决了不同分辨率问题，但没有消除深度和透视影响。

### 2.2 视觉重叠不代表真实接触

单目画面只能确认球和人在二维图像中接近。球可能：

- 实际位于球员前方或后方；
- 从身体旁边飞过；
- 被身体遮挡但没有接触球拍；
- 与远处球员在画面上重叠。

所以“球进入人体框”只能作为接触候选，不能直接判定击球。

### 2.3 其他视觉误差

除透视外，还需要考虑：

- 高速运动模糊导致球框中心偏移；
- 球只有几像素时，中心抖动会形成很大的 `px/s²`；
- 滚动快门导致高速球位置倾斜或拉伸；
- 相机平移、旋转和缩放残差；
- 人体遮挡导致球连续漏检；
- 人体框和手腕关键点自身抖动。

---

## 3. 总体架构

```text
Video frame
    |
    +--> Ball detector, high resolution, every frame
    |
    +--> Person/pose detector, low resolution, low frequency
              |
              v
         Person tracker
              |
              v
       Contact-zone estimator
              |
              v
Ball tracking + perspective-aware gates
    |-- normal flight constraints
    |-- player-contact candidate
    |-- post-impact recovery
    |-- ground-bounce candidate
    |
    v
MP4 + JSONL diagnostics
```

球检测仍然是最高优先级。人体检测不得阻塞球检测，也不能因为某一帧人体漏检就删除球轨迹。

---

## 4. 人体与接触区域

### 4.1 第一阶段：人体框近似

人体检测器输出：

```text
person_bbox = [x1, y1, x2, y2]
confidence
track_id
```

人体框高度：

```text
person_height = y2 - y1
```

使用人体高度作为局部透视尺度近似。接触距离不使用固定像素，而使用：

```text
normalized_contact_distance =
    pixel_distance_to_contact_zone / person_height
```

这样近处球员框大，允许区域按比例变大；远处球员框小，允许区域自动缩小。

人体框不应整体作为接触区。第一阶段可使用：

- 人体框上半部分；
- 人体框左右两侧扩展区；
- 排除头部中心和腿部中心；
- 优先选择离球最近的球员。

这是低成本近似，不能证明真实球拍接触。

### 4.2 第二阶段：手腕关键点

轻量 pose 模型提供：

```text
left_wrist
right_wrist
person_bbox
keypoint_confidence
```

围绕每个高置信度手腕建立接触区：

```text
contact_radius =
    clamp(
        person_height * contact_radius_ratio,
        min_contact_radius_px,
        max_contact_radius_px
    )
```

建议初始范围：

```text
contact_radius_ratio: 0.15～0.25
```

手腕区域比整个人体框可靠，但球拍可能伸出手腕较远，仍需适当扩展。

### 4.3 第三阶段：球拍检测

若有球拍标注和轻量模型，接触区可由球拍框直接生成：

```text
paddle_contact_zone =
    expanded(paddle_bbox, uncertainty_margin)
```

球拍检测是最直接的视觉证据，但需要专用数据集，远距离和运动模糊下也可能漏检。

### 4.4 比赛人员与观众筛选

通用人体模型只能输出 `person`，不能自动区分：

- 比赛球员；
- 场边观众；
- 裁判；
- 教练；
- 捡球人员；
- 从背景经过的人。

不能让所有人体框都触发 contact recovery。人体检测后必须增加轻量的 `PlayerSelector`，只把
符合条件的人体轨迹标记为：

```text
eligible_player
```

接触门控只消费 `eligible_player`，忽略普通观众轨迹。

#### 固定机位的空间先验

固定机位优先配置：

```text
court_polygon
expanded_play_area
spectator_exclusion_regions
```

判断人体位置时使用人体框底边中心，也就是近似脚点：

```text
foot_point = ((x1 + x2) / 2, y2)
```

脚点比人体框中心更适合判断人是否站在球场内。人体框可能向上覆盖墙面或观众席，但脚点通常仍能表示人的落脚区域。

空间规则应采用软评分：

- 脚点在球场多边形内：明显加分；
- 脚点在球场外但位于扩展比赛区：小幅加分；
- 脚点位于明确观众区：明显减分；
- 人体框与球场相交但脚点在观众区：不能直接认为是球员。

不要把球场外全部设为硬排除。球员可能冲出边线或底线救球。

#### 轨迹和活动范围

为每个人体维护稳定 ID，并在几秒时间窗内统计：

- 脚点是否主要位于比赛区域；
- 移动范围和平均速度；
- 是否持续出现在同一球场侧；
- 人体框尺寸是否符合该位置的透视尺度；
- 是否频繁靠近球轨迹；
- 是否曾经形成可信 contact candidate。

观众也可能移动，所以“运动的人就是球员”不成立；比赛间歇时球员也可能静止，所以“静止的人就是观众”同样不成立。

#### 人数与球场侧约束

配置比赛类型：

```text
singles: 2 players
doubles: 4 players
```

在初始化时间窗内，从比赛区域中的人体轨迹选出得分最高的 2 或 4 个主球员。可以结合球网两侧限制：

- 单打：每侧最多一个主球员；
- 双打：每侧最多两个主球员；
- 主球员短时遮挡时保留身份，不立即用观众替换；
- 替换主球员需要持续证据和滞回。

人数上限是重要约束，但不能简单选取置信度最高或人体框最大的几个人，因为近处观众往往框更大。

#### 球交互历史

球员分数可加入与球的历史关系：

```text
player_score =
    court_location_score
  + track_persistence_score
  + side_consistency_score
  + perspective_size_score
  + movement_area_score
  + ball_proximity_history_score
  + confirmed_contact_history_score
  - spectator_region_penalty
```

其中球接近人体只提供加分，不能单独把观众升级为球员。远处观众可能与球在二维画面上偶然重叠。

使用进入和退出滞回：

```text
player_enter_threshold > player_keep_threshold
```

避免球员身份在连续帧中频繁切换。

#### 姿态模型的计算优化

人体检测器可以检测所有人，但姿态模型没有必要处理所有观众。推荐两级调度：

```text
低频 person detector
    |
    v
PlayerSelector
    |
    v
只对 eligible players 的 ROI 运行 pose
```

这样既减少观众手腕误触发，也降低板端姿态推理开销。

#### 无球场标定时的降级

若暂时没有球场多边形，可先人工配置：

- 比赛区域矩形或多边形；
- 观众席排除区域；
- 球网大致位置；
- 单打或双打人数。

固定机位下，人工多边形通常比训练一个“球员/观众分类模型”更稳定、更轻量。相机明显移动后，该人工区域必须失效，不能继续硬套。

---

## 5. 透视尺度补偿层级

### 5.1 Level A：无标定轻量方案

当前可立即实验：

- 全局参数继续按 `reference_frame_width` 缩放；
- 球员附近的接触半径按 `person_height` 缩放；
- 人体框和手腕只用于事件门控；
- 普通飞行仍使用当前 `px/s`、方向、NIS 和加速度约束；
- 不声称得到真实米制速度或真实接触。

优点：

- 无需相机标定；
- 改动和板端开销较小；
- 可以直接验证人体信息是否减少错误折线。

限制：

- 无法解决球和人在不同深度但视觉重叠的问题；
- 人体高度受姿态影响；
- 球远离球员时仍缺少局部深度尺度。

### 5.2 Level B：弱标定透视尺度图

固定机位下，可利用球场线和已知尺寸建立地面单应性。虽然空中球不能直接投影到地面坐标，但可以得到近似的画面深度尺度：

```text
local_scale(u, v) = expected_pixels_per_meter_at_image_location
```

可用于：

- 根据球员脚点估计球员所处深度；
- 对附近球的速度和关联半径做局部尺度归一化；
- 区分远端和近端球员；
- 建立地面反弹候选区域。

需要注意：地面单应性对空中球仍然只是近似，不能当作真实三维位置。

### 5.3 Level C：完整标定和三维定位

双相机或其他可靠深度来源下，可建立：

```text
[X, Y, Z, Vx, Vy, Vz]
```

这时才能：

- 使用真实 `m/s` 和 `m/s²`；
- 判断球与球拍是否在三维空间接近；
- 使用真实重力和空气阻力；
- 区分视觉重叠与真实接触；
- 建立准确落地和反弹模型。

该层级不属于近期板端轻量版本。

---

## 6. 接触事件状态机

### 6.1 `flight`

普通飞行状态：

- 使用常规距离、速度、方向、加速度和 NIS 门控；
- 不允许仅因大角度变化就自动放宽关联；
- 保持短时预测限制。

### 6.2 `player_contact_candidate`

满足以下条件时进入候选状态：

- 球轨迹此前已稳定；
- 球中心或预测区域接近人体有效接触区；
- 人体/手腕/球拍证据达到最低置信度；
- 接触距离使用人体尺度归一化；
- 时间上接近当前帧，而不是使用很久以前的人体框。

候选状态只持续很短时间，例如：

```text
60～120ms
```

### 6.3 `post_impact`

在接触候选窗口内，同时出现以下一项或多项：

- 速度方向明显改变；
- 观测加速度显著上升；
- NIS 超过普通飞行阈值；
- 球在遮挡后重新出现；
- 新检测在接触区域合理出口方向上。

此时允许一次受限恢复：

- 放宽方向门控；
- 使用单独的 contact recovery 半径；
- 用新观测重置速度；
- 清空击球前加速度；
- 增大状态协方差；
- 要求后续 2～3 个检测重新确认。

若后续检测不连续，应撤销该候选，不能把错误点长期保留为主轨迹。

### 6.4 `ground_bounce_candidate`

落地反弹不一定靠近球员，必须独立处理：

- 球接近估计地面区域；
- 垂直运动方向发生合理反转；
- 水平速度变化受限；
- 不使用人体接触要求。

在没有球场弱标定前，该状态只应作为软候选，不做硬结论。

---

## 7. 关联评分建议

普通关联成本可扩展为：

```text
cost =
    position_cost
  + size_cost
  + direction_cost
  + acceleration_cost
  + nis_cost
  - detection_confidence_bonus
```

接触候选增加：

```text
contact_evidence =
    person_confidence
  * keypoint_or_paddle_confidence
  * proximity_score
  * temporal_freshness
```

只有 `contact_evidence` 超过阈值时，才能使用 contact recovery。不能用人体框存在与否作为二元硬开关。

建议使用滞回：

```text
enter_threshold > keep_threshold
```

避免球在接触区边缘时状态频繁切换。

---

## 8. 板端双模型调度

### 8.1 推荐调度

```text
球模型：
    高分辨率
    每帧运行

人体/姿态模型：
    320 或 416 输入
    每 3～10 帧运行一次

人体中间帧：
    Kalman、光流或轻量框追踪
```

例如 25 FPS 视频：

```text
球模型：25 次/秒
人体模型：5 次/秒
人体追踪：其余 20 帧
```

人体框变化比高速球慢，不需要每帧做完整人体推理。

最终目标不是 25 FPS，而是两路各 60 FPS：球模型总负载 120 张图/秒，人体模型
按每 5 帧一次总负载 24 张图/秒。25 FPS 这里只是说明检测间隔，不是部署预算。

### 8.2 运行时策略

- 最终 RK3588S 方案由一个推理调度器统一拥有 NPU；
- batch=2、受控串行或多上下文必须实测，不能提前写死；
- 共享解码和颜色转换结果；
- 人体模型超时可以跳过，球追踪必须继续；
- 人体结果带时间戳，超过有效期后不能继续触发接触；
- 优先量化人体模型，球模型量化必须单独验证小球召回。

### 8.3 无法提前保证的内容

双模型能否实时运行取决于：

- RK3588S 完整板卡、NPU/RKNN 版本和功耗模式；
- 可用 RAM/显存；
- 视频分辨率和目标 FPS；
- 模型输入尺寸；
- FP16/INT8 支持；
- 视频解码开销；
- 板卡对多模型图的调度能力。

SoC 已确认是 RK3588S，但完整载板、相机、RKNN 环境和功耗模式尚未冻结，
因此仍不能承诺最终实时 FPS。

---

## 9. 当前模块边界

当前采用：

```text
src/tracking/person_detector.py
src/tracking/person_tracking.py
```

职责：

- `person_detector.py`：人体框或姿态推理；
- `person_tracking.py`：低频检测之间维护人体 ID 和框，并完成比赛人员初筛；
- `multi_ball_tracker.py`：消费 contact evidence，不负责运行人体模型；
- `ball_pipeline.py`：按频率调度两个检测器并统一时间戳；
- `types.py`：定义 person 和球追踪的稳定序列化结构；当前没有 wrist 数据契约；
- `ball_pipeline.py` 的 diagnostics：记录人体检测、eligible player 和接触门控相关诊断。

不要把人体推理直接写进 `MultiBallTracker`，否则追踪逻辑会与具体模型耦合。

---

后续加入手腕/球拍和事件状态机时，再拆出独立 `contact_gate.py`。

## 10. 当前配置

完整配置见 `configs/tracking.yaml`，关键部分如下：

```yaml
runtime:
  person_detection:
    enabled: true
    model: artifacts/models/yolo11n.pt
    person_class_id: 0
    conf_threshold: 0.30
    imgsz: 416
    interval_frames: 5
    box_tracker:
      max_missing_detection_runs: 3
      association_iou: 0.10
      center_gate_height_scale: 0.80
    player_selection:
      max_players: 4
      min_track_hits: 1
      play_area_normalized: [0.0, 0.20, 1.0, 1.0]
      spectator_exclusion_regions: []

tracker:
  require_contact_for_impact_recovery: true
  contact_margin_ratio: 0.20
```

`play_area_normalized` 和 `spectator_exclusion_regions` 均使用 `[x1, y1, x2, y2]`
归一化坐标。不同机位不应盲目共用同一观众排除区；需要按固定机位建立配置。

---

## 11. 数据和标注要求

为了继续验证该功能，需要从现有视频中建立接触事件集：

- 每个人体轨迹的球员、观众、裁判或其他身份；
- 球场区域、扩展比赛区和观众排除区；
- 球员人体框或姿态关键点；
- 球轨迹中心；
- 击球发生帧；
- 击球球员 ID；
- 落地反弹帧；
- 球从人体附近经过但没有击打的困难负样本；
- 球与远处球员视觉重叠但深度不同的样本；
- 人体遮挡后球重新出现的样本。

击球帧可允许一个小时间范围，例如人工标注：

```text
contact_frame ± 1 frame
```

不同 FPS 视频应同时记录毫秒时间，不能只记录帧号。

---

## 12. 评价指标

除现有检测和追踪指标外，增加：

- contact candidate precision/recall；
- 真实击球后保持原 ID 的比例；
- 非击球经过人体附近时的误触发率；
- eligible player precision/recall 和球员 ID 切换次数；
- 观众被错误选为接触对象的次数；
- 错误大角度恢复次数；
- 落地反弹被人体规则错误拒绝的次数；
- 人体结果过期导致的错误恢复次数；
- 球模型、人模型和组合流水线 p50/p95 延迟；
- 峰值内存；
- FP16/INT8 下接触事件准确率。

必须分别报告 20、25、50、60 FPS 视频，因为事件窗口虽然使用毫秒，离散采样和运动模糊仍然不同。

---

## 13. 主要风险与降级策略

### 风险

- 人体框过大，误把经过身体附近的球当作击球；
- 人体框过小或关键点漏检，真实击球无法恢复；
- 单目深度歧义无法完全消除；
- 人体模型增加板端延迟；
- 两个球员靠近时选择错误球员；
- 近处观众人体框更大，挤掉远处真实球员；
- 球员冲出场地时被空间规则错误移除；
- 相机移动后人工球场区域失效；
- 落地反弹被错误套用人体接触规则；
- 人体检测结果过期仍触发恢复。

### 降级

```text
人体证据可靠：
    使用 contact-gated recovery

人体证据缺失或过期：
    回退到严格普通飞行门控
    允许极短预测
    不自动扩大恢复范围

全局相机运动异常：
    暂停接触判断
    增大状态不确定度
```

人体模型只能提供额外证据，不能成为球检测的硬依赖。

---

## 14. 推荐实施顺序

以下 Phase 0～3 是该功能最初的实施路线。低频人体检测、PlayerSelector 和
接触门控已经合入 revision 9，但尚未完成固定接触事件集上的完整影子对照；
因此“代码已合入”不等于“人体接触准确率已验收”。

### Phase 0：标注和离线验证

- 从现有视频标注击球、非击球近身经过和落地反弹片段；
- 统计当前 `impact_recovery` 的真实成功和误触发；
- 建立固定验收清单。

### Phase 1：人体框低频实验

- 桌面端接入轻量 person detector；
- 每 5 帧检测一次，中间帧追踪；
- 配置球场多边形、观众排除区和单打/双打人数；
- 先评估 PlayerSelector 能否稳定选出比赛人员；
- 只记录 contact evidence，不改变现有轨迹；
- 检查人体高度是否能稳定反映局部透视尺度。

### Phase 2：影子门控

- 新规则与旧规则并行计算；
- JSONL 同时记录两者决策；
- 不让新规则控制输出；
- 统计新规则会减少多少错误恢复，又会拒绝多少真实击球。

### Phase 3：接触门控生效

- 只有接触候选窗口允许大角度恢复；
- 保留 ground bounce 独立分支；
- 对恢复后的 2～3 帧重新确认；
- 与当前 revision 9 的接触门控关闭/开启结果做固定 A/B 测试。

### Phase 4：姿态或球拍增强

- 若人体框误触发仍高，再增加手腕关键点；
- 只有数据证明需要时才训练球拍模型；
- 不先假设更复杂模型一定更好。

### Phase 5：板端验证

- 确定板卡 SKU 和目标 FPS；
- 导出人体模型 ONNX/原生格式；
- 测试检测间隔、输入尺寸和量化；
- 记录组合延迟、内存、温度和精度；
- 达不到实时要求时优先降低人体检测频率，不降低球模型优先级。

---

## 15. 当前结论

该功能已经进入正式 pipeline，但仍应把人体信息理解为“接触概率证据”，不能
理解为真实三维接触证明。

近期最合理方案是：

```text
低频轻量人体检测
    +
球场脚点与持续轨迹筛选比赛人员
    +
只对比赛人员运行姿态/接触分析
    +
人体高度归一化接触距离
    +
短时 contact candidate 状态
    +
速度/方向/NIS/加速度共同确认
    +
独立 ground bounce 分支
```

后续扩大接触恢复范围、加入姿态或改变正式输出之前，必须先以影子模式验证收益。
真实解决前后深度错位，需要弱标定尺度图或双目三维定位。
