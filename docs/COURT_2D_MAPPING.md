# R9 球场二维映射（初版）

## 目标与边界

二维映射只消费 R9 双半场协调器最终选中的一个全局球，不参与检测接受、轨迹关联、预测、物理门控或左右相机切换决策。左右相机继续保留各自独立的像素坐标和本地 tracker；每路相机使用独立标定，但统一输出到同一个标准球场坐标系。

当前标定来自固定机位首帧的人工/辅助粗标定。投影视频右侧只绘制标准球场框架和球的二维轨迹，不复制摄像机画面。

显示面板采用横版布局，但不会修改坐标定义：标准坐标的 `Y=0..44` 沿画面
从左向右，左半场显示在左侧，右半场显示在右侧；`X=0..20` 沿画面从上向下，
球网 `Y=22` 显示为中央竖线。这与双半场视频的左/右空间关系一致。
正式双半场输出的右侧投影面板宽度为 1000 px；相较上一版 800 px，球场、
界外空间、轨迹和事件球标记随面板统一放大，同时保持四周 30 ft 的显示范围。

## 标准坐标系

坐标系名称为 `pickleball_full_court_ft`，版本为 `1`，单位为英尺：

- 完整球场宽 20 ft、长 44 ft。
- 原点 `(0, 0)` 位于标准球场图的左上角。
- `X` 从左边线指向右边线，范围为 `0..20`。
- `Y` 从上方底线指向下方底线，范围为 `0..44`。
- 球网位于 `Y=22`。
- 非截击区边界位于 `Y=15` 和 `Y=29`。

框外坐标不会被裁剪。例如 `X=-1.5` 是合法的有限投影结果，同时会记录
`inside_court=false` 和 `outside_court_bounds` 警告。正式配置在球场四周保留
30 ft 的固定显示区域；仍超出画布的极端点会在对应画布边缘显示方向标记和真实
`X/Y` 文字，JSONL 中的原始二维坐标始终不作裁剪或替换。

## 标定与投影

`configs/tracking.yaml` 是唯一正式配置。`runtime.court_projection.cameras.left/right` 分别保存：

- `calibration_id` 和标定来源；
- 标定图像尺寸；
- 至少四个图像关键点；
- 关键点对应的标准球场节点。

每路相机独立估计 `image -> court` 单应矩阵。点数不足、点集退化、矩阵不可逆、条件数异常或标定尺寸与输入视频不一致时，该路投影返回 `unavailable/invalid`，不生成伪坐标。当前粗标定只有四点，重投影误差对这四个拟合点不构成独立精度验证，因此输出会保留低精度警告。

投影点严格使用协调器最终全局 track 的 `center`。可靠模型观测保持 `observed`，R9 的短时预测保持 `predicted`；投影不会改变两者的身份。

## 输出契约

全局 JSONL 新增 `court` 对象；左右 JSONL 填充已有兼容 `court` 字段。核心字段包括：

- `coordinate_system` / `coordinate_system_version`
- `active_side`
- `calibration_id` / `calibration_source`
- `image_xy`
- `ball_court_xy`
- `projection_status` / `projection_valid`
- `homography_available`
- `reprojection_error_px`
- `projection_warnings`
- `track_status` / `observed` / `predicted`
- `inside_court`
- `event.phase` / `event.display_color`
- `event.events`
- `event.bounce_index_since_last_hit`
- `event.candidate` / `event.evidence` / `event.warnings`
- `event.contact_frame_index` / `event.contact_image_xy`
- `event.contact_court_xy` / `event.contact_inside_court`
- `event.metrics`

原有 JSONL 字段不删除、不改义。

投影面板底部同时显示当前 `X/Y`。`OUTSIDE COURT` 表示真实二维坐标位于标准
20×44 ft 矩形外；`EDGE MARKER` 表示坐标比当前可视范围更远，画布边缘的点只
表示方向，精确位置必须读取同一帧 JSONL 的 `ball_court_xy`。

## 裁判事件候选与颜色

`src/court/events.py` 是投影之后的只读解释层。它消费 R9 已接受的全局球、
tracker 诊断计数以及 R9 已经生成的 eligible-player 框，但不向 R9 返回任何
状态，也不启动球拍或其他新增检测模型：

- 荧光绿：飞行或普通状态；空中球即使投影到矩形外也保持绿色。
- 黄色：场内反弹候选。证据可以来自 R9 `bounce_recoveries`，也可以来自连续
  四个 observed 点形成的“向下—最低点—连续向上”图像轨迹。
- 红色：只在反弹候选发生的同一帧，地面投影位于标准矩形外时触发。
- 紫色：R9 `impact_recoveries`，或 eligible-player 身体/挥拍邻域内的显著速度
  突变支持的击拍候选。若 R9 把身体邻域内的急转记录为 bounce，事件层只在显示
  语义中将其重新分类为 hit，不修改 R9 轨迹或诊断。

黄色/红色会锁存至少3帧，并且只有 observed 轨迹连续2帧满足向上速度阈值后
才恢复荧光绿。当前正式配置中紫色显示5帧。击拍后的180 ms 内抑制新的反弹候选，同一运动
突变在160 ms 内只生成一次事件。轨迹尾迹从30帧缩短到15帧。

独立运动学反弹需要看到最低点之后连续两个 observed 点，因此事件生成比真实接触
晚约两帧；`contact_frame_index`、`contact_image_xy` 和 `contact_court_xy`
保存真正转折点，而不是确认时刻已经上升的球位置。速度分量、方向夹角和横纵冲量比
保存在 `event.metrics`，便于后续人工标注和阈值评估。

“击拍通常横向突变更大、落地主要纵向反转”是有效先验，但不能作为唯一规则。
图像 `X/Y` 是相机像素轴，不是球场真实水平/垂直轴；透视会让沿球场纵深的运动同时
改变图像 `Y`，切削或挑球也可能主要改变纵向速度，带旋转落地也可能改变横向速度。
因此当前分类优先组合“现有球员框邻域 + 多帧速度突变”，离开球员身体邻域后才使用
纵向 V 形和横纵冲量比判断反弹。球员脚部带仍允许反弹，避免把脚边落地全部判成击拍。

这些状态全部是 `candidate`，不是正式裁判结论。当前 R9 诊断计数属于某一路
tracker，不直接绑定到具体 track；人体框只能表示接触邻域，没有直接检测球拍。
JSONL 固定保留 `tracker_diagnostic_not_track_scoped`、
`paddle_not_directly_detected`、`image_plane_kinematics_only` 等警告，
禁止把颜色直接解释为已确认判罚。`second_bounce_candidate` 只在同一球场半区
连续累计，并在击拍候选、跨半区或轨迹不连续时重置；正式二弹判罚仍需要逐事件
真值、稳定回合身份和回合状态机。

当前候选规则与显示约定：

- 整个 `X=0..20 ft, Y=0..44 ft` 白色外框含边线，落地接触点只有严格位于外框
  之外时才输出 `out_of_bounds_bounce_candidate`，面板显示红色“出界”。
- 同一半场在球被击回前发生第二次落地时输出 `second_bounce_candidate`，面板显示
  黄色“二弹”。规则语义是“击回前弹了两次”，不是“N 秒内弹两次”。
- `rally_state_timeout_ms` 只是跟踪系统在长时间失联后清除旧回合计数的安全阈值，
  不是比赛规则阈值。
- `event.display_state` 和 `event.display_text_zh` 分别提供稳定机器状态与中文面板
  文本；`fault_candidate`、`fault_reasons` 明确区分展示状态和候选判罚。
- 顶部中文状态条使用缓存字图。Windows 自动寻找微软雅黑/黑体，Linux/RK3588S
  自动寻找 Noto CJK 或文泉驿；板端应通过 `status_font_path` 固定部署字体路径。
- 人体检测框不包含伸出的手臂和球拍，因此事件层另设
  `player_reach_margin_ratio` 有限伸展区。该区域本身不能触发击球，必须再满足：
  同侧换 ID 前后的短间隔速度反转，或跨相机后首次 observed 轨迹达到最小速度。
- `discontinuity_hit_max_speed_px_per_second` 拒绝不可能的跨 ID 图像跳跃。所有速度
  阈值按每路画面尺度缩放，且只影响事件候选，不影响 R9 接受或拒绝任何球。

## 单目物理限制

单应性严格描述球场地面平面。空中球的像素中心经过单应矩阵后，只是“当前视线与球场平面的交点近似”，不是球的真实三维位置，也不是已恢复的落点或高度。所有可用标定都固定携带 `ground_plane_homography_only` 和 `airborne_ball_is_line_of_sight_ground_plane_approximation` 警告。

## 活动模块

- `src/court/layout.py`：标准球场几何与统一坐标定义。
- `src/court/calibration.py`：固定机位关键点、单应矩阵估计与质量检查。
- `src/court/projector.py`：只读消费全局球并生成二维投影结果。
- `src/court/events.py`：只读解释反弹、界外反弹、二弹和击拍候选。
- `src/court/renderer.py`：空白球场框架、越界点及轨迹绘制。
- `src/court/text.py`：缓存中文状态条与无字体时的英文回退。
- `src/court/factory.py`：投影、事件解释和 renderer 的公共装配边界。
- `src/tracking/dual_camera/runner.py`：装配投影器、写 JSONL、拼接投影面板。
- `src/tracking/dual_camera/projection_replay.py`：从固定 R9 run 做无模型重放。

活动代码不依赖或导入 `legacy`。真实摄像头、RKNN、机器人控制、自动球场线识别、动态机位补偿和三维轨迹不属于本阶段。

## 固定 R9 结果的只读重放

本节只描述开发期调试加速方法，不改变最终产品架构。最终板端仍必须对实时摄像头
帧边检测追踪、边选择全局主球、边使用 active side 对应标定执行二维投影和事件
判断；预生成 JSONL 不能参与生产运行。

完成一次正式 R9 双半场运行后，左右 JSONL 已保存逐帧检测、轨迹、球员和 tracker
诊断，全局 JSONL 已保存协调器最终选中的单球，manifest 已保存画面尺寸、FPS 和
输入/输出来源。因此调整二维标定、事件阈值、颜色、尾迹或面板尺寸时，不需要再次
运行检测追踪。

`apps/replay_court_projection.py` 从一个已完成 run 派生新的重放 run：

- 默认读取旧 `dual_tracking.mp4`，保留已经画好的 R9 左右画面，只替换右侧投影
  面板；耗时主要来自大分辨率视频解码和重新编码。
- `--projection-only` 只读取三份 JSONL，完全不解码旧 MP4，只生成纯球场面板，
  适合快速调整事件参数。
- 两种模式都固定记录 `model_inference=false`，不会构建 detector、person
  detector、tracker 或 coordinator，也不会改写来源 run。
- 派生 `global_projection.jsonl` 保留原全局记录并替换 `court` 对象，同时记录
  `projection_replay.source_run_id` 和来源帧号。

离线与板端必须复用同一 `src/court` 几何、标定、投影和事件接口，差别只在上游
数据来源：调试重放读取固定 JSONL，板端在线流程读取当前帧 R9 输出。禁止形成一套
只适用于重放的投影数学或事件语义。
