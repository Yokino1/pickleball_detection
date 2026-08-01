# 检测追踪交接说明

Last audited: 2026-07-31

本文是当前里程碑的交接快照。算法规则以 `configs/tracking.yaml` 和
`docs/TRACKING_RULES.md` 为准；历史变化以 `CHANGELOG.md` 为准；板端与量化以
`docs/DEPLOYMENT.md` 为准。本文不替代这些权威来源。
完整代码和文档导航见 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)。

## 1. 当前正式身份

```text
软件包：pickleball-tracking 0.2.0
正式 profile：pickleball_tracking
正式 revision：9
正式配置：configs/tracking.yaml
正式离线双视频入口：apps/track_dual_halves.py
辅助单路诊断入口：apps/track_video.py
运行方式：在 Conda CMD 直接粘贴命令，不生成运行批处理
目标 SoC：RK3588S
当前发布分支：feature/dual-camera-60fps-rk3588s
```

mainline r1、temporal r1 和 physics r1 已归档到
`legacy/ball_tracking_handoff/configs/maintained_history/`，不再是活动版本。
`configs/tracking_edge.yaml` 只是单路 ONNX Runtime 部署研究基准，不是
revision 9 的等价板端配置，也不是 RKNN 发布配置。

## 2. 已实现并有自动测试覆盖

- 每帧球模型检测、检测去重和 ROI 坐标还原；
- 相机平移估计与连续帧局部运动证据；
- 常加速度 Kalman、真实时间戳、速度/方向/NIS/加速度门控；
- 60～120 ms 短预测、运动确认、静止抑制和轨迹过期；
- 人体模型每 5 帧运行、人体框逐帧延续、eligible player 筛选；
- 人体接触只允许有限 impact recovery，不突破速度和时间上限；
- 本地单球 observed 优先于 predicted；
- 离线双摄左右独立 tracker、全局单球协调、ROI retry 和受限 fast motion；
- 辅助候选的 handoff 预警、入口 ROI、连续确认、超时和切换锁；
- 球网出口越界预测抑制、切换时两侧尾迹清理；
- 成对裁切总宽度自动恢复物理尺度，本地 ID 变化或物理不连续时切断尾迹；
- 硬件无关 `FramePacket`、有界最新帧队列和时间戳配对；
- MP4、单路/左右/全局 JSONL、双摄 manifest 和输出完整性校验。
- 固定机位首帧人工关键点标定、左右独立 `image -> court` 单应矩阵、标准 20×44 ft
  球场框架以及全局主球的只读二维投影；框外点保留，观测/预测身份保持不变。
- 只读裁判事件候选层：15帧投影尾迹，绿色飞行、黄色场内反弹候选、红色场外
  反弹候选、紫色击拍候选，以及自上次击拍候选后的二弹计数。该层组合 R9
  恢复诊断、四个 observed 点的速度分量和已有 eligible-player 框，不运行
  新的球拍检测模型。
- 已完成 R9 run 的二维投影只读重放：复用保存的 MP4、左右/全局 JSONL 和
  manifest，只替换投影面板或生成纯二维视频，manifest 明确记录
  `model_inference=false`。该入口仅用于桌面调试/回归，不是板端部署入口。

当前自动验证为 131 项 unittest，另有引用/边界检查、Python compileall 和
`git diff --check`。

二维映射不改变 revision 9 的检测追踪算法。它只消费协调器最终选中的全局球，
并把结果写入兼容的 `court` 字段和纯框架投影面板。空中球的结果只是当前视线
投到地面球场平面的近似，不能解释为三维位置、球高或真实落点；详细接口见
[`COURT_2D_MAPPING.md`](COURT_2D_MAPPING.md)。

颜色和事件字段目前均为候选：空中投影越界不能判出界；红色只允许由反弹候选
接触点的场外地面投影触发；紫色没有直接球拍检测支持。图像横纵速度也不等同于
真实球场水平/垂直速度。正式裁判功能必须进一步冻结事件数据集、逐事件真值和
回合状态机。

右侧投影面板顶部现显示中文球状态。整个标准球场白色外框（含边线）视为界内，
落地接触点位于外框之外时输出 `out_of_bounds` 犯规候选；同一半场在击回、跨网
或轨迹不连续复位前的第二次落地输出 `second_bounce` 犯规候选。比赛规则本身没有
固定秒数条件，配置中的 `rally_state_timeout_ms` 只用于长时间失联后的旧回合清理。

击球候选层保留一份与 R9 无反馈关系的“持拍伸展区”。如果接触瞬间同侧 local
track ID 改变，必须同时满足短时间间隔、伸展区邻近、明显方向反转和合理速度；
如果在跨相机切换后只有一个 observed 点，则必须满足伸展区邻近和高速首次观测。
超过 `discontinuity_hit_max_speed_px_per_second` 的不可能跨 ID 跳跃会被拒绝。
这些仍是无直接球拍检测支持的候选，不得反向修正 R9 轨迹。

## 3. 必须如实保留的算法边界

### 模型观测优先的范围

observation-first 从“本地 tracker 已经接受并确认模型观测”开始。原始 YOLO/ONNX
框在此之前仍会经过：

```text
置信度阈值
  -> 去重
  -> temporal motion filter
  -> 高低置信度分层
  -> 本地关联与物理门控
  -> 轨迹确认
```

因此出现“JSONL 有 raw detection，但画面没有轨迹”时，不能归类为模型漏检。
必须检查 `raw_detection_count`、`deduplicated_detection_count`、
`diagnostics.temporal_motion` 和 `diagnostics.tracker`。滚动球被时序过滤误拒
仍是需要固定视频验证的高优先级风险。

### 当前跨摄切换不是最终生产状态机

revision 9 对 `fast_motion` 等辅助候选执行严格 handoff；但当旧侧只有预测或
已经丢失时，另一侧确认的 YOLO/ONNX 观测可以直接抢占，无需 handoff 已预警。
这保证真实模型观测不输给旧预测，但仍可能把另一侧误检解释为跨场。

真实机器人上线前必须把两条要求同时实现：

1. 真实模型观测不能被旧预测压制；
2. 跨摄 side switch 必须满足源侧、时间窗、接收入口、方向和锁定状态。

### 两路坐标不能直接比较

左右摄像头保持独立像素坐标系。当前离线成对裁切输入使用
`(left_width + right_width) / 1280` 恢复原视频像素尺度：640+640 得到 1.0，
1920+1920 得到 3.0。该公式不适用于真实独立双摄；真实机位必须逐路标定。
切换摄像头、本地轨迹 ID 变化或相邻显示点突破物理速度上限时清空对应尾迹，
禁止用全局 ID 1 把不连续位置连成直线。模型观测点本身仍保留，不通过平滑延迟
或隐藏正常检测来伪造连续性。

## 4. 尚未实现

- 真实双摄采集驱动和硬件时间戳接入；
- 单一 RKNN 推理调度器与成对球模型接口；
- batch=2、单上下文受控串行和多上下文板端实测选择；
- 非阻塞机器人控制/遥测输出；
- 真实相机尺度、比赛区、观众区、网口和透视标定；
- 生产级严格跨摄状态机；
- RKNN 模型转换、解码适配和正式发布配置；
- 自动 IDF1/HOTA、handoff 准确率和端到端双摄延迟评估工具；
- 经过真实侧拍验收的生产球模型。

`src/runtime/` 当前只有硬件无关帧契约、队列和配对器，不包含伪造的摄像头、
RKNN 或机器人控制实现。

最终板端必须形成单一在线链路：实时双摄采集与配对 → R9检测追踪 → 全局单球
选择 → active side 对应固定标定投影 → 事件解释 → 输出。开发期 JSONL 重放只
用于减少重复推理，不得成为板端依赖或替代在线映射。

## 5. 已确认问题与当前候选修复

固定回归 run
`outputs/experiments/dual_camera/20260729_full_clean_r9_court_events_panel1000_r1`
的左侧约 811～825 帧存在一条已定位的误关联：

- track 37 先把静止误检恢复到一个已失联的未成熟轨迹；
- 随后又关联到更低位置的另一个误检，CA Kalman 状态得到很大的正向图像
  `vy` 和 `ay`；
- 之后没有新观测时，短时预测沿已有状态向图像下方延伸；
- 这不是二维投影引入的重力，也不是最后一个静止框自身在移动，而是旧轨迹 ID
  错误拼接多个不相关检测后的状态外推。

该问题属于 R9 本地关联、恢复和运动确认边界，不属于 `src/court`。2026-07-31
经用户明确授权后，已把实际进入 tracker 的 808～825 帧检测保存为
`tests/fixtures/track37_frames_808_825.json`。当前候选修复只改变未运动确认的
tentative 轨迹：一旦中间发生漏帧，下一次匹配会清除跨漏帧运动证据及滤波速度/
加速度，并从当前点重新等待连续运动确认。已确认轨迹、普通短时预测、impact
recovery、反弹恢复和连续主模型纠偏不走这条重置路径。

固定序列回归与完整 131 项 unittest 已通过；该窗口不再生成错误的已确认轨迹或
向图像下方的预测。尚未重新运行完整双视频，因此当前修复仍是待视频验收的 R9
候选补丁，不应仅凭单元测试宣称已经完成正式回归验收。

当前二维事件层也仍是候选系统：击球、落地、界外和二弹尚未基于更大规模、多机位
真值数据完成精度验收。后续新视频到位后，应先使用无模型重放做事件层调参，再决定
是否需要独立修改 R9。

## 6. 量化前必须冻结

- `ball_best.pt` SHA-256、模型卡和不可覆盖策略；
- 固定 detector 验证集及 tiny/blur/ground/occlusion/no-ball 切片；
- 固定连续追踪和跨摄 handoff 视频；
- 校准数据 revision、SHA-256、抽样方法和实际可访问路径；
- 输入尺寸、颜色顺序、归一化、NMS 和输出解码；
- 完整 RK3588S 板卡/载板、内存、系统、RKNN 版本、功耗和散热；
- 两路摄像头型号、分辨率、曝光、稳定 60 FPS 和时间戳来源；
- 精度、召回、IDF1、handoff、p95 延迟、队列和 soak 验收门槛。

当前本地数据集 manifest/清洗报告不应被假定存在于新 checkout。量化或训练前
必须先验证本地资产，不能用通用图片临时代替校准集。

## 7. 输出契约

单视频/文件夹：

```text
<video>_tracked.mp4
<video>_tracking.jsonl
```

离线双摄：

```text
<output-dir>/<run_id>/
  dual_tracking.mp4
  left_tracking.jsonl
  right_tracking.jsonl
  global_tracking.jsonl
  manifest.json
```

`BallTrack.source` 的公开 JSON 值保持 `detector`、`prediction` 或 `none`。
具体模型来源 `yolo`、`onnxruntime`、`fast_motion` 当前只用于内部仲裁，不新增
到既有公开字段。任何 JSONL 字段删除、改名或语义变化都必须有兼容方案并更新
schema/CHANGELOG。

## 8. 标准运行与验证

离线双摄完整回归使用新的 run ID，以下代码直接粘贴到 Conda CMD：

```cmd
call "D:\anacondaa\Scripts\activate.bat" "D:\anacondaa\envs\torch-cu128"
cd /d D:\ball\ball_tracking_handoff\ball_tracking_handoff

python apps\track_dual_halves.py ^
  --config configs\tracking.yaml ^
  --pair NEW_RUN_ID "LEFT_VIDEO_PATH" "RIGHT_VIDEO_PATH" ^
  --output-dir outputs\experiments\dual_camera
```

交付前验证：

```cmd
D:\anacondaa\envs\torch-cu128\python.exe tools\check_project_refs.py
D:\anacondaa\envs\torch-cu128\python.exe -m unittest discover -s tests -v
D:\anacondaa\envs\torch-cu128\python.exe -m compileall -q apps src tools tests
git diff --check
git status --short
```

## 9. Git 与本地资产注意事项

- 工作开始前和提交前都执行 `git status`；
- 当前工作区可能含用户尚未提交的历史修改，禁止覆盖、回退或清理；
- 禁止 `git add .`，只显式暂存确认过的文件；
- 不提交视频、数据集、PT/ONNX/engine/RKNN、输出、缓存或 DOCX；
- 根目录业务 DOCX 只作背景资料；
- 未经明确要求，不 commit、push 或创建 PR。

当前交接阶段的本地工作区包含与算法文档迁移并存的历史资产变更。后续提交前
必须逐项复查 staged 列表，不得把 dataset 删除或业务 DOCX 自动混入算法提交。

## 10. 最近可复查产物

最终已验证的纯二维重放结果：

```text
outputs/experiments/court_projection_replay/
  20260730_full_clean_r9_court_hits_discontinuity_projection_only_r2/
    projection_replay.mp4
    global_projection.jsonl
    manifest.json
```

该结果 3030 帧、50 FPS，manifest 为 completed，JSONL 为 3030 行；
`model_inference=false`。它用于复查当前投影和事件显示，不替代完整在线双摄流程。
