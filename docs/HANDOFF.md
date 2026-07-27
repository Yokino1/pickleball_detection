# 检测追踪交接说明

Last audited: 2026-07-27

本文是当前里程碑的交接快照。算法规则以 `configs/tracking.yaml` 和
`docs/TRACKING_RULES.md` 为准；历史变化以 `CHANGELOG.md` 为准；板端与量化以
`docs/DEPLOYMENT.md` 为准。本文不替代这些权威来源。

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

当前自动验证为 89 项 unittest，另有引用/边界检查、Python compileall 和
`git diff --check`。

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

## 5. 量化前必须冻结

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

## 6. 输出契约

单视频/文件夹：

```text
<video>_tracked.mp4
<video>_tracking.jsonl
```

离线双摄：

```text
<run_id>_dual_tracking.mp4
<run_id>_left_tracking.jsonl
<run_id>_right_tracking.jsonl
<run_id>_global_tracking.jsonl
<run_id>_manifest.json
```

`BallTrack.source` 的公开 JSON 值保持 `detector`、`prediction` 或 `none`。
具体模型来源 `yolo`、`onnxruntime`、`fast_motion` 当前只用于内部仲裁，不新增
到既有公开字段。任何 JSONL 字段删除、改名或语义变化都必须有兼容方案并更新
schema/CHANGELOG。

## 7. 标准运行与验证

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

## 8. Git 与本地资产注意事项

- 工作开始前和提交前都执行 `git status`；
- 当前工作区可能含用户尚未提交的历史修改，禁止覆盖、回退或清理；
- 禁止 `git add .`，只显式暂存确认过的文件；
- 不提交视频、数据集、PT/ONNX/engine/RKNN、输出、缓存或 DOCX；
- 根目录业务 DOCX 只作背景资料；
- 未经明确要求，不 commit、push 或创建 PR。

当前交接阶段的本地工作区包含与算法文档迁移并存的历史资产变更。后续提交前
必须逐项复查 staged 列表，不得把 dataset 删除或业务 DOCX 自动混入算法提交。
