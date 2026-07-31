# 项目结构与模块责任

Last audited: 2026-07-31

本文是活动代码和维护文档的结构索引。算法身份以
[`VERSIONS.md`](VERSIONS.md) 和 `configs/tracking.yaml` 为准，精确追踪规则以
[`TRACKING_RULES.md`](TRACKING_RULES.md) 为准，当前工作快照以
[`HANDOFF.md`](HANDOFF.md) 为准。

## 1. 稳定分层

```text
用户入口
  apps/track_dual_halves.py
  apps/track_video.py
  apps/replay_court_projection.py
        |
        v
流程编排
  src/tracking/dual_camera/runner.py
  src/tracking/dual_camera/projection_replay.py
        |
        +---------------------------+
        v                           v
R9 检测追踪核心                 二维球场只读附加层
  src/tracking/*                  src/court/*
        |                           |
        +-> 全局单球协调结果 --------+
                                    |
                                    v
                         court JSON + 空白投影面板
```

允许的主依赖方向是：

```text
apps -> orchestration -> tracking core
                      -> court
tools/tests -> public module interfaces
```

禁止的方向：

- `src/tracking` 核心不得导入 `src/court`；
- `src/court` 不得决定检测是否接受、轨迹关联、预测、handoff 或全局选球；
- 活动代码不得导入 `legacy`；
- 活动运行时不得导入 `ground_detection`；
- 核心模块不得导入 `apps`；
- 桌面重放不得成为板端生产依赖。

## 2. 顶层目录

| 路径 | 责任 | 维护规则 |
| --- | --- | --- |
| `apps/` | 用户可直接运行的薄 CLI | 只解析参数和调用公共流程，不实现算法规则 |
| `configs/` | 正式算法及研究配置 | `tracking.yaml` 是唯一正式 R9 配置 |
| `src/tracking/` | R9 检测、关联、预测、人体辅助和公共契约 | 正式主线，投影功能不得反向修改 |
| `src/tracking/dual_camera/` | 离线双视频编排、全局单球、渲染、产物和重放 | 不承载真实摄像头、RKNN 或机器人控制 |
| `src/court/` | 标准场地、人工标定、单应投影、候选事件和面板 | 只读消费最终全局球 |
| `src/runtime/` | 未来板端实时契约 | 当前只有帧包、队列和时间戳配对；硬件接入待实现 |
| `ground_detection/` | 员工交付的独立首帧球场线/关键点工具 | 标定上游工具，不是活动包，不被运行时 import |
| `tests/` | 单元和契约回归 | 新行为必须增加测试，R9 基线测试数只能增加 |
| `tools/` | 引用检查和维护工具 | 不保存正式运行批处理 |
| `docs/` | 权威设计、规则、交接和维护记录 | 按本文第 7 节的权威关系同步 |
| `legacy/` | 原始交接与退役实现 | 只作设计参考，禁止活动依赖 |
| `data/`、`datasets/` | 本地视频和训练数据 | 不提交大文件 |
| `outputs/` | 本地 smoke、实验和正式候选产物 | 每次使用新 run ID，不覆盖旧结果 |
| `artifacts/` | 模型、基准和训练产物 | 权重和导出模型不进入普通源码提交 |

## 3. R9 检测追踪核心

以下模块拥有正式检测追踪行为：

| 模块 | 责任 |
| --- | --- |
| `src/tracking/ball_detector.py` | PT 检测协议与 Ultralytics 实现 |
| `src/tracking/onnx_detector.py` | ONNX Runtime 检测与后处理 |
| `src/tracking/temporal_motion.py` | 相机补偿后的连续帧运动证据 |
| `src/tracking/camera_motion.py` | 背景光流和全局相机平移估计 |
| `src/tracking/multi_ball_tracker.py` | 关联、轨迹生命周期、恢复和物理门控 |
| `src/tracking/motion_models.py` | CV/CA Kalman 状态和预测 |
| `src/tracking/person_detector.py` | 低频人体检测 |
| `src/tracking/person_tracking.py` | 人体框延续和 eligible-player 选择 |
| `src/tracking/fast_motion.py` | 受 ROI/handoff 约束的高速运动辅助候选 |
| `src/tracking/ball_pipeline.py` | 单路逐帧编排和诊断 |
| `src/tracking/factory.py` | detector、tracker、pipeline 公共装配边界 |
| `src/tracking/types.py` | JSON 可序列化公共数据契约 |
| `src/tracking/overlay.py` | 只做显示，不改变跟踪状态 |

二维映射阶段不得为了投影或候选判罚修改
`ball_pipeline.py`、`multi_ball_tracker.py`、`motion_models.py`、
`factory.py` 或本地关联规则。若未来明确批准修复 R9，必须先用固定帧/固定检测序列
建立回归测试，再单独评审行为变化。

## 4. 双摄编排与产物

| 模块 | 责任 |
| --- | --- |
| `dual_camera/coordinator.py` | 从左右本地结果中选择唯一全局主球 |
| `dual_camera/runner.py` | 同帧读取离线成对视频，依次运行左右 pipeline，再协调、投影和写出 |
| `dual_camera/rendering.py` | 左右画面和总标题排版 |
| `dual_camera/artifacts.py` | run 目录、partial 文件、原子提升和完整性校验 |
| `dual_camera/projection_replay.py` | 从已完成 R9 run 重新生成投影，不运行模型 |

当前离线双摄是“成对同步、应用层串行”：每帧先读左、再读右，随后依次调用左、
右 pipeline，最后执行协调、投影、渲染和写出。OpenCV/CUDA 底层可能内部并行，
但当前 Python 流程没有为两侧建立线程池或多进程。真实双摄并行采集与单加速器调度
属于未来 `src/runtime/`。

正式双摄 run：

```text
outputs/experiments/dual_camera/<run_id>/
  dual_tracking.mp4
  left_tracking.jsonl
  right_tracking.jsonl
  global_tracking.jsonl
  manifest.json
```

重放 run：

```text
outputs/experiments/court_projection_replay/<run_id>/
  projection_replay.mp4
  global_projection.jsonl
  manifest.json
```

## 5. 二维球场模块

| 模块 | 状态 | 责任 |
| --- | --- | --- |
| `src/court/layout.py` | 纯几何、无状态 | 20×44 ft 坐标、关键点、线段和界内判定 |
| `src/court/calibration.py` | 装配期 | 读取每路人工关键点、估计单应矩阵和质量警告 |
| `src/court/projector.py` | 逐帧只读 | 将 active side 的最终全局球从 image XY 映射到 court XY |
| `src/court/events.py` | 有状态只读解释 | 飞行、落地、界外、二弹和击球候选 |
| `src/court/renderer.py` | 有显示状态 | 空白横版球场、15 帧尾迹、界外标记和事件颜色 |
| `src/court/text.py` | 显示辅助 | 缓存中文状态条；无 CJK 字体时回退英文 |
| `src/court/factory.py` | 公共装配边界 | 从正式配置建立 projector、event interpreter、renderer |
| `src/court/__init__.py` | 公共 API | 只导出稳定的球场接口 |

事件层和 renderer 含跨帧状态。完整离线运行和 JSONL 重放都必须通过
`build_court_projection()` 各自创建新实例，不能跨 run 复用旧状态。

`ground_detection/pickleball_court_detector_handoff.py` 是独立首帧标定工具。
它通过白线增强、线段/交点和半场拓扑生成候选关键点；当前正式配置使用
`ground_detection/full_clean_first_frame_rough.json` 中人工复核后的粗四点结果。
长期接口应是“工具输出可审计标定 JSON -> 人工复核 -> 显式写入正式配置”，而不是
让 `src/court` 在每帧重新检测球场或 import 员工脚本。

## 6. 测试归属

| 测试 | 覆盖范围 |
| --- | --- |
| `test_court_projection.py` | 球场几何、标定、正反/异常投影、兼容字段和 renderer |
| `test_court_events.py` | 落地、界外、二弹、击球候选和状态显示 |
| `test_projection_replay.py` | 无模型重放、来源保护、产物和 manifest |
| `test_ground_detection_completion.py` | 独立标定工具的补全/输出契约 |
| `test_dual_camera_runner.py` | 双摄编排、协调接入和输出 |
| `test_run_manifest.py` | run 隔离、命名和 manifest |
| 其他 `test_*tracking*` / `test_ball_pipeline.py` | R9 核心回归 |

交付前统一运行完整 unittest，而不是只运行新增球场测试。当前已验证基线为
130 项 unittest 通过。

## 7. 文档结构与权威关系

| 文档 | 回答的问题 | 更新触发 |
| --- | --- | --- |
| `README.md` | 新人从哪里开始、正式入口是什么 | 入口、顶层结构或主要能力变化 |
| `docs/PROJECT_STRUCTURE.md` | 代码放在哪里、模块由谁负责 | 新模块、目录或依赖边界变化 |
| `docs/VERSIONS.md` | 当前正式算法身份是什么 | profile/revision/入口变化 |
| `docs/TRACKING_RULES.md` | R9 逐步怎样接受、关联和预测 | R9 规则或阈值变化 |
| `docs/ARCHITECTURE.md` | 数据流和系统边界怎样组织 | 流程、依赖或运行时架构变化 |
| `docs/COURT_2D_MAPPING.md` | 坐标、标定、投影和候选事件如何定义 | court 接口、标定或事件语义变化 |
| `docs/HANDOFF.md` | 此刻完成了什么、风险和下一步是什么 | 每次里程碑或窗口/人员交接 |
| `docs/MAINTENANCE.md` | 如何运行、保存、验证和提交 | 维护流程或产物规范变化 |
| `docs/DEVELOPMENT.md` | 如何建立开发环境 | 依赖、安装或开发命令变化 |
| `docs/DEPLOYMENT.md` | RK3588S 上线需要什么 | 板卡、RKNN、性能或验收变化 |
| `docs/CAMERA_CALIBRATION_TODO.md` | 哪些真实机位参数尚未冻结 | 获得实测标定后 |
| `docs/NEXT_STEPS.md` | 中期路线和优先级 | 阶段目标变化 |
| `CHANGELOG.md` | 用户可见行为历史 | 每次行为或输出契约变化 |

维护时不要在多份文档复制完整规则。结构写在本文，算法细节写在对应权威文档，
`HANDOFF.md` 只记录当前快照和明确的未完成项。
