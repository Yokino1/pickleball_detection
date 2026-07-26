# Pickleball Detection and Tracking

当前项目目标是优先提高单个 Pickleball 的检测和追踪效果。主流程对视频逐帧检测球，
维护一个主要输出 ID，在短时漏检时做保守预测，并输出带轨迹 MP4 和逐帧 JSONL。
球场二维投影和三维坐标不属于当前主流程。

## 保留版本

项目只维护三套桌面检测追踪配置：

| 版本 | 配置 | 运行脚本 | 用途 |
| --- | --- | --- | --- |
| 主线 | `configs/tracking.yaml` | `scripts/run_mainline_tracking.cmd` | 单帧 YOLO + CV Kalman + 相机补偿和关联约束 |
| 连续帧 | `configs/tracking_temporal.yaml` | `scripts/run_temporal_tracking.cmd` | 主线 + 轻量相邻帧运动证据过滤 |
| 物理约束 | `configs/tracking_physics.yaml` | `scripts/run_physics_tracking.cmd` | 连续帧 + CA Kalman + NIS/加速度约束 |

三套版本共用 `apps/track_video.py` 和 `src/tracking`，不存在三份相互复制的追踪代码。
`configs/tracking_edge.yaml` 仅作为后续板端 ONNX 部署配置保留，不属于桌面 A/B 三版本。

另有一条不替代上述三版的实验分支：

| 实验分支 | 配置 | 运行脚本 | 用途 |
| --- | --- | --- | --- |
| 人体接触门控 | `configs/tracking_person_contact.yaml` | `scripts/run_person_contact_tracking.cmd` | Physics + 每 5 帧人体检测 + 比赛人员筛选 + 击球恢复接触门控 |

详细差异见 [版本说明](docs/VERSIONS.md)。

## 直接运行

在 Conda CMD 中进入项目目录：

```bat
cd /d D:\ball\ball_tracking_handoff\ball_tracking_handoff
```

然后选择一版：

```bat
scripts\run_mainline_tracking.cmd
scripts\run_temporal_tracking.cmd
scripts\run_physics_tracking.cmd
scripts\run_person_contact_tracking.cmd
```

这些脚本默认处理 `data\sideview_raw` 中现有的全部视频，分别输出到：

```text
outputs\experiments\desktop_ab\mainline
outputs\experiments\desktop_ab\temporal
outputs\experiments\desktop_ab\physics
outputs\experiments\person_contact\current
```

每个视频生成：

```text
<video>_tracked.mp4
<video>_tracking.jsonl
```

当前默认只输出一个评分最高的运动球轨迹，轨迹、检测圈、预测圈和标签统一使用荧光绿色。

## 项目结构

```text
apps/                 单视频与同步双摄 CLI 入口
configs/              正式、实验和板端 profile
src/tracking/         检测、关联、预测和公共装配
src/tracking/dual_camera/  双摄同步、协调、渲染和产物管理
src/runtime/          未来机器人实时采集、同步和推理调度边界
scripts/              三版本 CMD 与项目检查脚本
tests/                核心算法回归测试
docs/                 版本、规则、架构、训练和部署文档
experiments/          可提交的实验元数据，不保存大文件
data/                 本地测试视频，不提交 Git
datasets/             本地训练数据，不提交大文件
artifacts/models/     本地模型，不提交权重
outputs/              本地运行结果，不提交 Git
legacy/               原始交接代码参考，不被主流程导入
```

## 文档入口

- [三个版本说明](docs/VERSIONS.md)
- [追踪规则与参数](docs/TRACKING_RULES.md)
- [物理约束方案](docs/Physics_EKF_Pickleball_Tracking.md)
- [人体接触门控与透视补偿方案](docs/PERSON_CONTACT_PERSPECTIVE_TRACKING.md)
- [当前架构](docs/ARCHITECTURE.md)
- [下一步工作](docs/NEXT_STEPS.md)
- [训练流程](docs/TRAINING.md)
- [板端部署](docs/DEPLOYMENT.md)
- [开发维护](docs/DEVELOPMENT.md)
- [长期维护与输出规范](docs/MAINTENANCE.md)
- [双摄 60 FPS 板端架构决策](docs/decisions/0002-dual-camera-60fps-edge-runtime.md)
- [变更记录](CHANGELOG.md)

同步双半场实验使用 `apps/track_dual_halves.py`。公共组件装配位于
`src/tracking/factory.py`，每个双摄 run 自动生成 manifest。未来输出按
`production / experiments / smoke / previews` 分类，具体见
[outputs/README.md](outputs/README.md)。

最终机器人目标是两个独立半场摄像头、每路 60 FPS。当前双视频入口是离线
回归工具，不是实时采集实现；候选板卡按 RK3588S 级硬件规划，准确 SKU 待确认。

## 当前边界

- 检测模型持续漏检时，追踪器只能补极短时间，不能凭空恢复球。
- 当前 physics 版本使用二维像素空间加速度，不使用真实 `9.81 m/s^2` 重力。
- 真实重力、空气阻力和三维轨迹需要相机标定及双目或其他深度来源。
- 主流程不得导入 `legacy`。
- 视频、数据集、权重、导出模型和输出结果不得随意提交 Git。
