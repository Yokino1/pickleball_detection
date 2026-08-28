# Pickleball Detection and Tracking

当前项目目标是优先提高单个 Pickleball 的检测和追踪效果。主流程对视频逐帧检测球，
维护一个主要输出 ID，在短时漏检时做保守预测，并输出带轨迹 MP4 和逐帧 JSONL。
R9 检测追踪仍是主流程；固定机位球场二维映射是只读消费最终全局球的附加输出，
不参与检测、关联、预测、物理门控或全局球选择。三维坐标不属于当前实现范围。

## 当前正式版本

项目只维护一套活动正式算法配置：

| 版本 | 配置 | 运行脚本 | 用途 |
| --- | --- | --- | --- |
| Pickleball Tracking revision 9 | `configs/tracking.yaml` | `apps/track_dual_halves.py` | 两个半场视频成对处理、裁切尺度自动推导、连续主模型观测纠偏、受限落地反弹恢复、CA Kalman、人体接触和全局单球协调 |

原 mainline、temporal 和 physics 三套桌面方案已经停止维护，配置与 CMD
保存在 `legacy/ball_tracking_handoff/`，只用于历史回归。`tracking_edge.yaml`
仍是板端部署研究配置，不代表另一套活动算法版本。

详细版本关系见 [版本说明](docs/VERSIONS.md)。

## 正式双半场运行

以下代码直接粘贴到 Conda CMD 窗口。每次完整测试必须使用新的 run ID：

```bat
call "D:\anacondaa\Scripts\activate.bat" "D:\anacondaa\envs\torch-cu128"
cd /d D:\ball\ball_tracking_handoff\ball_tracking_handoff

python apps\track_dual_halves.py ^
  --config configs\tracking.yaml ^
  --pair 20260727_full_clean_pickleball-r9_observation-continuity01 ^
  "data\derived\court_halves\全_干净背景_left.mp4" ^
  "data\derived\court_halves\全_干净背景_right.mp4" ^
  --output-dir outputs\experiments\dual_camera
```

当前正式流程同时读取左右两个半场视频，分别运行本地 pipeline，再由全局协调器
只输出一个主球。`apps/track_video.py` 仅保留为单路问题定位工具，不代表当前
正式产品运行方式。

每个双摄 run 自动写入
`outputs/experiments/dual_camera/<run_id>/`，目录内固定包含视频、三份 JSONL
和 manifest，不会再把不同 run 的产物平铺混放。

当检测追踪结果已经固定，只调整二维投影或事件规则时，使用
`apps/replay_court_projection.py` 从已有 run 重放；该入口不运行任何模型，
不会修改来源 run。`--projection-only` 可跳过旧 MP4 解码，只生成纯二维面板。
该入口只用于桌面调试和离线回归，不代表最终板端流程。最终 RK3588S 运行时必须
对实时双摄帧边检测、边追踪、边完成全局单球选择和二维投影，不能依赖预生成 JSONL。

当前默认只输出一个评分最高的运动球轨迹，轨迹、检测圈、预测圈和标签统一使用荧光绿色。

## 项目结构

```text
apps/                 单视频与同步双摄 CLI 入口
configs/              当前正式算法和板端研究 profile
src/tracking/         检测、关联、预测和公共装配
src/tracking/dual_camera/  离线双视频处理、协调、渲染和产物管理
src/court/            标准球场几何、固定机位标定、二维投影和纯框架渲染
src/runtime/          已实现时间戳帧/有界配对契约；采集、RKNN 调度和控制待接入
scripts/              项目检查和维护脚本，不保存正式运行批处理
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

- [项目结构与模块责任](docs/PROJECT_STRUCTURE.md)
- [当前与历史版本说明](docs/VERSIONS.md)
- [追踪规则与参数](docs/TRACKING_RULES.md)
- [物理约束方案](docs/Physics_EKF_Pickleball_Tracking.md)
- [人体接触门控与透视补偿方案](docs/PERSON_CONTACT_PERSPECTIVE_TRACKING.md)
- [当前架构](docs/ARCHITECTURE.md)
- [R9 球场二维映射约定](docs/COURT_2D_MAPPING.md)
- [下一步工作](docs/NEXT_STEPS.md)
- [训练流程](docs/TRAINING.md)
- [板端部署](docs/DEPLOYMENT.md)
- [真实双摄待标定参数](docs/CAMERA_CALIBRATION_TODO.md)
- [当前交接状态、已知风险与量化前置条件](docs/HANDOFF.md)
- [开发维护](docs/DEVELOPMENT.md)
- [长期维护与输出规范](docs/MAINTENANCE.md)
- [双摄 60 FPS 板端架构决策](docs/decisions/0002-dual-camera-60fps-edge-runtime.md)
- [变更记录](CHANGELOG.md)

同步双半场实验使用 `apps/track_dual_halves.py`。公共组件装配位于
`src/tracking/factory.py`，每个双摄 run 自动生成 manifest。未来输出按
`production / experiments / smoke / previews` 分类，具体见
[outputs/README.md](outputs/README.md)。

最终机器人目标是两个独立半场摄像头、每路 60 FPS。当前双视频入口是离线
回归工具，不是实时采集实现；板卡 SoC 已确认是 RK3588S，具体载板、内存、
相机接口、散热和功耗模式仍待记录与实测。

## 当前边界

- 检测模型持续漏检时，追踪器只能补极短时间，不能凭空恢复球。
- 当前物理约束使用二维像素空间加速度，不使用真实 `9.81 m/s^2` 重力。
- “模型观测优先”从本地 tracker 已接受的模型观测开始；原始检测仍可能被
  时序过滤或关联门控拒绝，必须结合 JSONL 诊断分类。
- 当前离线双摄允许另一侧主模型观测直接抢占旧侧预测；生产级跨摄状态机尚未完成。
- 二维投影只使用协调器最终选中的全局球；左右相机各自标定到同一标准球场坐标系，
  不能直接比较像素坐标。空中球的二维点只是视线投到地面平面的近似，不能视作三维位置、
  高度或真实落点。框外坐标会保留并标记为越界，不会被投影模块裁剪。
- 反弹、界外反弹、二弹和击拍颜色均是只读候选事件：绿=飞行，黄=场内反弹候选，
  红=场外反弹候选，紫=eligible-player 邻域和运动突变支持的击拍候选。事件层复用
  现有 R9 输出和球员框，不启动新检测模型；当前仍没有直接球拍检测，颜色不能作为
  正式裁判判罚。
- 右侧二维面板顶部显示中文状态：`飞行`、`短时预测`、`消失在屏幕`、`落地`、
  `二弹`、`出界`、`击球` 或 `投影不可用`。整个 20×44 ft 白色外框含边线；
  只有落地接触点在外框之外才产生出界候选。同侧在未击回前的第二次落地产生二弹
  犯规候选，不使用固定的比赛时间窗。
- 击球候选还允许使用人体框外的有限持拍伸展区：同侧换 track ID 时必须在短观测
  间隔内发生显著速度反转；跨相机时必须是伸展区内的高速首次观测。两者都受最大
  合理速度限制，只补充事件展示，不改变 R9 的检测、关联、预测或全局选球。
- 真实重力、空气阻力和三维轨迹需要相机标定及双目或其他深度来源。
- 主流程不得导入 `legacy`。
- 视频、数据集、权重、导出模型和输出结果不得随意提交 Git。
