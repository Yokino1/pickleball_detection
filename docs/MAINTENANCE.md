# 项目维护规范

## 1. 单一职责和依赖方向

依赖方向固定为：

```text
apps / tools
    |
    v
src/tracking/factory.py
    |
    v
src/tracking/*
```

- `apps/` 只解析参数、选择输入和调用核心流程。
- `src/tracking/factory.py` 统一装配 detector、tracker 和 pipeline。
- `src/tracking/dual_camera/` 只处理离线成对视频、全局单球协调、渲染和输出。
- `src/runtime/` 只处理未来板端实时采集、时间戳配对、推理调度和非阻塞输出；
  在真实实现前不放置占位算法。
- 核心模块不能导入 `apps`，活动代码不能导入 `legacy`。
- 算法判断放在 `src/tracking`，不要继续堆到视频读取循环。

现有 `src/tracking/dual_camera/runner.py` 是离线成对视频回归入口，不承担
真实摄像头线程、RKNN 上下文或机器人控制职责。

## 2. 数据和输出目录

```text
data/
  sideview_raw/          原始本地视频
  reference/             只读参考资料
  derived/court_halves/  由原视频生成的半场输入

outputs/
  production/            经过验收、需要保留的正式结果
  experiments/           参数或算法实验
  smoke/                 可随时清理的短片检查
  previews/              切割线、ROI、布局等静态预览

artifacts/
  models/                本地模型及模型卡
  benchmarks/            指标和板端性能报告
  training/              训练过程

experiments/             可提交 Git 的实验元数据，不存大文件
```

现有派生半场视频已迁移到 `data/derived/court_halves`。历史双摄 Phase 1、
人体接触和 Physics 输出已归档到 `outputs/experiments/` 对应目录。被播放器
占用的旧目录应在关闭播放器后再完成迁移，不强制结束用户进程。

## 3. Run ID 和输出保护

推荐 run ID：

```text
YYYYMMDD_<dataset>_<profile>-r<revision>_<purpose>
```

示例：

```text
20260727_full_clean_pickleball-r9_observation-continuity
```

双摄入口默认拒绝覆盖同名正式结果：

- 新实验使用新 run ID；
- 已完成结果可用 `--skip-existing` 跳过；
- 确认要替换时显式使用 `--overwrite`。

运行过程中写入 `.partial.*`。正常完成并通过帧数、FPS、尺寸和 JSONL
行数校验后才原子改名。异常退出会清理可控的临时文件；进程被强杀时残留的
`.partial.*` 可根据同名 manifest 判断并清理。

## 4. 自动运行记录

双摄每个 run 自动生成：

```text
<run_id>_manifest.json
```

记录内容包括：

- Git commit、分支和 dirty 状态；
- 配置路径、profile/revision 和 SHA-256；
- 左右输入路径、FPS、分辨率和帧数；
- 运行参数和实验备注；
- 输出文件及大小；
- 总帧数、运行速度、handoff、ROI retry 和 fast-motion 摘要。

正式结论必须能从实验登记、manifest、JSONL 和代码提交相互追溯。

## 5. 版本管理

项目同时维护三种版本号：

1. `pyproject.toml`：软件包/发布版本。
2. 配置 `profile.name + profile.revision`：算法配置版本。
3. Git commit/tag：精确代码版本。

`schema_version` 只表示文件结构兼容性，不等于算法版本。

行为变化流程：

1. 增加或更新测试；
2. 修改核心代码；
3. 修订配置 profile（确有参数行为变化时）；
4. 更新 `CHANGELOG.md`；
5. 跑固定 smoke/回归；
6. 保存 manifest 和实验结论；
7. 达到验收门槛后再提升正式 profile 或发布版本。

当前正式算法的稳定身份是：

```text
profile.name = pickleball_tracking
profile.revision = 9
profile.status = maintained
```

不能通过复制 `tracking_v9_final.yaml` 的方式发布新版本。行为或正式默认参数发生
变化时，先更新测试和 `CHANGELOG.md`，再明确决定是否递增
`profile.revision`。结构不兼容才递增 `schema_version`。

## 6. 文档权威来源与更新触发

同一事实只允许有一个主要权威来源，其他文档引用它：

| 内容 | 权威来源 | 必须同步更新的时机 |
| --- | --- | --- |
| 当前正式/历史版本 | `docs/VERSIONS.md`、`configs/tracking.yaml` | profile、revision、状态或默认入口变化 |
| 精确追踪顺序和阈值 | `docs/TRACKING_RULES.md`、`configs/tracking.yaml` | pipeline 顺序、门控或参数变化 |
| 模块边界 | `docs/ARCHITECTURE.md`、ADR | 新模块、依赖方向或运行时职责变化 |
| RK3588S、量化和验收 | `docs/DEPLOYMENT.md` | 工具链、模型格式、性能预算或验收结论变化 |
| 未标定参数 | `docs/CAMERA_CALIBRATION_TODO.md` | 获得真实相机/载板测量后逐项关闭 |
| 当前可交接状态 | `docs/HANDOFF.md` | 每次里程碑交付、人员交接或板端阶段变化 |
| 历史行为变化 | `CHANGELOG.md` | 每次可见行为、输出契约或版本关系变化 |

阈值不能只改 YAML 不改 `TRACKING_RULES.md`；命令不能只改 CLI 不改 README；
输出文件名不能只改代码不改 `outputs/README.md` 和交接文档。审查时以代码和 YAML
实际行为为准，发现冲突必须修正文档或明确登记为已知缺口。

## 7. Smoke 和实验生命周期

Smoke 只回答“程序能否完整运行、输出结构是否正确”，不能证明算法有效。

- 默认放在 `outputs/smoke/<run_id>/`；
- 建议 30～100 帧；
- 验证完成即删除，或最多保留七天；
- 若发现算法问题，把对应短片提升为固定回归片段，而不是长期保留整个 smoke 输出。

先预览七天前的 smoke：

```cmd
python tools\cleanup_smoke_outputs.py --older-than-days 7
```

确认列表后再显式删除：

```cmd
python tools\cleanup_smoke_outputs.py --older-than-days 7 --apply
```

实验结果放在 `outputs/experiments/`。只有影响决策的实验才在 `experiments/`
登记。正式候选放在 `outputs/production/`，并保留指标报告。

## 8. Git 暂存与资产边界

当前仓库经常包含用户尚未提交的历史改动，禁止用清理或回退命令处理不属于当前
任务的文件。每次工作开始先运行 `git status`，提交前再次逐文件复查。

- 禁止使用 `git add .`；
- 用户要求的“CMD 代码”指可直接粘贴到 Conda CMD 的命令块；活动目录不创建
  `run_*.cmd` 运行文件；
- 只显式暂存本次确认的源码、配置、测试、文档和小型实验元数据；
- 暂存列表不得包含 `mp4`、`pt`、`onnx`、`engine`、`rknn`、`docx`、数据集、
  `data/`、`outputs/`、缓存或训练中间结果；
- 根目录业务 DOCX 只作背景资料，不自动提交；
- 不覆盖、回退或清理已有未提交改动；
- 提交前使用 `git diff --cached --name-status` 和
  `git diff --cached --check` 审查暂存内容；
- 未经明确要求，不 commit、push 或创建 PR。

## 9. 量化与模型交接

量化候选不能只交一个模型文件。每个候选至少同时交付：

- 源 PT、导出 ONNX、RKNN 候选各自的 SHA-256；
- source Git revision、正式 profile/revision 和完整配置 SHA-256；
- 精确 Python、Ultralytics、ONNX Runtime、RKNN Toolkit/runtime 版本；
- 输入尺寸、颜色顺序、归一化、NMS、输出张量和解码约定；
- 校准数据 revision、数量、抽样方法和不可提交资产位置；
- FP32/FP16/INT8 同集检测报告及细分场景召回；
- 固定连续视频的追踪、handoff 和 observation/prediction 比例报告；
- RK3588S 完整板卡、功耗、散热、延迟、内存、队列和 30 分钟 soak 报告；
- 已知失败场景、回退模型和回退配置。

ONNX INT8、RKNN INT8 和桌面 PT 是三个不同运行产物，不能因为都写着 INT8 或
使用同一权重来源就视为等价。

## 10. 每次交付前检查

```cmd
D:\anacondaa\envs\torch-cu128\python.exe tools\check_project_refs.py
D:\anacondaa\envs\torch-cu128\python.exe -m unittest discover -s tests -v
D:\anacondaa\envs\torch-cu128\python.exe -m compileall -q apps src tools tests
git diff --check
git status --short
```

还需确认：

- 没有把视频、数据集、模型、输出或缓存加入 Git；
- `configs/tracking.yaml` 仍是唯一默认正式入口，历史配置没有被活动代码引用；
- JSONL 结构变化已经记录并有兼容策略；
- 实验结论不是只凭一段可视化视频得出；
- 板端测试记录 RK3588S 完整板卡/载板、两路 60 FPS 时间戳偏差、丢帧、
  队列深度和热稳定性。
- `docs/HANDOFF.md` 的已实现、未实现和已知风险与代码一致。
