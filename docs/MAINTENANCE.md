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
- `src/tracking/dual_camera/` 只处理双路同步、全局单球协调、渲染和输出。
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
20260725_full_clean_person-contact-r2_phase23
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

## 6. Smoke 和实验生命周期

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

## 7. 每次交付前检查

```cmd
D:\anacondaa\envs\torch-cu128\python.exe tools\check_project_refs.py
D:\anacondaa\envs\torch-cu128\python.exe -m unittest discover -s tests -v
git diff --check
git status --short
```

还需确认：

- 没有把视频、数据集、模型、输出或缓存加入 Git；
- 三套正式配置的默认输出没有被实验分支改变；
- JSONL 结构变化已经记录并有兼容策略；
- 实验结论不是只凭一段可视化视频得出。
- 板端测试记录准确 SKU、两路 60 FPS 时间戳偏差、丢帧、队列深度和热稳定性。
