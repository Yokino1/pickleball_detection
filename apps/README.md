# Applications

`apps/` 只保留用户入口，不实现算法规则。

## 当前正式入口：成对双半场

```cmd
D:\anacondaa\envs\torch-cu128\python.exe apps\track_dual_halves.py --config configs\tracking.yaml --pair RUN_ID "LEFT_VIDEO_PATH" "RIGHT_VIDEO_PATH"
```

运行命令应直接粘贴到 Conda CMD，不为每次运行生成 `.cmd` 文件。

双摄入口负责参数和运行策略；同步处理、全局单球协调、handoff、渲染和产物校验
位于 `src/tracking/dual_camera/`。默认拒绝覆盖同名结果，每次完成后生成 manifest。

每个 run 自动建立独立目录：

```text
<output-dir>/<run_id>/
  dual_tracking.mp4
  left_tracking.jsonl
  right_tracking.jsonl
  global_tracking.jsonl
  manifest.json
```

该入口面向已经同步且元数据一致的离线视频文件。未来两路 60 FPS 真实摄像头
入口将使用 `src/runtime/` 的采集、时间戳配对和推理调度，不把实时设备逻辑
加入此离线循环。

## 二维映射只读重放入口

当 R9 输入视频和检测追踪结果已经固定，只修改标定、二维投影、事件阈值或显示
样式时，使用：

```cmd
D:\anacondaa\envs\torch-cu128\python.exe apps\replay_court_projection.py --config configs\tracking.yaml --source-run "SOURCE_RUN_DIR" --run-id NEW_REPLAY_RUN_ID
```

该入口读取已完成 run 的 `dual_tracking.mp4`、左右/全局 JSONL 和 manifest，
不会构建或运行 detector、person detector、tracker 或 coordinator。默认裁掉旧
右侧面板并生成新的完整拼接视频；加 `--projection-only` 后只读三份 JSONL，
不解码旧 MP4，只输出纯二维球场视频。

重放结果自动写入：

```text
outputs/experiments/court_projection_replay/<run_id>/
  projection_replay.mp4
  global_projection.jsonl
  manifest.json
```

来源 run 保持不变；重放也必须使用新 run ID，不覆盖旧结果。

新生成的右侧面板顶部会显示中文球状态；事件 JSON 同时记录机器状态、中文文本、
犯规候选和原因。`--projection-only` 与完整拼接模式复用同一事件状态机。

重放入口是开发期缩短实验周期的工具，不是机器人或 RK3588S 的生产入口。最终
板端链路必须在同一在线数据流中执行采集/配对、检测、追踪、全局单球选择、对应
相机的固定标定投影、事件解释和输出；预生成 MP4/JSONL 不能成为生产依赖。

## 单视频/文件夹诊断入口

`apps/track_video.py` 仅用于定位某一侧的模型漏检、时序过滤、关联和物理门控
问题。它复用正式配置和 pipeline，但不运行双侧全局协调，因此不是当前正式产品
运行入口。

公共 detector/tracker/pipeline 装配位于 `src/tracking/factory.py`。任何 app 或 tool
都不应从另一个 app 导入装配函数。活动入口和 `src/tracking` 不得导入 `legacy`。
