# Applications

`apps/` 只保留用户入口，不实现算法规则。

## 当前正式入口：成对双半场

```cmd
D:\anacondaa\envs\torch-cu128\python.exe apps\track_dual_halves.py --config configs\tracking.yaml --pair RUN_ID "LEFT_VIDEO_PATH" "RIGHT_VIDEO_PATH"
```

运行命令应直接粘贴到 Conda CMD，不为每次运行生成 `.cmd` 文件。

双摄入口负责参数和运行策略；同步处理、全局单球协调、handoff、渲染和产物校验
位于 `src/tracking/dual_camera/`。默认拒绝覆盖同名结果，每次完成后生成 manifest。

每个 run 的正式文件名为：

```text
<run_id>_dual_tracking.mp4
<run_id>_left_tracking.jsonl
<run_id>_right_tracking.jsonl
<run_id>_global_tracking.jsonl
<run_id>_manifest.json
```

该入口面向已经同步且元数据一致的离线视频文件。未来两路 60 FPS 真实摄像头
入口将使用 `src/runtime/` 的采集、时间戳配对和推理调度，不把实时设备逻辑
加入此离线循环。

## 单视频/文件夹诊断入口

`apps/track_video.py` 仅用于定位某一侧的模型漏检、时序过滤、关联和物理门控
问题。它复用正式配置和 pipeline，但不运行双侧全局协调，因此不是当前正式产品
运行入口。

公共 detector/tracker/pipeline 装配位于 `src/tracking/factory.py`。任何 app 或 tool
都不应从另一个 app 导入装配函数。活动入口和 `src/tracking` 不得导入 `legacy`。
