# Applications

`apps/` 只保留用户入口，不实现算法规则。

## 单视频/文件夹

```cmd
python apps\track_video.py --config configs\tracking.yaml --input <video-or-dir>
```

支持的正式桌面 profile：

- `configs/tracking.yaml`
- `configs/tracking_temporal.yaml`
- `configs/tracking_physics.yaml`

## 同步双半场

```cmd
python apps\track_dual_halves.py --config configs\tracking_person_contact.yaml --pair <run_id> <left.mp4> <right.mp4>
```

双摄入口负责参数和运行策略；同步处理、全局单球协调、handoff、渲染和产物校验
位于 `src/tracking/dual_camera/`。默认拒绝覆盖同名结果，每次完成后生成 manifest。

该入口面向已经同步且元数据一致的离线视频文件。未来两路 60 FPS 真实摄像头
入口将使用 `src/runtime/` 的采集、时间戳配对和推理调度，不把实时设备逻辑
加入此离线循环。

公共 detector/tracker/pipeline 装配位于 `src/tracking/factory.py`。任何 app 或 tool
都不应从另一个 app 导入装配函数。活动入口和 `src/tracking` 不得导入 `legacy`。
