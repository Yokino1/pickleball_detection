# 检测追踪版本说明

## 共同能力

三个版本使用同一个入口和同一套核心代码：

- Ultralytics PT 或 ONNX Runtime 球检测；
- 检测框去重；
- 相机全局平移补偿；
- 基于实际时间的 `px/s` 速度；
- 按参考分辨率缩放门控参数；
- 单个主要输出轨迹；
- 静止误检抑制；
- 方向突变和 impact recovery；
- 普通球最长 120ms、高速球最短 60ms 的短时预测；
- 荧光绿色实线轨迹；
- MP4 和 JSONL 输出。

## 主线

配置：`configs/tracking.yaml`

脚本：`scripts/run_mainline_tracking.cmd`

输出：`outputs/experiments/desktop_ab/mainline`

主线使用单帧 YOLO 和四状态常速度 Kalman：

```text
[x, y, vx, vy]
```

它不启用连续帧运动过滤，也不估计加速度。该版本用于判断仅依靠检测、相机补偿和基础关联约束时的效果。

## 轻量连续帧

配置：`configs/tracking_temporal.yaml`

脚本：`scripts/run_temporal_tracking.cmd`

输出：`outputs/experiments/desktop_ab/temporal`

该版本在主线前增加 `TemporalMotionFilter`：

1. 将上一帧按相机运动估计对齐到当前帧。
2. 计算低分辨率灰度帧差。
3. 检查每个 YOLO 候选附近的局部运动比例。
4. 固定灯光、标牌和 overlay 没有运动证据时不送入追踪器。
5. 第一帧或全画面剧烈变化时失效开放，避免整帧漏检。

该方案不改变检测模型输入，也不能找回 YOLO 完全没有检测到的高速球。

## Physics Tracking

配置：`configs/tracking_physics.yaml`

脚本：`scripts/run_physics_tracking.cmd`

输出：`outputs/experiments/desktop_ab/physics`

该版本在连续帧方案基础上启用六状态常加速度 Kalman：

```text
[x, y, vx, vy, ax, ay]
```

并增加：

- 连续可靠观测达到门槛后才启用加速度；
- 图像空间加速度裁剪和漏检衰减；
- NIS 创新门控；
- 观测加速度上限；
- impact recovery 后重置旧加速度；
- 预测时长随速度连续缩短。

这里的加速度单位是 `px/s^2`，属于图像空间软约束，不代表真实重力。

## 板端配置

`configs/tracking_edge.yaml` 使用 ONNX Runtime 和较小的分析尺寸，保留用于量化和板端部署研究。
它不是第四套桌面算法版本。板端验收必须使用最终硬件、分辨率、FPS 和量化模型重新测试。

当前产品目标已确定为两个物理摄像头、每路 60 FPS。候选板卡按 RK3588S 级硬件
规划，准确 SKU 尚未确认。现有 edge 配置只是单路 ONNX 可移植基线，不代表双路
RKNN 已达到 60 FPS。实时运行时决策见
[`0002-dual-camera-60fps-edge-runtime.md`](decisions/0002-dual-camera-60fps-edge-runtime.md)。

## 人体接触门控实验分支

该分支已经有第一版可运行实现，但仍不是第四套正式版本。完整方案和限制见
[`PERSON_CONTACT_PERSPECTIVE_TRACKING.md`](PERSON_CONTACT_PERSPECTIVE_TRACKING.md)。

配置：`configs/tracking_person_contact.yaml`

脚本：`scripts/run_person_contact_tracking.cmd`

输出：`outputs/experiments/person_contact/current`

第一版在 Physics Tracking 上增加：

- 桌面和板端统一每 5 帧运行一次轻量人体模型；
- 中间帧用 `PersonBoxTracker` 延续人体框；
- 按脚点比赛区域、观众排除区、持续帧数和人数上限筛选比赛人员；
- 只有球的观测线段接触比赛人员扩展框时，才允许大角度 `impact_recovery`；
- JSONL 记录全部人体框、是否入选、接触区数量和接触门控拒绝次数。

当前接触区仍是人体框尺度近似，不是球拍接触识别。完成固定样本验证前，不修改三个正式版本的输出。

## A/B 对比规则

对比三个版本时必须使用：

- 相同模型；
- 相同输入视频；
- 相同检测置信度；
- 相同输出轨迹长度；
- 不覆盖旧结果的独立输出目录。

重点比较：

- 真实球检测/轨迹召回率；
- 固定背景误检进入主轨迹的次数；
- ID 切换次数；
- 非击球时异常折线数量；
- 击球后恢复原 ID 的成功率；
- 预测帧比例和 1 至 3 帧预测误差；
- 每帧耗时。

## 双摄 Phase 2/3 实验

双摄实验仍属于 `person_contact_dual_camera` 实验 profile，不是第四套正式桌面版本。

入口与核心模块：

```text
apps/track_dual_halves.py
src/tracking/dual_camera/
src/tracking/fast_motion.py
```

当前增加：

- 两路视频按相同帧号和时间戳同步处理；
- 两侧保持独立本地候选，只输出一个全局球 ID；
- 球靠近球网并朝球网高速运动时，对接收侧开启短时入场 ROI；
- 接收侧全图 YOLO 漏检后追加一次 ROI YOLO；
- 只有在 handoff ROI 或已有高速预测 ROI 内，连续高速小运动块才能辅助漏检；
- `dual_camera_streams` 为左右流保留真实像素密度，避免裁切宽度误缩物理门控；
- 人体数据继续参与接触门控并写入 JSONL，默认结果视频不显示人体框；
- 最终全局 ID 逐帧执行速度连续性门控，本地 ID 切换不能产生超速长线；
- 人体接触恢复限制在 120 ms 内，且长线仅穿过人体框不再算接触；
- 每次运行自动生成代码、配置、输入和结果 manifest。

该实验不比较两台摄像头的像素坐标，也不宣称完成双目标定或三维定位。
