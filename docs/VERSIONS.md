# 检测追踪版本说明

## 当前正式版本

当前唯一活动正式版本：

```text
配置：configs/tracking.yaml
Profile：pickleball_tracking
Revision：9
状态：maintained
正式入口：apps/track_dual_halves.py
辅助单路诊断：apps/track_video.py
运行方式：命令直接粘贴到 Conda CMD，不生成运行批处理
```

revision 9 的核心原则仍是：

> 可靠模型观测优先；只有当前没有可靠模型观测时，预测和辅助候选才补位。

这里的“可靠模型观测”不是原始 YOLO 框，而是已经通过检测阈值、去重、当前
时序过滤和本地 tracker 关联/确认的 YOLO 或 ONNX observed 轨迹。原始模型框
仍可能在进入该优先级之前被过滤或拒绝，具体诊断顺序见
[`TRACKING_RULES.md`](TRACKING_RULES.md)。

当前正式版本包含：

- 球 YOLO/PT 或 ONNX Runtime 检测；
- 成对裁切视频按左右总宽度自动恢复原视频物理尺度；
- 检测框去重和相机全局平移补偿；
- 相邻帧局部运动证据；
- 常加速度 Kalman；
- 基于真实时间戳的速度、方向、NIS 和加速度门控；
- 连续邻近且原始方向一致的主模型观测可纠正滞后的 Kalman 状态；
- 仅针对邻近落地点主模型观测的受限下落到上升反弹恢复；
- 高速球 60–120 ms 自适应短预测；
- 运动确认和静止轨迹抑制；
- 每 5 帧人体检测与中间帧人体框延续；
- eligible player 筛选和人体接触门控；
- 左右独立本地 tracker；
- 模型观测优先的全局单球协调；
- 球网入口 ROI、受限 `fast_motion` 和严格辅助 handoff；
- 球网出口越界预测抑制和跨侧尾迹清理；
- 本地 ID 变化或显示位置物理不连续时切断尾迹，但保留当前模型观测；
- MP4、左右/全局 JSONL 与运行 manifest。

正式入口对左右半场各运行一套本地 pipeline，再执行全局单球协调。单视频入口
只用于拆分问题和检查某一侧 JSONL；它会忽略双摄协调步骤，不作为当前正式版本
的独立运行形态。

## 当前判定优先级

```text
当前活动侧可靠 YOLO/ONNX observed
    >
另一侧可靠 YOLO/ONNX observed
    >
当前侧短时 predicted
    >
严格 handoff 内确认的 fast_motion 辅助候选
    >
temporarily_lost
```

物理门控负责判断检测能否继承旧 ID；它不能让已经被本地 tracker 接受的模型
观测输给旧预测。辅助运动候选不能冒充模型观测。

当前离线双摄允许另一侧确认的主模型观测在旧侧只有预测或已经丢失时直接抢占，
无需 handoff 已预警。这是 revision 9 延续的召回优先行为；同侧确认的主模型
观测也会立即替代旧 prediction。真实机器人上线前仍需
将跨摄切换收敛为同时满足观测优先与时空约束的生产状态机。

## 历史桌面版本

以下三套方案已停止维护，不再作为正式入口或默认配置：

| 历史版本 | 历史配置 | 说明 |
| --- | --- | --- |
| mainline r1 | `legacy/ball_tracking_handoff/configs/maintained_history/tracking_mainline_r1.yaml` | 单帧 YOLO + 常速度 Kalman |
| temporal r1 | `legacy/ball_tracking_handoff/configs/maintained_history/tracking_temporal_r1.yaml` | mainline + 相邻帧运动证据 |
| physics r1 | `legacy/ball_tracking_handoff/configs/maintained_history/tracking_physics_r1.yaml` | temporal + 常加速度和物理门控 |

对应历史 CMD 位于：

```text
legacy/ball_tracking_handoff/scripts/maintained_history/
```

退役的人体接触/双摄实验最终快照另存为
`legacy/ball_tracking_handoff/configs/maintained_history/tracking_person_contact_r4.yaml`。
它是实验历史，不属于上表三套旧正式版本，也不等价于 revision 9。

活动 app、`src/tracking` 和默认脚本不得依赖这些历史文件。需要复查旧算法时，
必须显式运行历史配置，并把结果写入新的历史回归目录。

## 板端配置

`configs/tracking_edge.yaml` 是单路 ONNX Runtime 可移植和量化研究配置，不是
第二套正式算法版本，也不是已验收的 RK3588S/RKNN 发布包。最终板端配置应在
当前 revision 9 行为基础上完成转换、量化和性能验收。

## 尚未冻结但不阻止转正的参数

真实左右摄像头尺度、比赛区、观众区、球网出口、handoff ROI、同步容差和板端
推理策略记录在
[`CAMERA_CALIBRATION_TODO.md`](CAMERA_CALIBRATION_TODO.md)。

这些参数属于安装和部署标定，不再用于把当前算法降级为“实验版本”。
