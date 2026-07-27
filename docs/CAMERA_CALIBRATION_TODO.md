# 真实双摄待标定参数

当前正式算法配置为 `configs/tracking.yaml` revision 9。以下参数尚未冻结，
但不阻止算法版本转正；它们属于真实机器人安装和板端验收阶段的标定项。

状态定义：

- `confirmed`：有设备标识或可复现测量证据；
- `pending`：尚未获得；
- `provisional`：仅适用于现有离线裁切数据，禁止带入真实双摄。

| 项目 | 当前状态 | 当前值/说明 |
| --- | --- | --- |
| SoC | confirmed | RK3588S |
| 完整板卡/载板、内存、系统、散热 | pending | 不创建假定可用的 RKNN release profile |
| 两路真实相机及硬件时间戳 | pending | 目标为每路 60 FPS |
| 离线裁切尺度 | confirmed rule | `paired_crop_total_width`：左右宽度之和除以 1280 |
| 真实左右尺度、比赛区、观众区 | pending | 必须逐路标定 |
| 生产 handoff 参数 | pending | 必须用真实跨场片段验收 |
| RKNN batch/串行/多上下文策略 | pending | 以整机实测选择 |

## 硬件与采集

- 已确认 SoC 为 RK3588S；仍需记录准确整机/载板型号、NPU 频率、内存容量和散热条件；
- 左右摄像头型号、镜头、曝光、快门、分辨率和稳定 60 FPS 能力；
- 硬件或驱动采集时间戳来源；
- 左右帧允许的最大时间偏差和丢帧策略。

## 每路独立图像参数

- `frame_scale_override`；
- `left_net_edge` / `right_net_edge` 的最终方向确认；
- 比赛区域 `play_area_normalized`；
- 观众排除区 `spectator_exclusion_regions`；
- 球网出口边界及容差；
- 近处、远处的透视尺度变化。

当前离线成对裁切数据使用
`(left_width + right_width) / reference_frame_width` 自动恢复原视频的像素尺度：
640+640 和 618+662 都得到 1.0，1920+1920 得到 3.0。该公式只适用于
同一原视频裁出的左右半场，不能直接用于真实独立双摄；真实双摄仍必须逐路标定
`frame_scale_override`。

## 跨摄 handoff

- `net_margin_ratio`；
- `receiving_band_ratio`；
- `min_toward_net_speed_px_per_second`；
- `alert_duration_ms`；
- `receiver_confirmation_hits`；
- `switch_lock_ms`。

## 板端性能选择

- RKNN 模型量化精度与检测召回率；
- 球模型 batch=2、单上下文受控串行或双上下文的实测选择；
- 球检测 120 张图/秒、人体检测 24 张图/秒的持续吞吐；
- 每个 60 FPS 帧对 16.67 ms 的采集、推理、追踪和输出延迟预算；
- 队列容量、超时、丢旧帧和最新帧优先策略。

所有标定结果必须记录摄像头、分辨率、FPS、模型哈希、配置 revision 和测试
视频，不能只修改 YAML 而没有验收记录。

每关闭一项至少附带：

```text
date:
operator:
complete_board:
camera_and_lens:
resolution_fps_exposure:
model_sha256:
config_profile_revision_sha256:
input_run_id:
measurement_method:
result:
evidence_path:
```

标定结果先写实验记录并完成固定视频回归；只有确认不会改变现有离线默认输出后，
才进入正式配置或板端专用配置。
