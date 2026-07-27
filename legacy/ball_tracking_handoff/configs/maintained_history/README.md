# Archived Desktop Tracking Profiles

该目录保存 2026-07-26 以前用于桌面 A/B 对比的三套历史配置：

- `tracking_mainline_r1.yaml`：单帧 YOLO + 常速度 Kalman；
- `tracking_temporal_r1.yaml`：mainline + 相邻帧运动证据；
- `tracking_physics_r1.yaml`：temporal + 常加速度 Kalman 和物理门控。

另保留一套曾用于人体接触/双摄实验的最终配置快照：

- `tracking_person_contact_r4.yaml`：已退役的实验版本，不属于三套正式桌面版本，
  也不等价于当前 revision 9。

它们已经停止维护，不再作为活动入口默认配置。对应历史 CMD 位于
`legacy/ball_tracking_handoff/scripts/maintained_history/`。

活动代码不得导入 `legacy`。只有明确执行历史回归时，才可以把这些 YAML
作为 `apps/track_video.py --config` 的外部输入。
