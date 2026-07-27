# Configuration Profiles

配置文件是算法版本的一部分，不是临时调参草稿。

## 当前活动配置

| Profile | 文件 | Revision | 状态 |
| --- | --- | ---: | --- |
| pickleball_tracking | `tracking.yaml` | 9 | maintained |
| edge | `tracking_edge.yaml` | 1 | deployment research |

`tracking.yaml` 是唯一活动正式算法配置，正式入口是离线双半场
`apps/track_dual_halves.py`。单视频工具可以复用它做局部诊断，但不构成另一套
正式运行方式。配置包含 observation-first、人体接触、物理门控和辅助 handoff。

`tracking_edge.yaml` 只是单路 ONNX Runtime 可移植基线，不是另一套正式算法版本，
也不是已经验收的 RK3588S/RKNN 发布配置。

## 历史配置

原 mainline、temporal 和 physics 三套桌面 YAML 已停止维护，位于：

```text
legacy/ball_tracking_handoff/configs/maintained_history/
```

同一目录另保存退役的 `tracking_person_contact_r4.yaml` 实验快照。它仅用于历史
复查，不属于三套旧正式版本，也不是当前 revision 9 的替代入口。

活动代码和默认入口不得依赖历史配置。需要回看旧结果时，必须显式指定历史 YAML，
并使用新的历史回归 run ID。

## Profile 元数据

```yaml
schema_version: 1
profile:
  name: pickleball_tracking
  revision: 9
  status: maintained
```

- `schema_version`：配置结构兼容版本；
- `profile.name`：稳定算法名称；
- `profile.revision`：当前正式算法行为修订号；
- `profile.status`：`maintained`、`experimental` 或 `deployment`。

真实双摄尚未冻结的参数记录在
[`docs/CAMERA_CALIBRATION_TODO.md`](../docs/CAMERA_CALIBRATION_TODO.md)。
当前实现和板端交接边界见
[`docs/HANDOFF.md`](../docs/HANDOFF.md)。
