# Configuration Profiles

配置文件是算法版本的一部分，不是临时调参草稿。每个正式配置必须包含：

```yaml
schema_version: 1
profile:
  name: physics
  revision: 1
  status: maintained
```

字段含义：

- `schema_version`：配置结构兼容版本；字段语义不兼容时才递增。
- `profile.name`：稳定的算法配置名称。
- `profile.revision`：同一 profile 的参数或行为修订号。
- `profile.status`：`maintained`、`experimental` 或 `deployment`。

当前配置：

| Profile | 文件 | 状态 |
| --- | --- | --- |
| mainline | `tracking.yaml` | maintained |
| temporal | `tracking_temporal.yaml` | maintained |
| physics | `tracking_physics.yaml` | maintained |
| person_contact_dual_camera | `tracking_person_contact.yaml` | experimental |
| edge | `tracking_edge.yaml` | deployment |

临时调参不要复制出 `v2_final_new.yaml` 一类文件。先通过命令参数或实验记录保存；
只有需要长期回归、板端部署或正式 A/B 对比时才新增/修订 profile。

`tracking_edge.yaml` 当前是单路 ONNX Runtime 可移植基线，不是双摄 60 FPS
RKNN 发布配置。准确 RK3588S 级板卡、RKNN 模型和摄像头参数冻结前，不创建
名义上的 release profile。
