# 本地运行结果

该目录只存放生成结果，不提交 Git。新运行按用途分类：

```text
outputs/
  production/   已验收、需要长期保留的候选结果
  experiments/  算法或参数实验
  smoke/        30～100 帧短片检查，可随时清理
  previews/     分界线、ROI 和布局预览
```

推荐每个实验使用独立目录和可追溯 run ID：

```text
outputs/experiments/dual_camera/
  20260725_full_clean_person-contact-r2_phase23_dual_person_contact.mp4
  20260725_full_clean_person-contact-r2_phase23_left_tracking.jsonl
  20260725_full_clean_person-contact-r2_phase23_right_tracking.jsonl
  20260725_full_clean_person-contact-r2_phase23_global_tracking.jsonl
  20260725_full_clean_person-contact-r2_phase23_manifest.json
```

规则：

- 不使用 `final2`、`new`、`v5` 等无法说明含义的目录或文件名。
- 同名正式结果默认不得覆盖；需要替换时显式使用 `--overwrite`。
- `.partial.*` 表示未完成输出，不参与结果比较。
- Smoke 验证完成后删除；有价值的失败片段应转成固定回归数据。
- 指标报告放在 `artifacts/benchmarks/`，可提交的实验结论放在 `experiments/`。
- 派生半场输入统一放在 `data/derived/court_halves/`，不再混入结果目录。

完整维护规则见 [docs/MAINTENANCE.md](../docs/MAINTENANCE.md)。
