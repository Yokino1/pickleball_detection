# Experiment Registry

这里保存值得复现和评审的实验元数据，不保存 MP4、JSONL、模型或数据集。

规则：

1. 实验 ID 使用 `YYYYMMDD_topic_purpose` 或稳定的里程碑名称。
2. 每个实验记录唯一配置 profile/revision、代码提交、输入片段、假设和验收指标。
3. 实际运行的代码提交、配置 SHA-256 和输入元数据由输出目录中的
   `*_manifest.json` 自动记录。
4. 临时 smoke 不需要登记；影响设计决策或准备提升为正式版本的实验才登记。
5. 实验结论写入记录，算法行为变化同步写入 `CHANGELOG.md`。
6. 历史实验不得改写成当前配置身份；使用 `record_type: historical`、
   `superseded_by` 和 `retained_config` 说明其可复现状态。

输出目录约定见 [outputs/README.md](../outputs/README.md)，维护流程见
[docs/MAINTENANCE.md](../docs/MAINTENANCE.md)。
