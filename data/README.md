# 本地视频目录

- `sideview_raw/`：当前批量验收的原始侧拍视频，是 `--input data/sideview_raw` 的输入目录。
- `reference/`：小规模参考或单视频冒烟样例，默认配置使用 `reference/test_2.mp4`。

视频文件是本地数据，不提交 Git。不要在 `data/` 根目录重复存放同一视频，也不要把检测结果写到
这里；所有生成结果统一进入 `outputs/`。
