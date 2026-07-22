# 交接投影代码归档

这里保留交接时收到的侧视角坐标投影实验代码，目的是追溯历史。它不属于当前维护产品，也不进入
构建、测试或发布包。

## 归档内容

- `apps/run_sideview_video.py`：旧单球、球员、场地和投影入口；
- `tools/annotate_video.py`：旧标注演示，只保留置信度最高的一个球；
- `src/court/`：标定、投影、绘制和区域判断；
- `src/tracking/ball_track.py`：旧单球跟踪器；
- `src/tracking/pipeline.py`、`events.py`、`player_detector.py`：旧组合流程；
- `configs/`：对应的标定与展示配置。

## 已知限制

旧入口引用了 `src.court.pose_tracker`、`src.court.template_tracker` 和 `src.court.lk_tracker`，但交接项目
没有包含这些模块。因此归档不承诺可运行，里面的旧导入和路径按收到时的状态保留，不能据此认为
这些能力已经完成。

`apps/`、`src/`、`tools/` 和 `tests/` 禁止引用该归档。以后恢复坐标投影时，应先新增 ADR，再基于当前
`FrameResult.ball_tracks` 接口重新实现并补充独立测试。
