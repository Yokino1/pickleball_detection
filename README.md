# Pickleball Detection and Tracking

本项目的当前主线是：在任意常见机位的视频中检测一个或多个 Pickleball，为每个球维护稳定
ID，并在短时漏检时用运动模型补全轨迹。场地坐标投影暂不属于主流程。

当前工作原则是先用真实侧拍视频验收继承模型，再根据漏检、误检、轨迹中断或板端性能问题
决定是否微调。现阶段不以重复训练后方视角公开数据作为默认动作。

## 当前能力

- 单帧 0-N 个球检测，支持 PyTorch/Ultralytics 和无 Torch 的 ONNX Runtime 后端。
- 多球独立 ID、两级置信度关联、卡尔曼预测、短时漏检补点和超时重捕获。
- 观测轨迹与预测轨迹使用不同可视化，JSONL 保留每帧完整检测和跟踪结果。
- 清洗后的 23,007 张检测数据，以及独立的训练、验证、测试视频片段划分。
- 模型训练、导出、精度回归、板端运行时基准和自动化单元测试。

## 快速开始

安装桌面训练环境：

```powershell
python -m pip install -r requirements-training.txt
```

运行当前 PyTorch 模型：

```powershell
python apps/track_video.py --config configs/tracking.yaml
```

运行板端 ONNX 配置：

```powershell
python -m pip install -r requirements-runtime.txt
python apps/track_video.py --config configs/tracking_edge.yaml
```

输出默认写入 `outputs/tracking_overlay.mp4` 和 `outputs/tracking.jsonl`。预测点标记为
`pred Nf`，表示已经连续 N 帧没有检测框、当前位置来自运动预测。

## 项目结构

```text
apps/                         可执行入口
  track_video.py              当前主入口：检测 + 多球跟踪
configs/                      可版本化运行配置
src/tracking/                 检测、关联、预测、结果结构和可视化
tools/                        数据、训练、导出、验证和基准工具
tests/                        不依赖模型文件的核心算法测试
datasets/                     原始数据与清洗后训练数据
artifacts/                    模型、训练结果和基准报告
docs/                         架构、训练、部署、维护和路线图
outputs/                      本地运行输出，不提交版本库
legacy/handoff_projection/    只读交接归档，不参与构建、测试或发布
```

## 文档入口

- [架构说明](docs/ARCHITECTURE.md)
- [当前推进方案](docs/NEXT_STEPS.md)
- [训练与评估](docs/TRAINING.md)
- [板端部署](docs/DEPLOYMENT.md)
- [开发维护规范](docs/DEVELOPMENT.md)
- [项目路线图](docs/ROADMAP.md)
- [数据说明](datasets/README.md)
- [遗留代码说明](legacy/handoff_projection/README.md)
- [变更记录](CHANGELOG.md)

## 当前边界

- 现有公开数据主要是离散检测图片，不能可靠计算 IDF1/HOTA；正式跟踪指标需要补充带
  `track_id` 的连续视频测试集。
- 最终推理格式取决于板卡。ONNX Runtime 是通用基线，NVIDIA、Rockchip、Intel 等平台应在
  确认硬件后切换对应加速后端。
- 继承权重是单类别检测模型，YOLO 本身支持一帧输出多个球；正式多球效果仍需用真实侧拍连续
  视频验证，不能只根据训练集数量下结论。
- 旧场地投影分支已归档，不属于支持范围，也不能被主线代码引用。
