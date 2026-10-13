# 美食猜猜看

一个用于比较传统计算机视觉模型与多模态大模型食物识别能力的 Web 应用。用户上传图片后，系统分别使用本地 YOLO 检测模型和 DeepSeek 视觉模型分析，并展示类别、区域框、置信度、数量、未知食物提示和处理耗时。

## 功能

- 本地 YOLO11n 五类食物检测：输出类别、边界框、置信度和检测区域数量。
- DeepSeek 多模态分析：输出支持类别、数量、未知类别提示和简短描述。
- 同一张预处理图片分别送入两条路线，结果并排展示，便于比较。
- FastAPI 上传接口和浏览器页面。
- 网页交互式 SAM2.1 tiny 分割：点击图片提供正点提示，或选择 YOLO 检测框作为提示；另外提供真实框提示对照实验。
- 固定测试样本的双模型对比评测，结果位于 `reports/comparison/`。

## 支持类别

项目将 UEC FOOD 100 的细分类别合并为五类：

| 项目类别 | UEC FOOD 100 类别 |
| --- | --- |
| 米饭料理 | rice、pilaf、fried rice |
| 面食 | udon noodle、soba noodle、ramen noodle、spaghetti |
| 饺子 | jiaozi |
| 披萨 | pizza |
| 沙拉 | potato salad、green salad、macaroni salad |

检测数据来自 [UEC FOOD 100](http://foodcam.mobi/dataset100.html)，仅按其非商业研究用途使用。数据集图片和训练权重不随仓库发布；运行项目需要在本地准备对应文件。

## 技术结构

```text
浏览器 -> FastAPI -> YOLO11n 本地推理
                 -> DeepSeek 视觉 API
```

YOLO 模型使用 UEC FOOD 100 的边界框标注进行五类微调。当前基线数据包含 2,327 张去重图片，训练、验证、测试数量分别为 1,627、350、350。计数表示标注或检测区域数，不表示单个食物颗粒数量。

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

准备以下本地文件：

- `models/yolo_runs/baseline_10epochs/weights/best.pt`
- `models/sam2.1_t.pt`（网页分割功能需要）
- `data/uecfood100_detection/`（需要重新准备数据或复现实验时）
- `.env` 中的 DeepSeek 配置，可参考 `.env.example`

真实 API 密钥不要提交到 Git。

## 启动应用

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。提交分析时后端会并行运行 YOLO 和 DeepSeek，DeepSeek 请求需要有效配置并产生 API 费用。SAM 分割支持图片点提示和 YOLO 框提示。蓝色掩膜表示 SAM 选中的区域，需要核对是否覆盖目标。接口限制单张图片最大 10 MB，并限制图片最长边和像素数量以控制资源消耗。

## 数据准备、训练和推理

复现训练前，从 [官方页面](http://foodcam.mobi/dataset100.html) 获取数据压缩包，放在 `data/dataset100.zip`。下列命令通过 Ultralytics 获取公开的 YOLO11n 和 SAM2.1 tiny 权重，需要联网。通用 YOLO11n 不是项目五类模型，必须执行后续训练生成 `best.pt`。

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from ultralytics import YOLO, SAM; Path('models').mkdir(exist_ok=True); YOLO('models/yolo11n.pt'); SAM('models/sam2.1_t.pt')"
```

转换脚本会生成训练所需的数据目录；已有转换结果时拒绝覆盖。单图推理命令从验证集中选择指定类别的样例。以下 CPU 训练在本项目环境中约需 23 分钟，实际时间取决于硬件。模型权重获取和校验方式见 [权重复现说明](docs/权重复现说明.md)。

```powershell
.\.venv\Scripts\python.exe scripts\prepare_uecfood100.py
.\.venv\Scripts\python.exe scripts\train_yolo_baseline.py
.\.venv\Scripts\python.exe scripts\predict_trained_yolo.py --category noodles --index 0 --conf 0.25
```

SAM 提示实验：

```powershell
.\.venv\Scripts\python.exe scripts\demo_sam.py
```

## 当前基线

YOLO11n 基线使用 `imgsz=320`、`batch=4`、`epochs=10` 和 `seed=42`。验证集指标为 Precision 0.7907、Recall 0.8009、mAP50 0.8053、mAP50-95 0.5657。固定 10 张测试图片的双模型对比结果及限制说明见 [交付说明与评测记录](docs/交付说明与评测记录.md)。

## 仓库内容

- `src/`：FastAPI 服务和前端页面
- `scripts/`：数据转换、训练、推理、评测和 SAM 实验脚本
- `configs/`：类别和检测数据配置
- `reports/`：数据检查、训练、评测和演示报告
- `docs/`：面向审核者的项目交付说明与评测记录

数据集、模型权重、虚拟环境、`.env` 和个人学习笔记均不提交。
