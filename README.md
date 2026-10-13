# 美食猜猜看

YOLO 食物检测与多模态大模型对比项目。交付目标：2026-10-15（周四），2026-10-16（周五）作为缓冲。

## 当前技术路线（2026-10-12 更新）

- 传统方案：小型 YOLO 检测模型，输出类别、边界框、置信度及数量。
- 多模态方案：独立分析同一张原图，按相同支持类别输出结构化识别结果，允许表示无法确定。
- 展示：FastAPI 与简单 HTML 页面，并排显示结果和耗时，支持答案揭晓。
- 学习实验：SAM 点／框提示分割，以及使用 YOLO 框作为分割提示。

## 数据与支持范围

检测主数据采用 UEC FOOD 100。已下载 `data/dataset100.zip`（991,355,222 字节，与官方响应头一致），已从压缩包读取类别表和 100 份 `bb_info.txt` 标注文件；已完成选定类别标注读取、样例检查和 YOLO 数据转换；全包 CRC 核验尚未记录。

检测方案从原六类调整为五类候选，移除蛋糕类：当前读取的 UEC FOOD 100 类别表没有对应蛋糕类别。以下映射已经过样例与数量核验，并用于首版训练。

| 项目类别 | UEC FOOD 100 原始类别编号与名称 |
| --- | --- |
| 米饭料理 | 1 rice、3 pilaf、9 fried rice |
| 面食 | 20 udon noodle、22 soba noodle、23 ramen noodle、27 spaghetti |
| 饺子 | 42 jiaozi |
| 披萨 | 18 pizza |
| 沙拉 | 86 potato salad、87 green salad、88 macaroni salad |

官方介绍：http://foodcam.mobi/dataset100.html 。官方限定非商业研究用途，项目说明应注明来源与使用范围，不将图片随代码重新发布。同一照片可能出现在多个类别目录，转换时合并所选类别框、去重并按图片划分，防止跨集合泄漏。原始框为像素坐标，需要根据实际图片宽高转换成 YOLO 归一化标签。

当前 `configs/classes.json` 为五类检测配置，类别编号依次为米饭料理 0、面食 1、饺子 2、披萨 3、沙拉 4。通用 COCO YOLO 权重不支持全部五类，需要在真实标注上微调才能形成项目模型。

Food-101 图片仅保留为早期探索素材和 YOLO 阈值演示输入。旧分类脚本、清单、报告和冒烟权重已清理，当前检测数据采用 UEC FOOD 100。

Food-101 来源：https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

## 实现状态

已完成项目初始化、Food-101 子集核验、分类清单及 CPU 虚拟环境；PyTorch、torchvision、Pillow、NumPy、scikit-learn、FastAPI、uvicorn 已安装。

ResNet18 探索已结束，相关脚本与临时实验产物已移除；历史 Git 提交保持原样。

已安装 Ultralytics 8.4.175，加载 YOLO11n 权重并通过 scripts/demo_yolo.py 完成披萨单图检测。本人已运行阈值 0.90／0.25 对照实验，结果分别为无框／保留分数约 0.842 的披萨框。检测数据转换、五类基线训练、双模型接口和上传页面已完成；SAM 最小提示实验已完成；正式评估、失败案例整理和运行视频尚未完成。本 README 尚非完整安装启动指南，后续随实现补齐。

## 配置与仓库约定

API 配置模板见 `.env.example`，真实密钥不入库。数据、权重和 `.venv` 保持本地；学习路线与个人学习笔记不提交。代码、依赖说明、数据准备与评估规则随实际进展持续提交。




## 已完成的数据与基线

像素内容去重后 2,327 张图片：训练 1,627、验证 350、测试 350。多类标注合并，完全重复图片不跨集合，近似重复仍需检查。计数为食物区域数，不代表单只饺子数量。

YOLO11n CPU 基线训练 10 轮，imgsz=320、batch=4、seed=42，总耗时约 22 分 37 秒。最佳模型验证 Precision=0.7907、Recall=0.8009、mAP50=0.8053、mAP50–95=0.5657；测试集未参与选模型。

数据转换脚本：scripts/prepare_uecfood100.py；小样本训练：scripts/train_yolo_smoke.py；完整基线：scripts/train_yolo_baseline.py。转换读取 configs/detection_classes.json；已有转换结果时拒绝覆盖。权重位于 models/yolo_runs/baseline_10epochs/weights，不入库。

scripts/demo_yolo.py 仍加载通用 COCO 权重，用于早期阈值演示；五类模型推理入口是 `scripts/predict_trained_yolo.py`。双模型页面已可运行，但正式测试和交付材料尚未完成。

## 本地双模型页面（2026-10-13）

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000 。本地 .env 按 .env.example 配置 DeepSeek；需要五类权重 models/yolo_runs/baseline_10epochs/weights/best.pt。上传图片会在确认后发送给 DeepSeek 并产生费用。支持最大 10 MB 图片，预处理限制最长边 1024，两个模型独立处理同一预处理图。YOLO 输入尺寸 320、阈值 0.25；该阈值是当前基线默认值，尚未完成正式阈值评估。

双模型上传接口已实现，支持独立错误反馈、60 秒 API 超时、结果字段检查。首次 YOLO 耗时包含模型加载及排队，API 耗时包含网络请求，不将二者直接解释为纯计算速度。SAM 最小实验已完成；正式评估和运行视频仍待完成。

## SAM 点／框提示实验（2026-10-13）

运行 `python scripts/demo_sam.py`，使用 SAM2.1 tiny 对同一验证图进行正点提示、真实标注框提示和 YOLO 预测框提示。首次运行下载公开权重约 74.5 MB；结果位于 reports/sam_demo，权重、图片和掩膜数组不入 Git。

三种提示均生成一个掩膜，单次 CPU 耗时约 2.1 秒。YOLO 框提示的可视化选中了锅体周围区域，未正确选择米饭，作为失败案例保留。生成掩膜不代表正确分割，SAM 分数也不是食物类别置信度。数据没有像素级真实掩膜，不报告分割准确率。SAM 当前是独立实验，尚未集成上传页面。
