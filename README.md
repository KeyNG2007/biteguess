# 美食猜猜看

六类食物识别与双模型拒答挑战。交付目标：2026-10-18。

## 两条独立路线

- 传统模型：微调预训练 ResNet18。
- 多模态模型：对同一图片使用相同类别清单，允许拒答。

目前完成项目初始化与类别映射，尚未训练模型或实现 API。

## 数据

Food-101 官方页面：https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

完整下载约 5 GB，解压需要额外空间。数据与权重不入 Git。
下载：`python scripts/download_food101.py`
下载完成后先检查归档，再按所需类别安全解压和建立划分；暂不自动解压。

类别映射见 `configs/classes.json`。保留官方训练／测试边界，从训练部分划出验证集；各阶段保存抽样记录。
粗类别仅覆盖映射中的子类，不代表支持全部同类料理。

## 配置与状态

API 配置模板见 `.env.example`，不要提交真实密钥。
当前 Python 环境未安装 PyTorch／torchvision／FastAPI。确认训练设备后再选择并安装适合的版本。


