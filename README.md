# 基于 MNIST 的手写数字识别

这是一个使用 **TensorFlow / Keras** 实现的本地手写数字识别项目。项目会在 MNIST 数据集上训练并比较 MLP 与 CNN，保存表现更好的模型；也可以直接加载仓库内已有的 CNN 模型，识别手机拍摄或扫描的手写数字图片。

本项目仅提供 Python 训练与预测流程，不包含 Web 前端。

## 项目特点

- 自动下载并加载 MNIST 数据集
- 训练、评估和比较 MLP 与 CNN
- 保存训练曲线、混淆矩阵、错误样本和分类报告
- 自动选择并保存测试集准确率更高的模型
- 支持本地 JPG、JPEG、PNG 和 BMP 图片
- 对真实图片执行裁剪、缩放、居中和背景抑制
- 自动比较 `0°`、`90°`、`270°`、`180°` 四种方向
- 支持通过 JSON 或命令行提供真实标签并统计准确率

## 当前结果

仓库内结果由完整 MNIST 测试集（10,000 张图片）得到：

| 模型 | 测试准确率 |
| --- | ---: |
| MLP | 97.92% |
| CNN | 99.38% |

当前保存的最佳模型是 CNN，位于 `models/best_digit_model.keras`。仓库自带的 10 张本地手写图片均已正确识别；这些样例结果只用于验证本地图片处理流程，不代表模型在任意真实场景中的泛化准确率。

## 项目结构

```text
.
├── train.py                         # 训练、评估并保存最佳模型
├── test.py                          # 识别本地手写数字图片
├── requirements.txt                 # pip 依赖
├── environment.yml                  # Conda 环境
├── expected_labels.json             # 示例图片的真实标签
├── models/
│   ├── best_digit_model.keras       # 已训练的最佳模型
│   └── best_digit_model_info.json   # 模型基本信息
├── outputs/                          # 训练与预测结果
└── t1.jpg ... t10.jpg               # 本地测试样例
```

## 环境要求

- Python 3.11
- TensorFlow
- NumPy
- Matplotlib
- Pillow
- SciPy

推荐使用 Conda 创建环境：

```powershell
conda env create -f environment.yml
conda activate mnist-ml
```

已有同名环境时可以更新：

```powershell
conda env update -f environment.yml --prune
```

也可以使用 pip：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 快速开始

仓库已经包含训练好的模型。安装依赖后，可直接识别根目录下的 `t*.jpg` 示例：

```powershell
python test.py
```

预测结果会打印到终端，并保存到 `outputs/`。

识别一张自己的图片：

```powershell
python test.py --images my_digit.jpg
```

同时识别多张图片：

```powershell
python test.py --images digit_1.png digit_2.jpg
```

提供真实标签以便统计准确率：

```powershell
python test.py --images digit_1.png digit_2.jpg --expected digit_1.png=3 digit_2.jpg=7
```

指定其他模型：

```powershell
python test.py --model-path path\to\model.keras --images my_digit.jpg
```

## 真实标签文件

默认情况下，`test.py` 会读取 `expected_labels.json`。文件格式如下：

```json
{
  "labels": {
    "t1.jpg": 6,
    "t2.jpg": 1
  }
}
```

命令行中的 `--expected` 会覆盖标签文件里的同名记录。若不想读取标签文件，可以使用：

```powershell
python test.py --expected-labels none --images my_digit.jpg
```

也可以指定自己的标签文件：

```powershell
python test.py --expected-labels labels.json --images digit_1.png digit_2.jpg
```

## 重新训练

运行完整训练流程：

```powershell
python train.py
```

该命令会分别训练 MLP 和 CNN，在 MNIST 测试集上进行评估，并将准确率更高的模型写入 `models/best_digit_model.keras`。

如需快速验证训练流程，可以限制训练样本数量：

```powershell
python train.py --train-limit 10000
```

`--train-limit 0` 表示使用全部训练数据，也是默认设置。首次运行时，TensorFlow 会自动下载 MNIST 数据集。

## 图片预处理

MNIST 是黑底白字的 `28 × 28` 灰度图，而手机照片通常包含背景、阴影、倾斜和方向差异。`test.py` 会依次进行：

1. 根据 EXIF 信息修正图片方向。
2. 比较四种旋转方向并生成候选图。
3. 提取深色笔迹，裁剪主要数字区域。
4. 缩放并按重心居中为 `28 × 28` 图像。
5. 过滤空白或质量过低的候选图。
6. 在低置信度时尝试更柔和的背景处理。
7. 选择综合质量和模型置信度更好的结果。

预处理可以缩小真实照片与 MNIST 的差异，但光照、纸张纹理、拍摄角度和书写风格仍可能影响结果。

## 输出文件

训练后会生成：

```text
outputs/01_dataset_examples.png
outputs/02_mlp_training_curves.png
outputs/03_cnn_training_curves.png
outputs/04_model_comparison.png
outputs/05_confusion_matrix.png
outputs/06_wrong_samples.png
outputs/results_summary.txt
outputs/results_summary.json
outputs/classification_report.txt
outputs/training_history.json
models/best_digit_model.keras
models/best_digit_model_info.json
```

执行预测后会生成：

```text
outputs/07_<图片名>_preprocessed_28x28.png
outputs/08_<图片名>_prediction.png
outputs/prediction_summary.txt
outputs/prediction_summary.json
```

其中预测图包含原图、预处理结果和 0–9 各类别的概率，JSON 文件适合进一步处理或集成到其他程序中。

## 注意事项

- `train.py` 只使用 MNIST，不读取根目录下的本地图片。
- `test.py` 只加载已保存模型，不会重新训练。
- 若模型文件不存在，请先运行 `python train.py`。
- 自定义图片中最好只包含一个数字，并尽量保证笔迹清晰、背景简单、光照均匀。
