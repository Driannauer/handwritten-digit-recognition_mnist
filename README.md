# 手写数字识别本地项目

这是一个使用 **TensorFlow/Keras + MNIST** 的本地手写数字识别项目。

项目分成两个主要脚本：

- `train.py`：训练并评估模型，保存最佳模型。
- `test.py`：加载已保存模型，预测本地手写数字图片。

如果只是测试自己的手写图片，通常只需要运行 `test.py`，不需要重新训练。

## 在线演示

GitHub Pages: [https://driannauer.github.io/handwritten-digit-recognition_mnist/](https://driannauer.github.io/handwritten-digit-recognition_mnist/)

静态前端位于 `docs/`，可以直接在浏览器里画数字、上传图片或点选样例。浏览器端会尽量模仿 `test.py` 的照片预处理流程：高斯背景估计、提取笔迹、选择主要连通域、裁剪、增强、缩放居中，并自动比较 `0`、`90`、`270`、`180` 度候选方向后再显示预测结果。

## 文件结构

```text
train.py                 训练 MLP/CNN 模型并保存最佳模型
test.py                  加载模型并预测本地图片
environment.yml          Conda 环境配置
requirements.txt         pip 依赖列表
expected_labels.json     默认本地测试图片真实标签
docs/                    GitHub Pages 静态前端和浏览器模型权重
models/                  保存训练好的模型
outputs/                 保存训练和预测输出
t*.jpg                   本地手写数字测试图片
```

## 环境配置

推荐使用 Conda：

```powershell
conda env create -f environment.yml
conda activate mnist-ml
```

如果环境已经存在，可以更新：

```powershell
conda env update -f environment.yml --prune
```

备用 pip 安装：

```powershell
pip install -r requirements.txt
```

## train.py 大致内容

`train.py` 只负责训练和评估，不读取 `t*.jpg` 本地手写图片。

主要流程：

1. 设置随机种子和输出目录。
2. 加载 MNIST 数据集，并划分训练集、验证集、测试集。
3. 保存 MNIST 示例图到 `outputs/01_dataset_examples.png`。
4. 构建并训练 MLP 模型。
5. 构建并训练 CNN 模型。
6. 在 MNIST 测试集上评估两个模型。
7. 保存训练曲线、模型对比图、混淆矩阵、错误样本和文本报告。
8. 选择测试准确率更高的模型，保存到 `models/best_digit_model.keras`。

主要模块：

- `load_mnist_data()`：加载并归一化 MNIST 数据。
- `build_mlp()`：定义全连接神经网络。
- `build_cnn()`：定义带数据增强的 CNN。
- `fit_with_history()`：训练模型并记录 loss、accuracy、learning rate。
- `evaluate_model()`：在测试集上计算准确率和预测结果。
- `save_training_curves()`：保存训练曲线。
- `save_confusion_matrix()`：保存混淆矩阵。
- `save_wrong_samples()`：保存错误样本图。
- `save_model()`：保存最佳模型和模型信息。

可选参数：

```powershell
python train.py --train-limit 10000
```

`--train-limit` 用于限制训练样本数量，默认 `0` 表示使用全部训练样本。

## test.py 大致内容

`test.py` 只负责预测本地图片，不重新训练模型。

主要流程：

1. 加载 `models/best_digit_model.keras`。
2. 默认查找项目目录中的所有 `t*.jpg`、`t*.jpeg`、`t*.png`、`t*.bmp`。
3. 读取图片，并根据 EXIF 信息自动转正。
4. 尝试 `0`、`90`、`270`、`180` 四种旋转方向。
5. 从照片中提取深色笔迹，裁剪主要数字区域。
6. 将数字缩放、居中成 MNIST 风格的 `28x28` 输入。
7. 过滤空白或质量太差的候选图。
8. 让模型批量预测候选图，并选择最可信的方向。
9. 对有明显顶部横画的候选图额外生成 `top_bar_boost` 候选，而不是直接覆盖原始候选。
10. 低置信度时使用 `soft_background` 做一次更柔和的背景预处理，并在候选质量足够时允许切换预测标签。
11. 保存预处理图、预测图、txt 报告和 json 报告。

主要模块：

- `discover_default_images()`：发现默认本地图片。
- `load_expected_labels_file()`：从 `expected_labels.json` 这类 manifest 读取真实标签。
- `parse_expected_labels()`：解析真实标签。
- `preprocess_digit_array()`：把照片区域转成 `28x28` 数字图。
- `score_processed_digit_quality()`：评估候选图质量。
- `build_top_bar_boost_candidate()`：把明显顶部横画增强作为额外候选。
- `choose_candidate_for_model()`：在多个旋转候选中选择最终候选。
- `refine_low_confidence_candidate()`：低置信度时尝试 `soft_background` 二次预处理。
- `save_prediction_figure()`：保存原图、预处理图和概率柱状图。
- `save_prediction_reports()`：保存预测摘要。

可选参数：

```powershell
python test.py --images my_digit.jpg
python test.py --images my3.jpg my7.jpg --expected my3.jpg=3 my7.jpg=7
python test.py --expected-labels expected_labels.json
python test.py --model-path models/best_digit_model.keras
```

## 常用命令

训练模型：

```powershell
python train.py
```

预测默认图片：

```powershell
python test.py
```

预测指定图片并提供真实标签：

```powershell
python test.py --images t9.jpg --expected t9.jpg=3
```

预测指定图片但不提供真实标签：

```powershell
python test.py --images my_digit.jpg
```

## 默认标签

项目用 `expected_labels.json` 保存默认本地测试图片标签，用于计算已知标签准确率。只有图片文件真实存在、并且 manifest 或命令行里提供了标签时，才会参与统计。

```text
t1.jpg=6
t2.jpg=1
t3.jpg=8
t4.jpg=9
t5.jpg=7
t6.jpg=2
t7.jpg=0
t8.jpg=5
t9.jpg=3
t10.jpg=4
```

新增图片时，可以在命令中显式写真实标签：

```powershell
python test.py --images new_digit.jpg --expected new_digit.jpg=7
```

命令行 `--expected` 会覆盖 `expected_labels.json` 中的同名条目。如果想完全不读取 manifest，可以使用：

```powershell
python test.py --expected-labels none
```

## 输出文件

训练输出：

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

预测输出：

```text
outputs/07_xxx_preprocessed_28x28.png
outputs/08_xxx_prediction.png
outputs/prediction_summary.txt
outputs/prediction_summary.json
```

## 预处理说明

真实手机照片和 MNIST 差异很大，所以 `test.py` 会先把图片尽量变成 MNIST 风格：

- 深色笔迹变成亮色数字。
- 背景、阴影和纸张纹理尽量压低。
- 裁剪出主要数字区域。
- 缩放到 `28x28`。
- 根据重心居中。
- 自动比较多个旋转方向。

`docs/` 中的浏览器前端也会尽量模仿 `test.py` 的预处理流程，并在识别时比较 `0`、`90`、`270`、`180` 度方向。

当前还有两个针对真实照片的增强：

- `top_bar_boost`：把顶部横画增强后的图作为额外候选，让模型自己和原始候选比较。
- `soft_background`：当模型低置信时，用更柔和的背景估计重新提取笔迹；只有新候选的质量和模型分数更好时才采用。

这些增强属于预测阶段的预处理，不会修改训练好的模型权重。

## 常见问题

为什么 `test.py` 不需要重新训练？

因为训练好的最佳模型保存在 `models/best_digit_model.keras`，`test.py` 会直接加载它。

为什么 `train.py` 中也有 `predict`？

训练结束后需要在 MNIST 测试集上预测，才能计算准确率、混淆矩阵和错误样本。这和本地图片预测不是一回事。

为什么本地图片有时置信度低？

模型只在 MNIST 上训练，而本地照片包含拍照角度、阴影、纸张纹理、笔画粗细和个人书写风格差异。预处理可以缓解，但如果想更稳，最好收集自己的手写样本并加入训练或微调。

如果提示 `conda` 不是命令怎么办？

说明当前终端没有加载 Conda。建议打开 Anaconda Prompt，或确认 Anaconda 已加入系统 PATH。
