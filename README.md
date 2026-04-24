# 手写数字识别本地项目

这是一个可以在本地运行的手写数字识别项目，使用 **TensorFlow/Keras + MNIST** 训练模型，然后用你自己的手写数字图片进行预测。

当前版本已经把功能拆清楚了：

1. `train.py` 只负责训练模型。
2. `test.py` 只负责加载已保存的模型并预测本地图片。

也就是说，`train.py` 不再读取 `t1.jpg` 到 `t11.jpg`，也不再生成本地图片预测图。  
如果只想测试新手写图片，直接运行 `test.py`，不需要重新训练。

## 1. 文件说明

`train.py`

训练主程序。它会训练 `MLP` 和 `CNN` 两个模型，比较它们在 MNIST 测试集上的准确率，保存表现最好的模型。

`test.py`

预测程序。它会加载 `models/best_digit_model.keras`，然后预测本地手写图片，例如 `t1.jpg` 到 `t11.jpg`。

`environment.yml`

Anaconda 环境配置文件，推荐用它创建环境。

`requirements.txt`

pip 依赖列表，作为备用安装方式。

`models/`

保存训练好的模型。

`outputs/`

保存运行结果图片和文字报告。

`t1.jpg` 到 `t11.jpg`

你自己的手写数字测试图片。它们只用于 `test.py` 预测，不参与训练。

## 2. 使用 Anaconda 运行

打开 Anaconda Prompt 或 PowerShell，进入项目文件夹：

```powershell
cd 你的项目路径
```

例如项目放在桌面的 `mnist` 文件夹，可以写成：

```powershell
cd C:\Users\你的用户名\Desktop\mnist
```

如果你把项目文件夹改名或移动了，就把上面这行换成新的项目路径即可。代码本身会自动根据 `train.py` 和 `test.py` 所在位置寻找 `models`、`outputs` 和 `t1.jpg` 到 `t11.jpg`。

如果是第一次运行，创建环境：

```powershell
conda env create -f environment.yml
```

如果环境已经存在，可以更新环境：

```powershell
conda env update -f environment.yml --prune
```

激活环境：

```powershell
conda activate mnist-ml
```

## 3. 训练模型

运行：

```powershell
python train.py
```

这一步会重新训练模型，所以耗时比较久。

`train.py` 会做这些事：

1. 加载 MNIST 数据集。
2. 训练 `MLP` 模型。
3. 训练 `CNN` 模型。
4. 在 MNIST 测试集上评估两个模型。
5. 保存 loss、accuracy、learning rate 曲线。
6. 保存模型对比图、混淆矩阵、错误样本图。
7. 保存最佳模型到 `models/best_digit_model.keras`。

注意：`train.py` 里仍然会有 `model.predict(...)` 这类代码，这是为了在 MNIST 测试集上计算准确率、混淆矩阵和错误样本，不是用来预测你的本地手写图片。

## 4. 预测本地手写图片

如果已经训练过模型，直接运行：

```powershell
python test.py
```

这一步不会重新训练。它只会加载：

```text
models/best_digit_model.keras
```

然后默认预测项目文件夹里的所有 `t*.jpg`、`t*.png`、`t*.bmp` 图片。  
例如当前会预测 `t1.jpg` 到 `t11.jpg`。

如果要预测新图片，例如 `my9.jpg`：

```powershell
python test.py --images my9.jpg --expected my9.jpg=9
```

如果要预测多张图片：

```powershell
python test.py --images my3.jpg my7.jpg --expected my3.jpg=3 my7.jpg=7
```

如果不知道真实数字，也可以不写 `--expected`：

```powershell
python test.py --images my_digit.jpg
```

## 5. 当前默认手写图片标签

`test.py` 默认知道这些图片的真实标签，方便统计本地测试准确率：

```text
t1.jpg=3
t2.jpg=5
t3.jpg=4
t4.jpg=7
t5.jpg=9
t6.jpg=2
t7.jpg=8
t8.jpg=9
t9.jpg=3
t10.jpg=6
t11.jpg=4
```

如果新增了 `t12.jpg` 这类图片，但没有在命令里写 `--expected t12.jpg=真实数字`，报告里会显示 `expected unknown`。

## 6. test.py 如何预处理图片

`test.py` 不是直接把照片丢给模型，而是先把照片处理成接近 MNIST 的 28x28 小图：

1. 读取图片，并根据照片自带的 EXIF 方向信息自动转正。
2. 分别尝试 `0`、`90`、`270`、`180` 四种旋转方向，避免横版照片被横着识别。
3. 从照片里提取深色笔迹，尽量压掉纸张背景和横线。
4. 找到最像数字主体的连通区域，把无关背景裁掉。
5. 把数字放到正方形画布中，缩放成 `28x28`。
6. 根据数字重心把笔迹移动到中间。
7. 让模型分别看几个旋转候选图，选择更可信的方向作为最终预测结果。

简单理解：这一步就是把手机拍的真实照片，尽量变成 MNIST 那种“居中、清晰、28x28”的数字小图。

## 7. 输出文件说明

训练时由 `train.py` 生成：

`outputs/01_dataset_examples.png`

MNIST 数据集示例图。

`outputs/02_mlp_training_curves.png`

MLP 的训练曲线，包含 train/val loss、train/val accuracy 和 learning rate。

`outputs/03_cnn_training_curves.png`

CNN 的训练曲线，包含 train/val loss、train/val accuracy 和 learning rate。

`outputs/04_model_comparison.png`

MLP 和 CNN 的测试集准确率对比图。

`outputs/05_confusion_matrix.png`

最佳模型在 MNIST 测试集上的混淆矩阵。

`outputs/06_wrong_samples.png`

最佳模型在 MNIST 测试集中识别错误的样本。

`outputs/results_summary.txt`

训练结果文字总结。

`outputs/classification_report.txt`

分类报告。

`outputs/training_history.json`

每一轮训练的原始指标数据。

预测时由 `test.py` 生成：

`outputs/07_xxx_preprocessed_28x28.png`

本地手写图片预处理后的 28x28 图。

`outputs/08_xxx_prediction.png`

本地手写图片预测结果图，包含原图、预处理图和 0 到 9 的概率柱状图。

`outputs/prediction_summary.txt`

本地图片预测结果文字总结。

`outputs/prediction_summary.json`

本地图片预测结果原始数据。

## 8. 常见问题

为什么运行 `test.py` 不需要重新训练？

因为最佳模型已经保存在 `models/best_digit_model.keras`。`test.py` 会直接加载这个文件做预测。

为什么 `train.py` 里还有预测相关单词？

训练完成后，需要让模型在 MNIST 测试集上输出预测结果，才能计算准确率、混淆矩阵和错误样本。所以 `train.py` 里的预测只服务于训练评估，不负责本地图片预测。

为什么本地手写图片预测放在 `test.py`？

这样更清楚：训练和预测互不干扰。以后你新增图片，只运行 `python test.py` 就可以。

如果提示 `conda` 不是命令，说明当前终端没有加载 Anaconda。建议打开 Anaconda Prompt 再运行。
