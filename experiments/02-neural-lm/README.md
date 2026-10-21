# 第二章：神经语言模型计算与梯度

6 个 token，2 个历史位置，2 维 embedding，3 维隐藏层，共 51 个参数。固定参数用于逐步核对前向计算。

主要入口：[forward.ipynb](forward.ipynb)。依次运行查表、拼接、隐藏层和 softmax，使用控件切换上下文、调整 embedding 坐标。[Jupyter 环境配置](../README.md)。

Python 3.11+，从仓库根目录执行：

```bash
python -m pip install -r experiments/02-neural-lm/requirements.txt
python experiments/02-neural-lm/forward.py
python experiments/02-neural-lm/forward.py --first 你 --second 喜欢
python experiments/02-neural-lm/forward.py --like-coordinate 1
```

程序依次输出 embedding、拼接向量、隐藏层、logits、概率和目标词 NLL。

最后一个命令修改共享 embedding 表中“喜欢”的第一维；可对照 [Notebook](forward.ipynb) 观察多个上下文的输出变化。

[前向计算 Notebook](forward.ipynb) 对应[第二章正文](../../docs/02-neural-language-models.md) 2.1–2.3。

## 输出层反向传播

[backward.ipynb](backward.ipynb) 对应 2.4.1–2.4.2，依次计算交叉熵对 logits、输出层权重、偏置和隐藏表示的梯度，用中心差分核对，再仅更新输出层观察目标概率与损失变化。Notebook 保存已执行输出，使用相同的 Jupyter 环境。

实验使用 CPU、NumPy float64、batch size 1、窗口长度 2、固定的 51 个模型参数，输入“我 喜欢”，目标“数学”。学习率为 0.1，差分步长为 1e-5；无随机采样、无数据集划分。embedding 和第一层暂不更新，完整反向传播与训练在后续展开。
