# 第二章：神经语言模型前向计算

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

[第二章正文](../../docs/02-neural-language-models.md) 对应 2.1–2.3；训练与梯度将在后续小节展开。
