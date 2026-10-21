# 第一章实验：计数语言模型

比较上下文长度和平滑强度对训练、验证损失的影响。Python 3.11+，CPU 即可。

主要入口：[ngram.ipynb](ngram.ipynb)。Notebook 包含逐步讲解、完整实验代码、已运行的输出、图表和参数控件；[Jupyter 环境配置](../README.md)。

## 命令行运行

从仓库根目录执行：

```bash
python -m pip install -r experiments/01-ngram/requirements.txt
python -m unittest discover -s experiments/01-ngram -p 'test_*.py' -v
python experiments/01-ngram/run.py
```

需要已克隆的 `refs/d2l-en`，固定提交为 `23d7a5aecceee57d1292c56e90cce307f183bb0a`；获取方法见 [参考资料](../../refs/README.md)。运行不需要联网下载语料。

结果写入当前实验目录下的 [results/](results/README.md)。打开其中的 [index.html](results/index.html) 查看图表；Markdown、PNG、SVG 与 JSON 可分别阅读或导出。

要保留已发布结果、在另一个目录重跑：

```bash
python experiments/01-ngram/run.py --output /tmp/ngram-results
```

## 阅读代码

- `corpus.py`：清理 D2L Markdown，按文档划分数据，保存来源校验和。
- `ngram.py`：词表、文档边界、计数、平滑与损失。
- `run.py`：比较 40 组配置，用验证集选择一次，随后评估测试集并绘图。

上下文长度为 `0, 1, 2, 4, 8`；平滑强度为 `0.1, 1, 10, 100`。两种平滑方法共享公式：

```text
p(word | context) = (count(context, word) + alpha * prior(word))
                    / (count(context) + alpha)
```

`uniform` 使用均匀先验 `1 / V`；`backoff` 使用递归计算的短上下文分布。这里的 `alpha` 是虚拟计数总量；通常按每个词加 beta 的加性平滑对应 `alpha = beta * V`。

词表仅由训练集建立，训练出现不足 2 次的词及验证、测试中的词表外词映射为 UNK。每篇文档的历史独立重置，所有词和一个 EOS 参与评分，BOS 不计分。不同模型使用完全相同的目标位置。

## 语料来源

使用 D2L 八个基础章节目录内的 Markdown 文档，去掉索引、scratch/concise 实现页、代码、公式、图片和练习。转为小写英文词，每篇最多 1,200 词，保留至少 200 词的文档，并去除完全相同的 token 序列。每个章节目录随机取一篇验证、一篇测试，其余训练，种子为 42。

另用计算性能章节作为主题迁移集，在模型选定后评估。这个语料来自同一本教材，文件之间仍有共享术语；它衡量教材内文档泛化，不代表跨领域语言能力。

来源：[Dive into Deep Learning](https://github.com/d2l-ai/d2l-en)。作者 Aston Zhang、Zachary C. Lipton、Mu Li、Alexander J. Smola 及贡献者；原文许可 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。主仓库保存来源、统计结果和图表，直接从本地参考仓库读取原文。
