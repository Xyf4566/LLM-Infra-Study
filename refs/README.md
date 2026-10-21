# 学习参考资料

收集学习 LLM Infra 的参考资料，包括论文、官方文档、课程、技术文章和开源项目。

## 核心参考仓库

以下参考仓库以 Git 子模块管理。主仓库固定每份资料的提交版本，参考仓库保留各自的历史与许可文件。

| 本地目录 | 上游仓库 | 学习用途 | 当前提交 |
| --- | --- | --- | --- |
| `d2l-en/` | [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en) | 语言模型、神经网络、注意力的基础推导与练习 | `23d7a5aecceee57d1292c56e90cce307f183bb0a` |
| `cs336-2025/` | [Stanford CS336 Spring 2025 lectures](https://github.com/stanford-cs336/spring2025-lectures) | 语言模型、训练、数据、推理与系统的课程主线 | `b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b` |
| `gpu-mode-lectures/` | [GPU MODE lectures](https://github.com/gpu-mode/lectures) | GPU 性能分析、CUDA/Triton、算子、通信与工程案例 | `77a8df418834e5789c12da23e7d2719e0efabef1` |
| `InfraTech/` | [InfraTech](https://github.com/CalvinXKY/InfraTech) | 训练与推理框架、模型架构、性能分析及 Notebook 实践 | `48e358a4b5119c24cf472adc64d0b4f86f894e99` |

首次获取本仓库及全部参考资料：

```bash
git clone --recurse-submodules https://github.com/Xyf4566/LLM-Infra-Study.git
```

如果已经克隆主仓库，在仓库根目录执行：

```bash
git submodule update --init --recursive
```

该命令检出主仓库记录的固定版本，便于复现实验。

CS336 克隆的是 2025 年课程讲义，不包含另外发布的作业仓库。GPU MODE 克隆的是讲座补充材料，视频和部分外链资料需单独访问。课程里的软件安装方式与硬件要求按对应实验核对。

## InfraTech 阅读入口

- [MLP 训练全流程](InfraTech/deeplearning_framework/mini_dl_framework.ipynb)：前向计算、反向传播与参数更新。
- [集合通信](InfraTech/deeplearning_framework/collective_operations.ipynb)：分布式训练与推理的通信基础。
- [推理练习](InfraTech/llm_infer/README.md)：采样、KV cache、投机解码、调度及推理框架。
- [模型架构](InfraTech/models/README.md)：主流模型的结构与计算流程。
- [共享资料](InfraTech/docs/README.md)：AI Infra 相关资料。

## 第一章阅读入口

- [D2L：文本预处理](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recurrent-neural-networks/text-sequence.md)：文本、token 与词表。
- [D2L：语言模型](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recurrent-neural-networks/language-model.md)：概率链式法则、n-gram、平滑与困惑度。
- [CS336 Lecture 1](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_01.py)：语言模型历史与分词。先阅读 `current_landscape()` 和 `tokenization()`。

学习顺序与实验见 [学习路径](../docs/00-learning-path.md)。
