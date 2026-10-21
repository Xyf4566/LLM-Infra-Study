# LLM Infra 学习路径

## 资料的使用方式

1. D2L 用于基础概念、推导和小实验，重点阅读语言模型与注意力相关章节。
2. CS336 2025 讲义用于串联模型、数据、训练、评估、推理和系统。其作业另行选取并缩小到单机规模。
3. GPU MODE 按主题穿插，先做 profiling，再逐步进入 CUDA、Triton、通信与工程案例。
4. 前沿部分结合官方论文、仓库和技术报告，分析模型、算子、通信与调度之间的相互影响。

参考仓库的位置、来源、版本与获取方式见 [资料索引](../refs/README.md)。

## 阶段与实验

| 阶段 | 核心内容 | 主要阅读入口 | 练习与可视化 |
| --- | --- | --- | --- |
| 1. 语言模型的概率基础与历史 | 序列分布、条件概率、链式法则、n-gram、最大似然、交叉熵、困惑度 | D2L `text-sequence.md`、`language-model.md`；CS336 Lecture 1、12 | CPU：统计小语料，画转移概率热力图，比较上下文长度、平滑和验证集损失 |
| 2. 神经语言模型 | 词向量、softmax、反向传播、RNN/LSTM、Seq2Seq 与早期注意力 | D2L 循环网络、现代循环网络与注意力章节；CS336 Lecture 2 选读 | NumPy/PyTorch 梯度对照、小型 MLP/RNN 语言模型；观察损失与生成样本 |
| 3. Transformer | Q/K/V、因果掩码、多头注意力、残差、归一化、位置编码、现代解码器 | D2L Transformer；CS336 Lecture 3 架构讲义 | 实现小型解码器；可视化注意力权重、掩码与张量形状 |
| 4. 完整预训练与评估 | 分词、去重、数据混合、优化、检查点、规模化规律 | CS336 Lecture 1、2、9、11、12、13、14 | 单 GPU：从约 5M–30M 参数、128–256 token 上下文开始；比较验证损失、tokens/s 和显存 |
| 5. 后训练原理 | SFT、LoRA、偏好学习、RLHF/RLVR 的目标与数据流 | CS336 Lecture 15、16、17；后续补充 LoRA 原始资料 | 小模型 SFT 和可验证奖励任务；观察质量变化与训练稳定性 |
| 6. 推理机制 | 采样、KV cache、prefill/decode、TTFT、逐 token 延迟 | CS336 Lecture 10 | 比较同一模型带/不带缓存的输出与性能；绘制上下文长度、batch size、显存和延迟曲线 |
| 7. GPU 与性能分析 | 内存层次、算术强度、Roofline、同步、benchmark 与 profiler | CS336 Lecture 2、5、6；GPU MODE 1、3、4、8、16 | GPU：矩阵乘法、带宽测量、算子时间线；计时包含预热和正确同步 |
| 8. 算子与编译 | 分块、融合、softmax、FlashAttention、CUDA、Triton、图编译 | CS336 Lecture 6；GPU MODE 9、12、14、18、23 | 小型 softmax/GEMM/attention；先验证误差，再比较性能与访存 |
| 9. 分布式训练 | DP/TP/PP、ZeRO/FSDP、CP、通信、重计算、检查点恢复 | CS336 Lecture 7、8；GPU MODE 17、39、67、70 | CPU 多进程验证集合通信与分片；用成本模型和公开 trace 学习多 GPU 性能，真实扩展效率需多卡硬件验证 |
| 10. 高效架构与生成 | MoE、专家路由、MQA/GQA/MLA、低精度、稀疏注意力、MTP 与投机解码 | CS336 Lecture 4、10；GPU MODE 7、22、30；当前官方实现 | 小型 MoE 与草稿/目标模型实验；比较负载、接受率、有效吞吐和误差 |
| 11. 推理引擎与集群服务 | 连续批处理、分页/前缀缓存、异步调度、P/D 分离、存储与负载均衡 | GPU MODE 35、40；vLLM/SGLang 与缓存、编排项目文档 | 单 GPU 请求回放与调度模拟；区分吞吐、尾延迟和满足服务目标的有效吞吐 |
| 12. RL Infra 与复杂负载 | rollout、奖励、异步训练、权重同步、策略滞后、多模态、Agent | CS336 Lecture 17；verl、TitanRL 和当前服务架构资料 | 缩小的生成—奖励—更新闭环；记录策略版本、队列和耗时，分析任务级成本 |
| 13. 前沿协同设计 | 模型、算子、精度、通信、缓存、调度之间的相互影响 | 按技术问题选取前沿论文与官方实现 | 建立假设与基线，复现局部机制，分析硬件适用条件和端到端收益 |

阶段 7–12 根据前置知识交叉推进。CPU 和单 GPU 的实验可以验证算法与局部行为；集群扩展、RDMA 和跨机故障恢复需要另外的硬件或公开测量资料。

实验入口与环境配置见 [实验索引](../experiments/README.md)。

## 第一章：语言模型究竟在建模什么

从一句话的概率到下一个 token。

1. 从 Shannon 对语言预测与信息量的研究出发，明确 token、序列和结束符。
2. 推导序列概率的链式分解，区分恒等式与有限上下文的建模假设。
3. 用小语料估计 unigram/bigram；观察未见组合与零概率问题。
4. 从最大似然推导平均负对数似然、交叉熵和困惑度。
5. 实验比较不同上下文长度、数据量和加性平滑强度，解释泛化与存储成本。

正文见 [第一章](01-language-modeling.md)，配套 [交互演示](visuals/01-language-modeling.html)、[实验 Notebook](../experiments/01-ngram/ngram.ipynb) 和 [实验结果](experiments/01-ngram/README.md)。

## 第二章：神经语言模型

正文见 [第二章](02-neural-language-models.md)，配套 [前向计算 Notebook](../experiments/02-neural-lm/forward.ipynb)。

固定窗口 MLP → 参数共享与 embedding → softmax 与梯度 → 小模型训练 → RNN/LSTM。
