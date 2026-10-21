# 学习篇章

按学习路径组织 LLM Infra 学习篇章，保存核心概念、数学推导和实验。

篇章按学习顺序命名，例如 `01-language-modeling.md`；引用的参考资料链接到 `../refs/` 中的记录或固定版本的上游文件。

## 阅读入口

- [学习路径与本地实验规划](00-learning-path.md)
- [第一章：语言模型究竟在建模什么](01-language-modeling.md)（已完成）
- [第一章交互演示](visuals/01-language-modeling.html)
- [第一章实验图表](experiments/01-ngram/index.html) · [实验结果](experiments/01-ngram/README.md) · [复现代码](../experiments/01-ngram/README.md)
- [第一章实验 Notebook](../experiments/01-ngram/ngram.ipynb)

## 文档维护约定

- 正文与可视化只保留学习路径中的核心概念、推导、关键示例和实验；即时问答、纠错补充和延伸话题留在对话中，不逐轮写入正文。
- 保留必要的数学条件，省去重复的用途说明与限定措辞。后续章节的主题按学习路径展开。
- Markdown 保存正文、参考资料和小节状态；使用 Mermaid 表达静态结构。
- HTML 按需补充交互演示，优先采用无外部依赖、可离线打开的单文件。
- 实验优先使用 Jupyter Notebook，按讲解、代码、输出、图表组织；保存运行结果，提供可修改参数的单元或控件。`.py` 模块复用核心计算，Markdown 保留核心推导与结论。
- 每完成一个核心小节，再整理对应正文；仅在有助于理解核心内容时更新可视化。未执行的实验保留计划状态。
- 一章结束时，检查关键概念、练习和待解问题，再标记完成。文档更新不自动触发 Git 提交或推送。
