# 实验 Notebooks

实验采用 **讲解 → 代码 → 输出 → 图表 → 练习** 的组织方式。Notebook 是主要阅读与交互入口，`.py` 模块保存可复用的计算逻辑。

- [第一章：n-gram、平滑与泛化](01-ngram/ngram.ipynb)
- [第二章：神经语言模型前向计算](02-neural-lm/forward.ipynb)
- [第二章：输出层反向传播](02-neural-lm/backward.ipynb)

Notebook 将讲解、代码与运行输出组织在一起；要调整参数，在 Jupyter 中重新运行单元。

## 本地启动

Python 3.11+，在仓库根目录执行：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 改用：.venv\Scripts\Activate.ps1
python -m pip install -r experiments/requirements-notebooks.txt
python -m jupyterlab
```

也可用 VS Code 的 Jupyter 扩展打开 `.ipynb`，选择安装了这些依赖的 Python 环境。第一章需要本地 `refs/d2l-en`，获取方式见 [参考资料](../refs/README.md)。

选择 **Restart Kernel and Run All Cells** 可从头复现。已有静态图表与报告继续保留，便于快速查看和导出。
