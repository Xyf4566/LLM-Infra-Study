"""Run the chapter 1 experiment and generate reproducible tables and figures."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform

from corpus import load_corpus
from ngram import BOS, EOS, UNK, NGram, encode, select_best, vocabulary

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = [0, 1, 2, 4, 8]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
METHODS = ["uniform", "backoff"]
LABELS = {"uniform": "Uniform prior", "backoff": "Recursive interpolation"}


def plot_results(rows, model, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "llm-infra-chapter01"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, method in enumerate(METHODS):
        fixed = [row for row in rows if row["method"] == method and row["alpha"] == 1]
        ax = axes[0, col]
        for split, color in [("train", "#2961c7"), ("validation", "#b96616")]:
            ax.plot(CONTEXTS, [row[split]["nll"] for row in fixed], "o-", color=color,
                    label=split.title())
        ax.set(title=f"{LABELS[method]} (alpha = 1)", xlabel="Context length k",
               ylabel="Mean NLL (nats/token)", xticks=CONTEXTS, ylim=(0, 9))
        ax.grid(alpha=0.2)
        ax.legend()
        values = np.array([[next(row["validation"]["nll"] for row in rows
                                 if row["method"] == method and row["alpha"] == alpha
                                 and row["k"] == k) for k in CONTEXTS] for alpha in ALPHAS])
        ax = axes[1, col]
        chart = ax.imshow(values, cmap="YlGnBu", vmin=3, vmax=9, aspect="auto")
        ax.set(title="Validation NLL: lower is better", xlabel="Context length k",
               ylabel="Total prior count alpha", xticks=range(len(CONTEXTS)),
               xticklabels=CONTEXTS, yticks=range(len(ALPHAS)), yticklabels=ALPHAS)
        for i in range(len(ALPHAS)):
            for j in range(len(CONTEXTS)):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        color="white" if values[i, j] > 6.5 else "#17304b")
        fig.colorbar(chart, ax=ax, label="nats/token")
    fig.savefig(output / "loss-curves.png", dpi=160)
    fig.savefig(output / "loss-curves.svg")
    svg_path = output / "loss-curves.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    buckets = [("1 occurrence", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20),
               (">20", 21, float("inf"))]
    bottom = np.zeros(len(CONTEXTS))
    for (label, low, high), color in zip(buckets, ["#b96616", "#dfad6c", "#7bafc5", "#2961c7"]):
        values = np.array([sum(low <= n <= high for n in model.totals[k].values())
                           / len(model.totals[k]) for k in CONTEXTS]) * 100
        axes[0].bar(range(len(CONTEXTS)), values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0].set(title="How much evidence does each context have?", xlabel="Context length k",
                ylabel="Share of distinct training contexts (%)", xticks=range(len(CONTEXTS)),
                xticklabels=CONTEXTS, ylim=(0, 100))
    axes[0].legend(fontsize=8)
    selected = [row for row in rows if row["method"] == "backoff" and row["alpha"] == 1]
    for key, label, color in [("unseen_context_rate", "Unseen context", "#2961c7"),
                              ("unseen_transition_rate", "Unseen (context, next token)", "#b96616")]:
        axes[1].plot(CONTEXTS, [row["validation"][key] * 100 for row in selected],
                     "o-", label=label, color=color)
    axes[1].set(title="Validation positions missing from training counts", xlabel="Context length k",
                ylabel="Share of validation positions (%)", xticks=CONTEXTS, ylim=(0, 100))
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "sparsity.png", dpi=160)
    plt.close(fig)

    words = [word for word, _ in model.counts[0][()].most_common()
             if word not in {BOS, EOS, UNK}][:12]
    matrix = np.array([[model.counts[1].get((context,), {}).get(word, 0)
                        / model.totals[1][(context,)] for word in words] for context in words])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set(xticks=range(len(words)), xticklabels=words, yticks=range(len(words)),
           yticklabels=words, xlabel="Next token", ylabel="Previous token",
           title="Training bigram probabilities (12 frequent tokens shown)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "#17304b")
    fig.colorbar(im, ax=ax, label="P(next token | previous token), full-vocabulary denominator")
    fig.savefig(output / "transitions.png", dpi=160)
    plt.close(fig)
    return {"matplotlib": matplotlib.__version__, "numpy": np.__version__}


def write_report(result, output):
    best = result["selected"]
    method_cn = {"uniform": "均匀先验", "backoff": "递归插值"}
    baseline = next(row for row in result["grid"] if row["method"] == "uniform"
                    and row["k"] == 0 and row["alpha"] == 1)
    lines = ["# 第一章实验：上下文长度、平滑与泛化", "",
             "语料来自本地 D2L 英文讲义；按 Markdown 文档划分，固定种子 42。", "",
             "## 实验设置", "",
             "| 集合 | 文档数 | 计分 token 数（含 EOS） | UNK 比例 |", "| --- | ---: | ---: | ---: |"]
    for name, stats in result["data"].items():
        lines.append(f"| {name} | {stats['documents']} | {stats['targets']} | {stats['unk_rate']:.2%} |")
    lines += ["", f"词表大小：{result['vocabulary_size']}。仅保留训练中出现至少 2 次的词；其他词映射为 UNK。",
              "每篇取清理后的前 1,200 个英文词，保留至少 200 词的文档；去掉代码、公式、图片和练习。",
              "每个文档重置 BOS，上下文最长为 8；计分覆盖全部词和一个 EOS。",
              "语料属于同一本教材，结果反映教材内部的文档泛化。topic_shift 是额外的计算性能主题讲义。", "",
              "## 平滑方式", "",
              "两种方法都使用 `(count + alpha * prior) / (context_count + alpha)`，alpha 是虚拟计数总量。",
              "均匀先验使用 `prior = 1 / V`；递归插值使用短一阶的预测，最底层同样使用均匀先验。", "",
              "## 训练与验证", "",
              "以下固定 alpha = 1；图中同时列出全部 alpha 的验证损失。", "",
              "| 方法 | 上下文 k | 训练 NLL | 验证 NLL | 验证未见上下文 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["grid"]:
        if row["alpha"] == 1:
            lines.append(f"| {method_cn[row['method']]} | {row['k']} | {row['train']['nll']:.3f} | "
                         f"{row['validation']['nll']:.3f} | {row['validation']['unseen_context_rate']:.1%} |")
    lines += ["", "![训练验证曲线与平滑强度](loss-curves.png)", "",
              "![上下文计数与未见组合](sparsity.png)", "", "## 验证集选择与最终评估", "",
              f"本次网格中，验证集选中：**{method_cn[best['method']]}，k = {best['k']}，alpha = {best['alpha']:g}**。", "",
              f"验证 NLL：**{best['validation']['nll']:.3f}**；选定后测试 NLL：**{result['test']['nll']:.3f}**。",
              f"计算性能主题 NLL：**{result['topic_shift']['nll']:.3f}**。测试集与主题迁移集均不参与参数选择。", "",
              f"作为参照，alpha = 1 的 unigram 验证 NLL 为 {baseline['validation']['nll']:.3f}。",
              "UNK 将多个稀有词合并为一个预测目标；这些损失不等于完整原始词表上的损失。", "",
              "## 计数表规模", "", "| 最大上下文 k | 本阶不同上下文 | 本阶非零转移项 | 递归模型累计转移项 |",
              "| --- | ---: | ---: | ---: |"]
    for stats in result["storage"]:
        lines.append(f"| {stats['k']} | {stats['contexts']} | {stats['transitions']} | {stats['cumulative_transitions']} |")
    lines += ["", "![常见词转移概率](transitions.png)", "",
              "热力图只显示 12 个常见词，分母仍包含完整词表，因此可见行的概率之和可以小于 1。", "",
              "## 复现", "", "主要入口：[实验 Notebook](../ngram.ipynb)；运行方式见 [实验说明](../README.md)。",
              "完整数值见 [results.json](results.json)，语料文件及校验和见 [manifest.json](manifest.json)。", "",
              "数据来源：[Dive into Deep Learning](https://github.com/d2l-ai/d2l-en)，作者 Aston Zhang、Zachary C. Lipton、Mu Li、Alexander J. Smola 及贡献者。",
              f"固定版本：`{result['source_revision']}`；原文许可：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一章实验：上下文长度与平滑</title>
<style>body{{margin:0;background:#f3f6fa;color:#17304b;font:17px/1.8 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:24px}}section{{background:white;border:1px solid #dce4ed;border-radius:14px;padding:24px;margin:20px 0}}h1{{font-size:30px}}h2{{font-size:23px}}img{{width:100%;height:auto}}a{{color:#2961c7}}.metrics{{display:flex;gap:16px;flex-wrap:wrap}}.metrics div{{background:#edf3fc;padding:14px 22px;border-radius:8px}}strong{{font-size:26px}}@media(max-width:600px){{main{{padding:12px}}section{{padding:14px}}}}</style></head>
<body><main><h1>上下文越长，预测就越好吗？</h1>
<p>D2L 讲义 · {result['data']['train']['documents']} 篇训练文档 · {result['vocabulary_size']} 个 token · CPU 计数模型</p>
<p><a href="../../../docs/01-language-modeling.md">第一章正文</a> · <a href="README.md">实验设置与数据表</a></p>
<section><h2>训练拟合与验证表现</h2><p>上图固定 α = 1；下图比较不同平滑强度。NLL 越低越好。</p><a href="loss-curves.png"><img src="loss-curves.png" alt="两种平滑方法的训练验证曲线与验证损失热力图"></a></section>
<section><h2>更具体的上下文，有多少样本支持？</h2><a href="sparsity.png"><img src="sparsity.png" alt="不同上下文长度的样本数分布和验证未见组合比例"></a></section>
<section><h2>验证集选择的模型</h2><p>{method_cn[best['method']]} · k = {best['k']} · α = {best['alpha']:g}</p>
<div class="metrics"><div>验证 NLL<br><strong>{best['validation']['nll']:.3f}</strong></div><div>测试 NLL<br><strong>{result['test']['nll']:.3f}</strong></div><div>计算性能主题 NLL<br><strong>{result['topic_shift']['nll']:.3f}</strong></div></div>
<p>使用同一词表与评分规则；测试集和计算性能主题文档不参与模型选择。</p></section>
<section><h2>词语之间的转移统计</h2><a href="transitions.png"><img src="transitions.png" alt="训练语料中十二个常见词的转移概率"></a><p>图中展示词表的一个子集，概率分母包含全部候选词。</p></section>
<p><a href="../ngram.ipynb">实验 Notebook</a> · <a href="../README.md">复现说明</a> · <a href="results.json">完整结果</a> · <a href="manifest.json">语料清单</a></p>
</main></body></html>'''
    (output / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "refs/d2l-en")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/01-ngram/results")
    args = parser.parse_args()
    raw, manifest = load_corpus(args.source)
    vocab = vocabulary(raw["train"], min_count=2)
    encoded = {name: encode(docs, vocab) for name, docs in raw.items()}
    model = NGram(vocab, max_context=max(CONTEXTS)).fit(encoded["train"])
    rows = []
    for method in METHODS:
        for k in CONTEXTS:
            for alpha in ALPHAS:
                row = {"method": method, "k": k, "alpha": alpha}
                for split in ("train", "validation"):
                    row[split] = model.evaluate(encoded[split], k, alpha, method)
                rows.append(row)
            print(f"Evaluated {method}, k={k}", flush=True)
    best = select_best(rows)
    # Freeze the selected configuration before touching held-out target scores.
    configuration = {key: best[key] for key in ("k", "alpha", "method")}
    heldout = {name: model.evaluate(encoded[name], **configuration)
               for name in ("test", "topic_shift")}
    data = {}
    for name, docs in encoded.items():
        targets = sum(len(doc) + 1 for doc in docs)
        data[name] = {"documents": len(docs), "targets": targets,
                      "unk_rate": sum(doc.count(UNK) for doc in docs) / targets}
    storage = [{"k": k, "contexts": len(model.counts[k]),
                "transitions": sum(len(counts) for counts in model.counts[k].values()),
                "cumulative_transitions": sum(len(counts) for order in range(k + 1)
                                               for counts in model.counts[order].values())}
               for k in CONTEXTS]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    versions = plot_results(rows, model, output)
    result = {"source_revision": manifest["revision"], "seed": manifest["seed"],
              "python": platform.python_version(), "packages": versions,
              "vocabulary_size": len(vocab), "min_count": 2, "data": data,
              "contexts": CONTEXTS, "alphas": ALPHAS, "grid": rows,
              "selected": best, "storage": storage, **heldout}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "grid.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["method", "k", "alpha", "train_nll", "validation_nll"])
        writer.writerows([row["method"], row["k"], row["alpha"],
                          row["train"]["nll"], row["validation"]["nll"]] for row in rows)
    write_report(result, output)
    print(json.dumps({"selected": configuration, "validation_nll": best["validation"]["nll"],
                      "test_nll": heldout["test"]["nll"], "topic_shift_nll": heldout["topic_shift"]["nll"]}, indent=2))


if __name__ == "__main__":
    main()
