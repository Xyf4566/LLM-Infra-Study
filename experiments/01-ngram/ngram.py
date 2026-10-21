"""Small count language models. Only the Python standard library is required."""

from collections import Counter, defaultdict
import math

BOS, EOS, UNK = "<bos>", "<eos>", "<unk>"


def vocabulary(documents, min_count=2):
    """Fit on training documents only; BOS is context-only, not an output."""
    counts = Counter(word for doc in documents for word in doc)
    return tuple(sorted({word for word, count in counts.items() if count >= min_count}
                        | {EOS, UNK}))


def encode(documents, vocab):
    known = set(vocab)
    return [[word if word in known else UNK for word in doc] for doc in documents]


def events(documents, max_context):
    """Reset history for every document; score all words and one EOS, never BOS."""
    for doc in documents:
        history = [BOS] * max_context
        for word in [*doc, EOS]:
            yield tuple(history[-max_context:]) if max_context else (), word
            history.append(word)


class NGram:
    def __init__(self, vocab, max_context=8):
        if max_context < 0 or not vocab or BOS in vocab:
            raise ValueError("Need a nonempty output vocabulary and nonnegative context length")
        self.vocab = tuple(vocab)
        self.known = set(vocab)
        self.max_context = max_context
        self.counts = [defaultdict(Counter) for _ in range(max_context + 1)]
        self.totals = [Counter() for _ in range(max_context + 1)]

    def fit(self, documents):
        self.counts = [defaultdict(Counter) for _ in range(self.max_context + 1)]
        self.totals = [Counter() for _ in range(self.max_context + 1)]
        known = set(self.vocab)
        for history, word in events(documents, self.max_context):
            if word not in known:
                raise ValueError("Encode words outside the vocabulary before fitting")
            for k in range(self.max_context + 1):
                context = history[-k:] if k else ()
                self.counts[k][context][word] += 1
                self.totals[k][context] += 1
        return self

    def probability(self, word, history, k, alpha, method):
        """alpha is TOTAL prior count for both methods, not count per token."""
        if word not in self.known:
            raise ValueError("Map out-of-vocabulary targets to UNK first")
        if not 0 <= k <= self.max_context or alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("Invalid context length or smoothing strength")
        if method not in {"uniform", "backoff"}:
            raise ValueError("Unknown smoothing method")
        prior = 1 / len(self.vocab)
        orders = range(k + 1) if method == "backoff" else [k]
        for order in orders:
            context = tuple(history[-order:]) if order else ()
            observed = self.counts[order].get(context, {})
            count = observed.get(word, 0)
            total = self.totals[order].get(context, 0)
            prior = (count + alpha * prior) / (total + alpha)
        return prior

    def evaluate(self, documents, k, alpha, method):
        total_loss = 0.0
        targets = unseen_contexts = unseen_transitions = unknowns = 0
        for history, word in events(documents, self.max_context):
            context = history[-k:] if k else ()
            counts = self.counts[k].get(context, {})
            unseen_contexts += context not in self.counts[k]
            unseen_transitions += counts.get(word, 0) == 0
            unknowns += word == UNK
            total_loss -= math.log(self.probability(word, history, k, alpha, method))
            targets += 1
        if not targets:
            raise ValueError("Cannot evaluate an empty document collection")
        nll = total_loss / targets
        return {"nll": nll, "ppl": math.exp(nll), "targets": targets,
                "unseen_context_rate": unseen_contexts / targets,
                "unseen_transition_rate": unseen_transitions / targets,
                "unk_rate": unknowns / targets}


def select_best(rows):
    """Selection reads validation NLL only; ties prefer smaller context/alpha."""
    return min(rows, key=lambda row: (row["validation"]["nll"], row["k"],
                                      row["alpha"], row["method"]))
