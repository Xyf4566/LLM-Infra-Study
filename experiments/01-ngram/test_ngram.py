import copy
import math
import unittest

from corpus import tokenize_markdown
from ngram import BOS, EOS, UNK, NGram, encode, events, select_best, vocabulary


class CountModelTests(unittest.TestCase):
    def setUp(self):
        self.documents = [["a", "b"], ["a", "c"]]
        self.vocab = vocabulary(self.documents, min_count=1)
        self.model = NGram(self.vocab, max_context=2).fit(self.documents)

    def test_document_boundaries_and_targets(self):
        observed = list(events(self.documents, 2))
        self.assertEqual(len(observed), 6)
        self.assertEqual(observed[0], ((BOS, BOS), "a"))
        self.assertEqual(observed[3], ((BOS, BOS), "a"))
        self.assertEqual(sum(word == EOS for _, word in observed), 2)
        self.assertNotIn((EOS,), self.model.counts[1])
        self.assertEqual(self.model.totals[0][()], 6)

    def test_hand_calculated_smoothing(self):
        # V=5, C(a)=2, C(a,b)=1, alpha=2: (1+2/5)/(2+2).
        p = self.model.probability("b", [BOS, "a"], 1, 2, "uniform")
        self.assertAlmostEqual(p, 0.35)
        # Unigram b: (1+2/5)/(6+2)=0.175; then (1+2*0.175)/4.
        p = self.model.probability("b", [BOS, "a"], 1, 2, "backoff")
        self.assertAlmostEqual(p, 0.3375)

    def test_positive_normalized_distributions(self):
        for method in ("uniform", "backoff"):
            for k in range(3):
                for history in [(BOS, BOS), (BOS, "a"), ("unseen", "history")]:
                    probs = [self.model.probability(word, history, k, 0.1, method)
                             for word in self.vocab]
                    self.assertTrue(all(p > 0 for p in probs))
                    self.assertAlmostEqual(sum(probs), 1)

    def test_unseen_context_reverts_to_prior(self):
        for word in self.vocab:
            p = self.model.probability(word, ["unseen", "history"], 2, 1, "backoff")
            self.assertAlmostEqual(p, self.model.probability(word, [], 0, 1, "backoff"))
            uniform = self.model.probability(word, ["unseen", "history"], 2, 1, "uniform")
            self.assertAlmostEqual(uniform, 1 / len(self.vocab))

    def test_evaluation_does_not_change_counts(self):
        before = copy.deepcopy(self.model.counts)
        self.model.evaluate([["c", "b", UNK]], 2, 1, "backoff")
        self.assertEqual(before, self.model.counts)

    def test_vocab_and_unknowns_use_training_only(self):
        vocab = vocabulary([["a", "a", "b"]], min_count=2)
        self.assertEqual(set(vocab), {"a", EOS, UNK})
        self.assertEqual(encode([["a", "new", "b"]], vocab), [["a", UNK, UNK]])
        with self.assertRaises(ValueError):
            self.model.probability("new", [], 0, 1, "uniform")

    def test_unigram_nll_matches_manual_calculation(self):
        # Training has six targets: a=2, b=1, c=1, EOS=2; prior total=5.
        result = self.model.evaluate([["a", "b"]], 0, 5, "uniform")
        expected = -(math.log(3 / 11) + math.log(2 / 11) + math.log(3 / 11)) / 3
        self.assertAlmostEqual(result["nll"], expected)
        self.assertEqual(result["targets"], 3)

    def test_every_context_scores_same_positions(self):
        for k in range(3):
            self.assertEqual(self.model.evaluate(self.documents, k, 1, "backoff")["targets"], 6)

    def test_selection_ignores_training_and_test(self):
        rows = [dict(k=1, alpha=1, method="backoff", validation={"nll": 2},
                     train={"nll": 1}, test={"nll": 0.1}),
                dict(k=2, alpha=1, method="backoff", validation={"nll": 1},
                     train={"nll": 5}, test={"nll": 10})]
        self.assertEqual(select_best(rows)["k"], 2)

    def test_markdown_cleaning(self):
        text = "# Hello\nUse [words](https://example.org) and $x^2$.\n```python\nsecret_code\n```\n## Exercises\nnot included"
        self.assertEqual(tokenize_markdown(text), ["hello", "use", "words", "and"])


if __name__ == "__main__":
    unittest.main()
