"""An inspectable fixed-window neural LM forward pass, using NumPy."""

import argparse
import numpy as np

TOKENS = ["我", "你", "喜欢", "研究", "数学", "物理"]
E = np.array([[1., 0.], [.8, .2], [0., 1.], [.2, .9], [.6, .7], [.5, .8]])
W1 = np.array([[.5, -.2, .1], [.1, .4, -.3], [-.2, .3, .5], [.6, .1, .2]])
B1 = np.array([0., .1, -.1])
W2 = np.array([[-.5, -.4, .1, .2, .8, .7],
               [.2, .1, -.3, .1, .4, .5],
               [0., -.1, .2, .3, .6, .8]])
B2 = np.array([-.3, -.3, -.2, -.2, .1, .1])


def forward(contexts, like_coordinate=0.0):
    """contexts: integer IDs with shape [batch, 2]; rows are feature vectors."""
    ids = np.asarray(contexts, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[1] != 2 or np.any(ids < 0) or np.any(ids >= len(TOKENS)):
        raise ValueError("Expected [batch, 2] IDs in the vocabulary")
    embedding = E.copy()
    embedding[TOKENS.index("喜欢"), 0] = like_coordinate
    embedded = embedding[ids]                       # [B, 2, 2]
    x = embedded.reshape(len(ids), 4)                # [B, 4]
    preactivation = x @ W1 + B1                     # [B, 3]
    hidden = np.tanh(preactivation)                  # [B, 3]
    logits = hidden @ W2 + B2                       # [B, 6]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return {"embedded": embedded, "x": x, "preactivation": preactivation,
            "hidden": hidden, "logits": logits, "log_probs": log_probs,
            "probs": np.exp(log_probs)}


def parameter_count():
    return sum(parameter.size for parameter in (E, W1, B1, W2, B2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", choices=TOKENS, default="我")
    parser.add_argument("--second", choices=TOKENS, default="喜欢")
    parser.add_argument("--target", choices=TOKENS, default="数学")
    parser.add_argument("--like-coordinate", type=float, default=0.)
    args = parser.parse_args()
    result = forward([[TOKENS.index(args.first), TOKENS.index(args.second)]], args.like_coordinate)
    np.set_printoptions(precision=6, suppress=True)
    print(f"Context: {args.first} {args.second}; parameters: {parameter_count()}")
    for name in ("embedded", "x", "preactivation", "hidden", "logits", "probs"):
        print(f"{name:>13} {str(result[name].shape):>10}: {result[name]}")
    print("Output order:", TOKENS)
    print("Probability sum:", result["probs"].sum())
    print("Target NLL:", -result["log_probs"][0, TOKENS.index(args.target)])


if __name__ == "__main__":
    main()
