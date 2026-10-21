"""Reproducible, document-level corpus from the pinned local D2L checkout."""

from collections import defaultdict
import hashlib
from pathlib import Path
import random
import re
import subprocess

REVISION = "23d7a5aecceee57d1292c56e90cce307f183bb0a"
CHAPTERS = (
    "preliminaries", "linear-regression", "linear-classification",
    "multilayer-perceptrons", "convolutional-neural-networks",
    "recurrent-neural-networks", "recurrent-modern",
    "attention-mechanisms-and-transformers",
)
SHIFT_CHAPTER = "computational-performance"


def tokenize_markdown(text):
    text = re.split(r"(?m)^## Exercises\s*$", text)[0]
    text = re.sub(r"(?ms)^```.*?^```[^\n]*", " ", text)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$[^$\n]*\$", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"(?m)^\s*:[^\n]*|<[^>]*>|https?://\S+", " ", text)
    return re.findall(r"[a-z]+", text.lower())


def load_corpus(root, seed=42, max_words=1200):
    root = Path(root)
    if not root.exists():
        raise ValueError(f"Missing D2L checkout: {root}; see refs/README.md")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Use D2L revision {REVISION}; found {revision}")
    groups = defaultdict(list)
    hashes = set()
    for chapter in (*CHAPTERS, SHIFT_CHAPTER):
        for path in sorted((root / f"chapter_{chapter}").glob("*.md")):
            # Avoid paired scratch/concise implementation pages and chapter indexes.
            if path.stem == "index" or path.stem.endswith(("-scratch", "-concise")):
                continue
            raw = path.read_bytes()
            words = tokenize_markdown(raw.decode("utf-8"))[:max_words]
            if len(words) < 200:
                continue
            digest = hashlib.sha256(" ".join(words).encode()).hexdigest()
            if digest in hashes:
                continue
            hashes.add(digest)
            groups[chapter].append({"path": path.relative_to(root).as_posix(),
                                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                                    "tokens_sha256": digest, "words": words})
    splits = {key: [] for key in ("train", "validation", "test", "topic_shift")}
    rng = random.Random(seed)
    for chapter in CHAPTERS:
        docs = groups[chapter]
        if len(docs) < 3:
            raise ValueError(f"Need at least three eligible documents in {chapter}")
        rng.shuffle(docs)
        splits["validation"].append(docs[0])
        splits["test"].append(docs[1])
        splits["train"].extend(docs[2:])
    splits["topic_shift"] = groups[SHIFT_CHAPTER]
    manifest = {"source": "https://github.com/d2l-ai/d2l-en", "revision": revision,
                "seed": seed, "max_words_per_document": max_words,
                "split_policy": "one validation and one test document per chapter; remainder training",
                "documents": {name: [{key: value for key, value in doc.items() if key != "words"}
                                      | {"word_count": len(doc["words"])} for doc in docs]
                              for name, docs in splits.items()}}
    return {name: [doc["words"] for doc in docs] for name, docs in splits.items()}, manifest
