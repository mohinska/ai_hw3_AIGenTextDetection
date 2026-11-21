"""Text features shared by EDA and models: TF-IDF, stylometry, chunking of long texts for transformers."""
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF = {
    "word_1-2": dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=300_000,
                     sublinear_tf=True, lowercase=True),
    "char_wb_2-5": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=300_000,
                        sublinear_tf=True, lowercase=False),
}


def fit_tfidf(names, train_texts):
    """Fit the named TF-IDF vectorizers on train texts only."""
    return {n: TfidfVectorizer(dtype=np.float32, **TFIDF[n]).fit(train_texts) for n in names}


def tfidf_transform(vecs, texts):
    return hstack([v.transform(texts) for v in vecs.values()]).tocsr()


def tfidf_feature_names(vecs):
    return np.concatenate([[f"{n}:{f}" for f in v.get_feature_names_out()] for n, v in vecs.items()])


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"\w+", re.UNICODE)


def style_features(t):
    """Cheap stylometric statistics of one text (dict of floats)."""
    words = WORD.findall(t)
    n_w = max(len(words), 1)
    sents = [s for s in SENT_SPLIT.split(t) if s.strip()]
    sent_len = np.array([len(WORD.findall(s)) for s in sents]) if sents else np.array([0])
    n_c = max(len(t), 1)
    return {
        "type_token_ratio": len(set(w.lower() for w in words[:300])) / min(n_w, 300),
        "mean_word_len": np.mean([len(w) for w in words]) if words else 0,
        "mean_sent_len": sent_len.mean(),
        "sent_len_cv": sent_len.std() / sent_len.mean() if sent_len.mean() > 0 else 0,
        "punct_per_100c": 100 * sum(ch in ",;:()\"'-–—" for ch in t) / n_c,
        "uppercase_ratio": sum(ch.isupper() for ch in t) / n_c,
        "digit_ratio": sum(ch.isdigit() for ch in t) / n_c,
        "newlines_per_100w": 100 * t.count("\n") / n_w,
        "non_ascii_ratio": sum(ord(ch) > 127 for ch in t) / n_c,
    }


def chunk_ids(texts, tokenizer, max_len=512, stride=64, max_chunks=None):
    """Split each text into overlapping windows of max_len tokens (special tokens included).
    Returns (input_ids, text_idx): one entry per chunk, text_idx maps the chunk to its text.
    max_chunks keeps only the first k chunks of a text (limits compute on very long texts)."""
    enc = tokenizer(list(texts), truncation=True, max_length=max_len, stride=stride,
                    return_overflowing_tokens=True)
    text_idx = np.array(enc["overflow_to_sample_mapping"])
    ids = enc["input_ids"]
    if max_chunks is not None:
        rank = np.zeros(len(text_idx), dtype=int)
        for i in range(1, len(text_idx)):
            rank[i] = rank[i - 1] + 1 if text_idx[i] == text_idx[i - 1] else 0
        keep = rank < max_chunks
        ids, text_idx = [x for x, k in zip(ids, keep) if k], text_idx[keep]
    return ids, text_idx


def mean_by_text(chunk_probs, text_idx, n_texts):
    """Average chunk-level class probabilities back to one row per text."""
    out = np.zeros((n_texts, chunk_probs.shape[1]))
    np.add.at(out, text_idx, chunk_probs)
    return out / np.bincount(text_idx, minlength=n_texts)[:, None]
