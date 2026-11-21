"""Shared helpers for all HW3 notebooks: data location, fixed validation split, submission checks."""
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 0.10
LABELS = ["human", "ai", "fusion"]
TEXT_COL, LABEL_COL = "text", "label"

ON_KAGGLE = os.path.exists("/kaggle/input")
WORK_DIR = "/kaggle/working" if ON_KAGGLE else "."


def find_data_dir():
    """Kaggle: competition files attached as input. Locally: ./data or ../data."""
    kaggle = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))]
    candidates = kaggle + ["data", "../data", "."]
    for d in candidates:
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    raise FileNotFoundError("train.csv not found: attach the competition data on Kaggle or put it into ./data")


def load_data(data_dir=None):
    data_dir = data_dir or find_data_dir()
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    return train, test, sample


def lang_col(df):
    return next((c for c in ["language", "lang"] if c in df.columns), None)


def split_train_val(train, seed=SEED, val_size=VAL_SIZE):
    """Fixed stratified split (label + language). Deterministic for a given seed,
    so every notebook gets exactly the same validation rows."""
    strat = train[LABEL_COL].astype(str)
    lc = lang_col(train)
    if lc:
        strat = strat + "_" + train[lc].astype(str)
    _, val_part = train_test_split(train, test_size=val_size, stratify=strat, random_state=seed)
    is_val = train["id"].isin(set(val_part["id"]))
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)


def subsample(df, n, seed=SEED):
    if n is None or n >= len(df):
        return df
    part, _ = train_test_split(df, train_size=n, stratify=df[LABEL_COL], random_state=seed)
    return part.reset_index(drop=True)


def save_submission(test_ids, test_pred, sample, path):
    """Align predictions to sample_submission order, validate, save without index."""
    pred_by_id = pd.Series(list(test_pred), index=list(test_ids))
    sub = pd.DataFrame({"id": sample["id"], "label": sample["id"].map(pred_by_id)})
    assert len(sub) == len(sample) and sub["id"].is_unique
    assert sub["label"].isin(LABELS).all(), "missing or invalid labels"
    sub.to_csv(path, index=False)
    return sub
