"""Shared helpers for all HW3 notebooks: data location, fixed validation split, submission checks."""
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 0.10
LABELS = ["human", "ai", "fusion"]
TEXT_COL, LABEL_COL = "text", "label"

WORK_DIR = "/kaggle/working"


def find_data_dir():
    """Competition files attached as Kaggle input."""
    found = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))
    if not found:
        raise FileNotFoundError("train.csv not found: attach the competition data as Kaggle input")
    return os.path.dirname(found[0])


def load_data(data_dir=None):
    """train, test and sample_submission DataFrames."""
    data_dir = data_dir or find_data_dir()
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    return train, test, sample


def lang_col(df):
    """Name of the language column, if present."""
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
    """Fixed stratified subsample of n rows (None = all rows)."""
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


def f1_by(y_true, y_pred, groups):
    """Macro-F1 per group (e.g. domain or language), plus group size."""
    from sklearn.metrics import f1_score
    df = pd.DataFrame({"y": list(y_true), "p": list(y_pred), "g": list(groups)})
    return (df.groupby("g")[["y", "p"]]
              .apply(lambda d: pd.Series({"n": len(d), "macro_f1": f1_score(d.y, d.p, average="macro")}))
              .astype({"n": int}).round(4).sort_values("macro_f1"))


def save_fig(name):
    """Save the current matplotlib figure to WORK_DIR/images (download it for the report) and show it."""
    import matplotlib.pyplot as plt
    img_dir = os.path.join(WORK_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, name), dpi=150, bbox_inches="tight")
    plt.show()


def save_proba(ids, proba, classes, name, split):
    """Write outputs/{split}_proba_{name}.csv: id + p_human, p_ai, p_fusion (fixed column order)."""
    cols = [f"p_{c}" for c in classes]
    df = pd.DataFrame(proba, columns=cols)[[f"p_{c}" for c in LABELS]]
    df.insert(0, "id", list(ids))
    out_dir = os.path.join(WORK_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{split}_proba_{name}.csv"), index=False)


def find_outputs(pattern):
    """Files written by earlier notebooks: attached as Kaggle input (notebook output) or in this session."""
    found = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    found += glob.glob(os.path.join(WORK_DIR, "outputs", pattern))
    return sorted(set(found))


def load_proba(path, ids):
    """Probabilities from a save_proba file, aligned to ids, columns in LABELS order."""
    df = pd.read_csv(path).set_index("id").loc[list(ids)]
    return df[[f"p_{c}" for c in LABELS]].to_numpy()
