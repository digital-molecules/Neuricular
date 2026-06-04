"""
ml_model.py — MetricularPro
============================
Train and persist Random Forest classifiers for BBBP and ClinTox datasets.

Run once before launching the Streamlit app:
    python ml_model.py

Domain exceptions and data schemas are imported from exceptions.py and
schemas.py respectively; no class definitions live in this file.

Set LOG_LEVEL=DEBUG environment variable for verbose per-molecule output.
"""

import logging
import os
import pickle
import sys
import urllib.request
import gzip
import io

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import CalibratedClassifierCV 
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from exceptions import (
    DatasetLoadError, InsufficientDataError, ModelTrainingError,
)
from schemas import DatasetStats, ModelArtefact
from chem_calc import get_descriptor_profile

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("metricularpro.training")

# ── Fingerprint / RF hyperparameters ──────────────────────────────────────────
FP_RADIUS       = 2
FP_NBITS        = 2048
RF_N_ESTIMATORS = 150
RF_RANDOM_STATE = 42
TEST_SIZE       = 0.20

# ── Dataset configuration ─────────────────────────────────────────────────────
DATASETS = {
    "bbbp": {
        "urls": [
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
            "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/BBBP.csv",
            "https://github.com/deepchem/deepchem/raw/master/datasets/BBBP.csv",
        ],
        "smiles_col":  "smiles",
        "label_col":   "p_np",
        "label_pos":   1,
        "description": "Blood-Brain Barrier Permeability (Martins et al. 2012)",
        "output_path": "bbbp_model.pkl",
        "local_path":  "BBBP.csv",
    },
    "clintox": {
        "urls": [
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
            "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/clintox.csv",
            "https://github.com/deepchem/deepchem/raw/master/datasets/clintox.csv",
        ],
        "smiles_col":  "smiles",
        "label_col":   "CT_TOX",
        "label_pos":   1,
        "description": "Clinical Toxicity — FDA trial failures (Gayvert et al. 2016)",
        "output_path": "clintox_model.pkl",
        "local_path":  "clintox.csv",
    },
}

_UA = "Mozilla/5.0 (compatible; MetricularPro/1.0)"


# ── Data loading ──────────────────────────────────────────────────────────────

def _fetch_dataframe(config: dict, name: str) -> pd.DataFrame:
    """
    Try each URL in config["urls"] in order, then fall back to a local file.
    Uses a browser-like User-Agent to avoid 403s from GitHub/S3.

    Raises: DatasetLoadError if all remote URLs fail and no local file exists.
    """
    for url in config["urls"]:
        try:
            logger.debug("[%s] Trying %s", name, url)
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
            if url.endswith(".gz"):
                raw = gzip.decompress(raw)
            df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
            logger.info("[%s] Downloaded %d rows from %s", name, len(df), url)
            return df
        except Exception as exc:
            logger.warning("[%s] URL failed (%s): %s", name, url, exc)

    local = config.get("local_path", "")
    if local and os.path.isfile(local):
        logger.info("[%s] Using local file '%s'", name, local)
        return pd.read_csv(local)

    raise DatasetLoadError(
        f"All download URLs failed for '{name}' and no local file '{local}' found.\n"
        "Download manually:\n"
        "  BBBP:    https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv\n"
        "  ClinTox: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz\n"
        "Place the file(s) in the same directory as ml_model.py and re-run."
    )


def _smiles_to_fp(smiles: str) -> list | None:
    """Convert SMILES to Morgan fingerprint. Returns None (logs warning) on failure."""
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            raise ValueError("RDKit returned None")
        return list(rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, FP_RADIUS, nBits=FP_NBITS
        ))
    except Exception as exc:
        logger.debug("Skipping invalid SMILES '%s': %s", smiles, exc)
        return None


def _load_dataset(config: dict, name: str):
    """
    Fetch, validate, and fingerprint a dataset.
    Returns (X_train, X_test, y_train, y_test, DatasetStats).

    Raises: DatasetLoadError, InsufficientDataError
    """
    try:
        df = _fetch_dataframe(config, name)
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(f"Unexpected error loading '{name}': {exc}") from exc

    required = {config["smiles_col"], config["label_col"]}
    missing  = required - set(df.columns)
    if missing:
        raise DatasetLoadError(
            f"Dataset '{name}' missing columns: {missing}. Found: {list(df.columns)}"
        )

    df = df[[config["smiles_col"], config["label_col"]]].dropna()
    df.columns = ["smiles", "label"]
    n_raw = len(df)

    try:
        df["label"] = df["label"].astype(int)
    except (ValueError, TypeError) as exc:
        raise DatasetLoadError(
            f"Label column '{config['label_col']}' cannot be coerced to int: {exc}"
        ) from exc

    valid_labels = df["label"].isin([0, 1])
    n_bad_labels = (~valid_labels).sum()
    if n_bad_labels:
        logger.warning("[%s] Dropping %d rows with labels outside {0,1}", name, n_bad_labels)
    df = df[valid_labels]

    fps, labels, n_skipped = [], [], 0

    for _, row in df.iterrows():
        feat = _build_features(row["smiles"])

        if feat is not None:
            fps.append(feat)
            labels.append(row["label"])
        else:
            n_skipped += 1

    if n_skipped:
        logger.warning("[%s] %d / %d molecules skipped (bad SMILES)", name, n_skipped, n_raw)

    n_valid = len(fps)
    if n_valid < InsufficientDataError.MIN_REQUIRED:
        raise InsufficientDataError(
            f"Only {n_valid} valid molecules for '{name}' "
            f"(minimum: {InsufficientDataError.MIN_REQUIRED}). "
            "Dataset source may have changed."
        )

    X = np.array(fps, dtype=np.uint8)
    y = np.array(labels, dtype=np.int32)

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    if len(class_counts) < 2:
        raise InsufficientDataError(
            f"Dataset '{name}' has only one class after filtering — cannot train."
        )

    ratio = max(class_counts.values()) / min(class_counts.values())
    if ratio > 10:
        logger.warning(
            "[%s] Severe class imbalance (%.1f:1). class_weight='balanced' applied.",
            name, ratio,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RF_RANDOM_STATE, stratify=y
    )

    stats = DatasetStats(
        name=name, n_raw=n_raw, n_valid=n_valid, n_skipped=n_skipped,
        n_train=len(X_train), n_test=len(X_test), class_balance=class_counts,
    )
    stats.log(logger)
    return X_train, X_test, y_train, y_test, stats


# ── Training & evaluation ─────────────────────────────────────────────────────

def _train_and_evaluate(X_train, X_test, y_train, y_test,
                        config: dict, stats: DatasetStats) -> ModelArtefact:
    """
    Fit RandomForest and compute full evaluation metrics on the held-out test set.
    Returns a ModelArtefact (schemas.py).

    Raises: ModelTrainingError
    """
    name = stats.name
    logger.info("[%s] Fitting RandomForest (n_estimators=%d) …", name, RF_N_ESTIMATORS)
    try:

        rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        )

        clf = CalibratedClassifierCV(
        estimator=rf,
        method="isotonic",
        cv=5
        )

        clf.fit(X_train, y_train)
    except Exception as exc:
        raise ModelTrainingError(f"Fit failed for '{name}': {exc}") from exc

    y_prob = clf.predict_proba(X_test)[:, config["label_pos"]]
    y_pred = clf.predict(X_test)

    try:
        auc      = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=config["label_pos"])
    except ValueError as exc:
        raise ModelTrainingError(
            f"ROC evaluation failed for '{name}' — test set may lack both classes: {exc}"
        ) from exc

    cm        = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    logger.info("[%s] AUC=%.3f  P=%.3f  R=%.3f  F1=%.3f", name, auc, precision, recall, f1)

    return ModelArtefact(
        dataset_name=name, dataset_description=config["description"],
        model=clf, stats=stats,
        auc=auc, precision=precision, recall=recall, f1=f1,
        fpr=fpr, tpr=tpr, cm=cm,
        feature_importances=clf.feature_importances_,
        fp_radius=FP_RADIUS, fp_nbits=FP_NBITS, rf_n_estimators=RF_N_ESTIMATORS,
    )


def _save_artefact(artefact: ModelArtefact, path: str) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(artefact, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved → %s", path)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot write '{path}': {exc}. Check directory write permissions."
        ) from exc


# ── Entry point ───────────────────────────────────────────────────────────────

def train_all() -> None:
    errors = []
    for name, config in DATASETS.items():
        logger.info("=" * 60)
        logger.info("Dataset: %s", config["description"])
        logger.info("=" * 60)
        try:
            X_train, X_test, y_train, y_test, stats = _load_dataset(config, name)
            artefact = _train_and_evaluate(X_train, X_test, y_train, y_test, config, stats)
            _save_artefact(artefact, config["output_path"])
            logger.info("✓ %s", artefact.summary())
        except (DatasetLoadError, InsufficientDataError, ModelTrainingError) as exc:
            logger.error("✗ '%s': %s", name, exc)
            errors.append((name, exc))
        except Exception as exc:
            logger.exception("✗ Unexpected error for '%s'", name)
            errors.append((name, exc))

    if errors:
        logger.error("%d model(s) failed:", len(errors))
        for name, exc in errors:
            logger.error("  • %s: %s", name, exc)
        sys.exit(1)
    logger.info("All models trained successfully.")

def _build_features(smiles: str):
    """
    Combined feature vector:
    - Morgan fingerprint (2048)
    - Physicochemical descriptors (6 values)
    """

    fp = _smiles_to_fp(smiles)
    if fp is None:
        return None

    desc = get_descriptor_profile(smiles)

    # Numeric descriptors (important for BBB + transporter edge cases)
    desc_vec = [
        desc.mw,
        desc.logp,
        desc.logd,
        desc.tpsa,
        desc.hbd,
        desc.hba,
        desc.rotbond,
        desc.qed,
    ]

    return fp + desc_vec

if __name__ == "__main__":
    train_all()
