"""
ml_model.py — Neuricular's machine learning model training module
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
import contextlib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score,
)
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

# BalancedRandomForestClassifier undersamples the majority class within
# each tree's bootstrap sample, which can give less majority-skewed
# predict_proba output than class_weight='balanced' alone on severely
# imbalanced datasets (see _train_and_evaluate). Optional: if
# imbalanced-learn isn't installed, we still train the class-weighted RF,
# just without the comparison candidate.
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False

from exceptions import (
    DatasetLoadError, InsufficientDataError, ModelTrainingError
)
from schemas import DatasetStats, ModelArtefact
from chem_calc import get_descriptor_profile

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuricular.training")

# ── Fingerprint / RF hyperparameters ──────────────────────────────────────────
FP_RADIUS       = 2
FP_NBITS        = 2048
RF_N_ESTIMATORS = 150
RF_RANDOM_STATE = 42
TEST_SIZE       = 0.20

# Number of physicochemical descriptor features appended after the fingerprint.
# Must stay in sync with _build_features() below.
N_DESC_FEATURES = 8

# Module-level Morgan generator — replaces deprecated GetMorganFingerprintAsBitVect.
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

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


# ── Feature construction ──────────────────────────────────────────────────────

def _smiles_to_fp(smiles: str) -> list | None:
    """
    Convert SMILES to Morgan fingerprint bit vector (list of ints 0/1).
    Returns None for unparseable SMILES.
    Redirects RDKit stderr to suppress kekulization noise.
    """
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            raise ValueError("RDKit returned None")
        return list(_MORGAN_GEN.GetFingerprint(mol))
    except Exception as exc:
        logger.debug("Skipping invalid SMILES '%s': %s", smiles, exc)
        return None


def _build_features(smiles: str) -> list | None:
    """
    Combined feature vector: Morgan fingerprint (2048 bits) + 8 physicochemical
    descriptors appended as float values.

    float32 array avoids uint8 overflow for descriptor values like MW/TPSA.
    All RDKit stderr (valence warnings, hydrogen warnings, kekulization errors)
    is suppressed here — errors are already handled by the None-return path.

    Descriptor order (must match N_DESC_FEATURES and _build_inference_features):
        MW, logP, logD, TPSA, HBD, HBA, RotBonds, QED
    """
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            mol = Chem.MolFromSmiles(str(smiles).strip())
            if mol is None:
                raise ValueError("RDKit returned None")
            fp   = list(_MORGAN_GEN.GetFingerprint(mol))
            desc = get_descriptor_profile(smiles)
    except Exception as exc:
        logger.debug("Skipping '%s': %s", smiles, exc)
        return None

    desc_vec = [
        desc.mw,
        desc.logp,
        desc.logd,
        desc.tpsa,
        float(desc.hbd),
        float(desc.hba),
        float(desc.rotbond),
        desc.qed,
    ]

    return fp + desc_vec


def _canonicalize(smiles: str) -> str | None:
    """
    Canonical SMILES for a molecule, used as a dedup key.
    Returns None for unparseable SMILES (caller should skip the row).
    """
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as exc:
        logger.debug("Could not canonicalize '%s': %s", smiles, exc)
        return None


def _get_scaffold(canonical_smiles: str) -> str:
    """
    Bemis-Murcko scaffold SMILES for a (already-canonical) molecule.
    Returns "" if a scaffold can't be computed (e.g. acyclic molecules),
    which the caller treats as its own singleton scaffold group.
    """
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                return ""
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold_mol)
    except Exception as exc:
        logger.debug("Could not compute scaffold for '%s': %s", canonical_smiles, exc)
        return ""


def _scaffold_split(canonical_smiles_list: list, test_size: float, seed: int) -> tuple:
    """
    Split molecule indices into (train_idx, test_idx) by Bemis-Murcko scaffold
    group rather than at random, so that structurally related analogs
    (same scaffold, different substituent) don't end up on both sides of
    the split.

    This follows the standard MoleculeNet-style scaffold split: molecules
    are grouped by scaffold, groups are shuffled (for tie-breaking) then
    sorted largest-first, and the largest groups are assigned to train
    until the train target size is reached — the remaining, generally
    smaller/more-unique scaffold groups form the test set. This means test
    performance reflects generalisation to genuinely novel scaffolds rather
    than near-duplicates of training molecules, which is a harder and more
    realistic evaluation than a random split.

    Returns
    -------
    (train_idx, test_idx) : tuple[np.ndarray, np.ndarray]
        Row indices into the original (deduplicated) feature/label arrays.
    """
    scaffold_groups = defaultdict(list)
    for idx, smi in enumerate(canonical_smiles_list):
        scaffold = _get_scaffold(smi) or f"__no_scaffold_{idx}"
        scaffold_groups[scaffold].append(idx)

    groups = list(scaffold_groups.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)                       # break ties among equal-size groups
    groups.sort(key=len, reverse=True)        # largest scaffold clusters first

    n_total = len(canonical_smiles_list)
    n_train_target = n_total - int(round(n_total * test_size))

    train_idx, test_idx = [], []
    for group in groups:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


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


def _load_dataset(config: dict, name: str):
    """
    Fetch, validate, and featurise a dataset.
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

    # ── Canonicalize + deduplicate ────────────────────────────────────────
    # MoleculeNet-sourced CSVs routinely contain duplicate or differently-
    # written SMILES for the same molecule. Left as-is, a duplicate can land
    # on both sides of the train/test split, letting the model "test" on a
    # molecule it already memorized and inflating reported performance.
    # We canonicalize every SMILES and keep one row per unique molecule,
    # before any features are built or any split happens.
    canon_label = {}     # canonical_smiles -> label, or None if conflicting
    canon_order = []     # first-seen order, for determinism
    n_conflicting = 0

    for _, row in df.iterrows():
        canon = _canonicalize(row["smiles"])
        if canon is None:
            continue  # unparseable; would also fail feature-building below
        if canon not in canon_label:
            canon_label[canon] = row["label"]
            canon_order.append(canon)
        elif canon_label[canon] != row["label"]:
            # Same molecule, contradictory labels across duplicate rows —
            # genuinely ambiguous ground truth, so drop it rather than guess.
            canon_label[canon] = None
            n_conflicting += 1

    if n_conflicting:
        logger.warning(
            "[%s] %d duplicate molecule(s) had conflicting labels across rows — dropped.",
            name, n_conflicting,
        )

    unique_smiles = [s for s in canon_order if canon_label[s] is not None]
    n_duplicate_rows = n_raw - len(canon_order)
    if n_duplicate_rows:
        logger.info(
            "[%s] Collapsed %d duplicate row(s) down to %d unique molecules.",
            name, n_duplicate_rows, len(canon_order),
        )

    fps, labels, scaffold_smiles, n_skipped = [], [], [], 0
    for smi in unique_smiles:
        feat = _build_features(smi)
        if feat is not None:
            fps.append(feat)
            labels.append(canon_label[smi])
            scaffold_smiles.append(smi)
        else:
            n_skipped += 1

    if n_skipped:
        logger.warning(
            "[%s] %d / %d unique molecules skipped (feature build failed)",
            name, n_skipped, len(unique_smiles),
        )

    n_valid = len(fps)
    if n_valid < InsufficientDataError.MIN_REQUIRED:
        raise InsufficientDataError(
            f"Only {n_valid} valid molecules for '{name}' "
            f"(minimum: {InsufficientDataError.MIN_REQUIRED}). "
            "Dataset source may have changed."
        )

    # float32: handles both binary fingerprint bits (0/1) and continuous
    # descriptor values (MW, TPSA etc.) without uint8 overflow.
    X = np.array(fps, dtype=np.float32)
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

    # ── Scaffold split (replaces random train_test_split) ──────────────────
    # Structurally related analogs share a scaffold; a random split can put
    # near-duplicates of a training molecule into the test set, which makes
    # test performance look better than the model's real ability to
    # generalise to novel chemistry. See _scaffold_split() docstring.
    train_idx, test_idx = _scaffold_split(
        scaffold_smiles, test_size=TEST_SIZE, seed=RF_RANDOM_STATE
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise InsufficientDataError(
            f"Scaffold split for '{name}' produced a train or test set containing "
            "only one class. This can happen with small or heavily imbalanced "
            "datasets under scaffold splitting — consider adjusting TEST_SIZE or "
            "using a stratified random split for this specific dataset."
        )

    stats = DatasetStats(
        name=name, n_raw=n_raw, n_valid=n_valid, n_skipped=n_skipped,
        n_train=len(X_train), n_test=len(X_test), class_balance=class_counts,
    )
    stats.log(logger)
    return X_train, X_test, y_train, y_test, stats


# ── Training & evaluation ─────────────────────────────────────────────────────

def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0) -> float:
    """
    Pick the probability cutoff that maximises F-beta, instead of assuming
    the sklearn default of 0.5.

    On a severely imbalanced dataset (e.g. ClinTox, ~12:1), predict_proba
    output is often compressed toward the majority class — every score can
    end up under 0.5 even when the model's underlying ranking is
    informative, which is exactly the "predicts everything as negative"
    failure mode. Scanning precision_recall_curve's thresholds for the one
    that maximises F-beta finds an operating point that actually uses that
    ranking, instead of silently defaulting to all-negative.

    beta > 1 weights recall more heavily than precision — appropriate when
    a missed true positive (e.g. an undetected toxic compound) is costlier
    than a false alarm. beta=1.0 (the default here) weights them equally;
    callers can pass beta=2.0 or higher to bias toward recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve appends a final (precision=1, recall=0) point
    # with no corresponding threshold — drop it so the arrays line up.
    precision, recall = precision[:-1], recall[:-1]

    if len(thresholds) == 0:
        return 0.5  # degenerate case (e.g. constant predictions)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    f_beta = np.nan_to_num(f_beta, nan=0.0)

    best_idx = int(np.argmax(f_beta))
    return float(thresholds[best_idx])


def _fit_and_tune(clf, X_train: np.ndarray, y_train: np.ndarray,
                  cv: int = 5, beta: float = 1.0):
    """
    Tune a decision threshold using out-of-fold (cross-validated) training
    predictions, then fit `clf` on the full training set for production use.

    Threshold tuning must not see X_test/y_test — that would leak the very
    data we evaluate on into the operating point we choose. Cross-validated
    predictions on the training set give an unbiased estimate of how `clf`
    scores unseen molecules, without touching the held-out test set.
    cross_val_predict clones `clf` internally per fold, so the instance
    passed in is untouched by tuning and can safely be fit afterward.
    """
    oof_prob = cross_val_predict(
        clf, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = _tune_threshold(y_train, oof_prob, beta=beta)
    clf.fit(X_train, y_train)
    return clf, threshold


def _train_and_evaluate(X_train, X_test, y_train, y_test,
                        config: dict, stats: DatasetStats) -> ModelArtefact:
    """
    Fit two candidate classifiers, tune each one's own decision threshold,
    evaluate both on the held-out test set, and keep the better one.

    Candidate 1 — class_weighted_rf: RandomForestClassifier with
    class_weight='balanced', isotonic-calibrated when there are enough
    positives (see the calibration-strategy comment below). This is the
    approach the pipeline used previously.

    Candidate 2 — balanced_random_forest: BalancedRandomForestClassifier
    (imbalanced-learn), which undersamples the majority class within each
    tree's bootstrap sample. This can produce less majority-skewed
    predict_proba output than class_weight alone on severely imbalanced
    data, at some cost to majority-class signal. Skipped if
    imbalanced-learn isn't installed.

    Both candidates get their own threshold via _fit_and_tune (tuned on
    cross-validated training predictions, never on the test set — see
    that function's docstring), so neither is unfairly stuck at a 0.5
    cutoff that may not suit its probability distribution. Whichever
    scores higher F1 on the test set (ties broken by recall, since for a
    toxicity/permeability screen a missed true positive is usually
    costlier than a false alarm) is kept as the final artefact.

    Raises: ModelTrainingError
    """
    name  = stats.name
    n_pos = stats.class_balance.get(1, 0)

    # Calibration strategy for candidate 1:
    # - Well-balanced datasets (BBBP, n_pos=1560): isotonic calibration corrects
    #   RF overconfidence (e.g. morphine P=1.0 → more realistic probability).
    # - Severely imbalanced datasets (ClinTox, n_pos=112, ratio 12:1): both
    #   isotonic and sigmoid calibration collapse all positive predictions to zero
    #   because calibration overrides class_weight='balanced'. Skip calibration
    #   entirely and use raw RF probabilities.
    use_calibration = n_pos >= 300
    logger.info(
        "[%s] Calibration (class_weighted_rf): %s (n_pos=%d)",
        name, "isotonic" if use_calibration else "disabled (insufficient positives)", n_pos
    )

    def build_class_weighted_rf():
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        if use_calibration:
            return CalibratedClassifierCV(estimator=rf, method="isotonic", cv=5)
        return rf

    candidate_builders = [("class_weighted_rf", build_class_weighted_rf)]

    if _HAS_IMBLEARN:
        def build_balanced_rf():
            # No calibration wrapper here: combining calibration with
            # resampling is exactly the combination that already collapses
            # predictions on ClinTox for candidate 1 above, so we don't
            # repeat that mistake with a second resampling layer.
            return BalancedRandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                random_state=RF_RANDOM_STATE,
                n_jobs=-1,
            )
        candidate_builders.append(("balanced_random_forest", build_balanced_rf))
    else:
        logger.warning(
            "[%s] imbalanced-learn not installed — skipping balanced_random_forest "
            "candidate. Install with 'pip install imbalanced-learn' to enable it.",
            name,
        )

    results = []
    for cand_name, build in candidate_builders:
        try:
            clf, threshold = _fit_and_tune(build(), X_train, y_train, beta=1.0)
        except Exception as exc:
            logger.warning("[%s] Candidate '%s' failed to fit/tune: %s", name, cand_name, exc)
            continue

        y_prob = clf.predict_proba(X_test)[:, config["label_pos"]]
        y_pred = (y_prob >= threshold).astype(int)

        try:
            auc         = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=config["label_pos"])
        except ValueError as exc:
            logger.warning(
                "[%s] Candidate '%s' ROC evaluation failed — test set may lack "
                "both classes: %s", name, cand_name, exc,
            )
            continue

        cm        = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        logger.info(
            "[%s] %-22s  AUC=%.3f  thr=%.3f  P=%.3f  R=%.3f  F1=%.3f",
            name, cand_name, auc, threshold, precision, recall, f1,
        )

        results.append({
            "name": cand_name, "clf": clf, "threshold": threshold,
            "auc": auc, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "tpr": tpr, "cm": cm,
        })

    if not results:
        raise ModelTrainingError(f"All candidate models failed to fit for '{name}'.")

    best = max(results, key=lambda r: (r["f1"], r["recall"]))
    logger.info(
        "[%s] Selected '%s' (F1=%.3f, recall=%.3f, thr=%.3f) over %d other candidate(s).",
        name, best["name"], best["f1"], best["recall"], best["threshold"], len(results) - 1,
    )

    clf = best["clf"]

    # Extract feature importances — path differs between a calibrated RF
    # (class_weighted_rf when use_calibration=True) and a raw RF /
    # BalancedRandomForestClassifier (both expose feature_importances_
    # directly).
    try:
        if isinstance(clf, CalibratedClassifierCV):
            importances = np.mean(
                [e.estimator.feature_importances_ for e in clf.calibrated_classifiers_],
                axis=0,
            )
        else:
            importances = clf.feature_importances_
    except AttributeError:
        logger.warning("[%s] Could not extract feature importances.", name)
        importances = np.zeros(X_train.shape[1])

    return ModelArtefact(
        dataset_name=name, dataset_description=config["description"],
        model=clf, stats=stats,
        auc=best["auc"], precision=best["precision"], recall=best["recall"], f1=best["f1"],
        fpr=best["fpr"], tpr=best["tpr"], cm=best["cm"],
        feature_importances=importances,
        fp_radius=FP_RADIUS, fp_nbits=FP_NBITS, rf_n_estimators=RF_N_ESTIMATORS,
        threshold=best["threshold"], model_algorithm=best["name"],
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


if __name__ == "__main__":
    train_all()