"""
ml_predict.py — Neuricular's runtime prediction helpers for the Streamlit app
==============================
Runtime prediction helpers loaded by the Streamlit app.

All exception types are imported from exceptions.py.
All result types are imported from schemas.py.
No class definitions live in this file.
"""

import logging
import pickle

import numpy as np

from chem_calc import get_morgan_fp_array
from exceptions import InvalidSMILESError, ModelLoadError, PredictionError
from schemas import ModelArtefact, PredictionResult

logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(path: str) -> ModelArtefact:
    """
    Load and validate a ModelArtefact from disk.

    Validates that all expected fields are present and the classifier
    supports predict_proba, catching version-mismatch issues early.

    Raises: ModelLoadError
    """
    try:
        with open(path, "rb") as f:
            artefact = pickle.load(f)
    except FileNotFoundError:
        raise ModelLoadError(
            f"Model file '{path}' not found. "
            "Run 'python ml_model.py' to train the models before launching the app."
        )
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as exc:
        raise ModelLoadError(
            f"Model file '{path}' could not be loaded: {exc}. "
            "The file may be corrupt or was built with an incompatible version of "
            "scikit-learn. Re-run 'python ml_model.py'."
        )

    required = {
        "model", "auc", "cm", "fpr", "tpr",
        "feature_importances", "stats",
        "dataset_name", "dataset_description",
        "fp_nbits", "fp_radius",
    }
    missing = required - set(vars(artefact).keys())
    if missing:
        raise ModelLoadError(
            f"Artefact '{path}' is missing fields: {missing}. "
            "Re-run 'python ml_model.py' to regenerate."
        )

    if not hasattr(artefact.model, "predict_proba"):
        raise ModelLoadError(
            f"Classifier in '{path}' does not support predict_proba. "
            "Expected a RandomForestClassifier."
        )

    logger.info(
        "Loaded '%s' (AUC=%.3f, n_train=%d)",
        artefact.dataset_name, artefact.auc, artefact.stats.n_train,
    )
    return artefact


# ── Inference ─────────────────────────────────────────────────────────────────

def _build_inference_features(smiles: str) -> np.ndarray:
    """
    Build the combined feature vector for inference — must exactly match
    the vector built by ml_model._build_features() at training time:
        [Morgan fingerprint (2048 bits)] + [8 physicochemical descriptors]
        = 2056 features total, dtype float32.

    Descriptor order: MW, logP, logD, TPSA, HBD, HBA, RotBonds, QED

    Raises
    ------
    InvalidSMILESError  — SMILES cannot be parsed
    PredictionError     — descriptor calculation failed
    """
    from chem_calc import get_descriptor_profile

    fp_list = get_morgan_fp_array(smiles)   # raises InvalidSMILESError if bad

    try:
        desc = get_descriptor_profile(smiles)
    except Exception as exc:
        raise PredictionError(
            f"Descriptor calculation failed for '{smiles}': {exc}"
        ) from exc

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

    return np.array(fp_list + desc_vec, dtype=np.float32).reshape(1, -1)


def _predict(smiles: str, artefact: ModelArtefact,
             pos_label: str, neg_label: str) -> PredictionResult:
    """
    Core inference routine shared by all model-specific wrappers.

    Raises
    ------
    InvalidSMILESError  — SMILES cannot be parsed (propagated from chem_calc)
    PredictionError     — feature construction or sklearn inference error
    """
    X = _build_inference_features(smiles)

    expected = artefact.fp_nbits + 8   # 2048 fingerprint bits + 8 descriptors
    if X.shape[1] != expected:
        raise PredictionError(
            f"Feature vector length mismatch: built {X.shape[1]} features "
            f"but model '{artefact.dataset_name}' expects {expected}. "
            "Re-run ml_model.py to retrain with the current feature set."
        )

    try:
        prob = float(artefact.model.predict_proba(X)[0][1])
    except Exception as exc:
        raise PredictionError(
            f"Inference failed for '{smiles}' on model '{artefact.dataset_name}': {exc}"
        ) from exc

    return PredictionResult.from_prob(
        smiles=smiles, prob=prob, model_name=artefact.dataset_name,
        pos_label=pos_label, neg_label=neg_label,
    )


def predict_bbbp(smiles: str, artefact: ModelArtefact) -> PredictionResult:
    """Predict BBB permeability. P(positive) = P(BBB-permeable)."""
    return _predict(smiles, artefact,
                    pos_label="BBB-permeable",
                    neg_label="Not BBB-permeable")


def predict_clintox(smiles: str, artefact: ModelArtefact) -> PredictionResult:
    """Predict clinical toxicity. P(positive) = P(toxic in trials)."""
    return _predict(smiles, artefact,
                    pos_label="Likely toxic in clinical trials",
                    neg_label="Likely safe in clinical trials")


def get_top_features(artefact: ModelArtefact, n: int = 20) -> tuple:
    """Return (indices, importances) for the top-n most important fingerprint bits."""
    importances = artefact.feature_importances
    top_idx     = np.argsort(importances)[::-1][:n]
    return top_idx, importances[top_idx]


# ── CNS reference drug panel ──────────────────────────────────────────────────
#
# SMILES are sourced exclusively from cns_drugs.py (single source of truth).
# Pharmacological annotations (known BBB status, model agreement expectation,
# discussion points) are maintained here as a lookup dict keyed by drug name,
# then merged at runtime into CNS_REFERENCE_DRUGS.
#
# To add a new reference drug: add it to cns_drugs.py first, then add its
# annotation entry to _REFERENCE_ANNOTATIONS below.

_REFERENCE_ANNOTATIONS = {
    "Donepezil": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "Lipophilic, low-polarity scaffold — passive diffusion is well-captured "
            "by fingerprint models. Expected true positive."
        ),
    },
    "Sertraline": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "Classic lipophilic amine. Must reach serotonin transporters in the CNS. "
            "Model should predict high permeability — good positive control."
        ),
    },
    "Clozapine": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "High CNS penetration underpins both its efficacy and its haematological "
            "side-effect profile. Reliable benchmark for a true positive."
        ),
    },
    "Levodopa": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": False,
        "discussion_point": (
            "CONFIRMED FAILURE CASE (P ≈ 0.36, moderate confidence): Levodopa is polar "
            "and zwitterionic. A fingerprint model predicts low permeability, yet it "
            "crosses the BBB via the LAT1 large amino acid transporter. "
            "Compare with gabapentin: also a LAT1 substrate, but correctly predicted "
            "as permeable because its cyclohexane ring contributes lipophilic bits that "
            "resemble passive diffusion scaffolds. Levodopa has no such compensating "
            "structural features — the cleanest example in this panel of a "
            "mechanism-invisible failure."
        ),
    },
    "Memantine": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "Adamantane scaffold contributes strong lipophilicity; designed for rapid "
            "CNS penetration. Useful structural contrast with the polar levodopa case."
        ),
    },
    "Valproic acid": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "Small, lipophilic carboxylic acid. Low CNS MPO score likely due to "
            "sparse fingerprint (few bits set) rather than true structural liability. "
            "Illustrates limitations of MPO for very small molecules."
        ),
    },
    "Caffeine": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "Textbook CNS compound and positive control. Low MW, moderate logP, "
            "low TPSA — sits comfortably within CNS MPO space."
        ),
    },
    "Atenolol": {
        "known_bbb_permeable":   False,
        "expected_model_agrees": True,
        "discussion_point": (
            "Deliberately designed NOT to cross the BBB to avoid CNS side effects "
            "(fatigue, depression). High TPSA and HBD count. "
            "Good true-negative control — model should correctly predict low permeability."
        ),
    },
    "Morphine": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "OVERCONFIDENCE CASE (P = 1.0, high confidence): The model predicts morphine "
            "as BBB-permeable with maximum probability — the highest in the entire panel. "
            "This is technically correct, but the certainty is misleading. Morphine is a "
            "P-glycoprotein (P-gp) efflux substrate, substantially limiting free CNS "
            "exposure relative to more lipophilic opioids like fentanyl. The BBBP dataset "
            "uses a binary label that cannot encode this nuance. P = 1.0 does not mean the "
            "model understands the pharmacology — it means the opioid scaffold is "
            "overrepresented or highly consistent in the training data."
        ),
    },
    "Gabapentin": {
        "known_bbb_permeable":   True,
        "expected_model_agrees": True,
        "discussion_point": (
            "Interesting contrast to levodopa: gabapentin is also a LAT1 substrate, yet "
            "the model correctly predicts BBB permeability (P ≈ 0.93, high confidence). "
            "Likely because its cyclohexane scaffold contributes lipophilic fingerprint "
            "bits resembling passively-permeable compounds in the training set. "
            "The correct prediction here is for the wrong structural reason — a useful "
            "reminder that model accuracy does not imply mechanistic understanding."
        ),
    },
}

# Build CNS_REFERENCE_DRUGS by merging cns_drugs.py SMILES with annotations above.
# Only drugs that have an entry in _REFERENCE_ANNOTATIONS are included,
# ensuring the reference panel stays curated rather than using all 90 drugs.
def _build_reference_panel() -> list:
    from cns_drugs import CNS_DRUG_DATABASE
    db_by_name = {d["name"]: d for d in CNS_DRUG_DATABASE}
    panel = []
    for name, annotation in _REFERENCE_ANNOTATIONS.items():
        if name not in db_by_name:
            logger.warning(
                "Reference drug '%s' is in _REFERENCE_ANNOTATIONS but not in "
                "cns_drugs.CNS_DRUG_DATABASE — skipping.", name
            )
            continue
        entry = {**db_by_name[name], **annotation}
        panel.append(entry)
    return panel

CNS_REFERENCE_DRUGS = _build_reference_panel()

