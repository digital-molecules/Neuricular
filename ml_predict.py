"""
ml_predict.py — MetricularPro
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
from chem_calc import get_descriptor_profile
from ml_model import _smiles_to_fp

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

def _predict(smiles: str, artefact: ModelArtefact,
             pos_label: str, neg_label: str) -> PredictionResult:
    """
    Core inference routine shared by all model-specific wrappers.

    Raises
    ------
    InvalidSMILESError  — SMILES cannot be parsed (propagated from chem_calc)
    PredictionError     — fingerprint length mismatch or sklearn inference error
    """
    fp_list = get_morgan_fp_array(smiles)
    desc = get_descriptor_profile(smiles)

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

    X = np.array(fp_list + desc_vec, dtype=np.float32).reshape(1, -1)

    if len(fp_list) != artefact.fp_nbits:
        raise PredictionError(
            f"Fingerprint length mismatch: molecule has {len(fp_list)} bits "
            f"but model '{artefact.dataset_name}' expects {artefact.fp_nbits}. "
            "Ensure FP_NBITS is consistent between ml_model.py and ml_predict.py."
        )

    X = np.array(fp_list, dtype=np.uint8).reshape(1, -1)
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

def _build_features(smiles: str):
    fp = get_morgan_fp_array(smiles)
    desc = get_descriptor_profile(smiles)

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

# ── CNS reference drug panel ──────────────────────────────────────────────────
#
# Structured data driving the Reference Drugs tab in app.py.
# Each entry documents the BBB mechanism explicitly so the UI can distinguish
# mechanistically interesting model failures from noise.

CNS_REFERENCE_DRUGS = [
    {
        "name":                  "Donepezil",
        "smiles":                "COc1cc2c(cc1OC)CC(CC(=O)c1ccccc1)C2",
        "indication":            "Alzheimer's disease — AChE inhibitor",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "Passive transcellular diffusion",
        "expected_model_agrees": True,
        "discussion_point": (
            "Lipophilic, low-polarity scaffold — passive diffusion is well-captured "
            "by fingerprint models. Expected true positive."
        ),
    },
    {
        "name":                  "Sertraline",
        "smiles":                "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
        "indication":            "Depression — SSRI",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "Passive transcellular diffusion",
        "expected_model_agrees": True,
        "discussion_point": (
            "Classic lipophilic amine. Must reach serotonin transporters in the CNS. "
            "Model should predict high permeability — good positive control."
        ),
    },
    {
        "name":                  "Clozapine",
        "smiles":                "CN1CCN(c2nc3ccccc3nc2Cl)CC1",
        "indication":            "Schizophrenia — atypical antipsychotic",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "Passive transcellular diffusion",
        "expected_model_agrees": True,
        "discussion_point": (
            "High CNS penetration underpins both its efficacy and its haematological "
            "side-effect profile. Reliable benchmark for a true positive."
        ),
    },
    {
        "name":                  "Levodopa", "category": "Anti-Parkinsonian",
        "smiles":                "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
        "indication":            "Parkinson's disease — dopamine precursor",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "LAT1 active transporter — not passive diffusion",
        "expected_model_agrees": False,
        "discussion_point": (
            "CONFIRMED FAILURE CASE (P ≈ 0.36, moderate confidence): Levodopa is polar "
            "and zwitterionic — its catechol + amino acid scaffold carries fingerprint "
            "bits strongly associated with non-permeable compounds in the BBBP training "
            "set. The model confidently predicts low permeability. In reality, levodopa "
            "crosses the BBB via the LAT1 large amino acid transporter. "
            "Compare with gabapentin: also a LAT1 substrate, but correctly predicted "
            "as permeable because its cyclohexane ring contributes lipophilic bits that "
            "resemble passive diffusion scaffolds. Levodopa has no such compensating "
            "structural features. This is the cleanest example in this panel of a "
            "mechanism-invisible failure — and the most scientifically interesting "
            "talking point for the poster."
        ),
    },
    {
        "name":                  "Memantine",
        "smiles":                "CC12CC(CC(C1)(CN)C)(C2)N",
        "indication":            "Alzheimer's disease — NMDA antagonist",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "Passive transcellular diffusion",
        "expected_model_agrees": True,
        "discussion_point": (
            "Adamantane scaffold contributes strong lipophilicity; designed for rapid "
            "CNS penetration. Useful structural contrast with the polar levodopa case."
        ),
    },
    {
        "name":                  "Valproic acid",
        "smiles":                "CCCC(CCC)C(=O)O",
        "indication":            "Epilepsy / bipolar — mood stabiliser",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "Passive transcellular diffusion (MCT contribution reported)",
        "expected_model_agrees": True,
        "discussion_point": (
            "Small, lipophilic carboxylic acid. Low CNS MPO score likely due to "
            "sparse fingerprint (few bits set) rather than true structural liability. "
            "Illustrates limitations of MPO for very small molecules."
        ),
    },
    {
        "name":                  "Caffeine",
        "smiles":                "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "indication":            "CNS stimulant — adenosine antagonist",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "Passive transcellular diffusion",
        "expected_model_agrees": True,
        "discussion_point": (
            "Textbook CNS compound and positive control. Low MW, moderate logP, "
            "low TPSA — sits comfortably within CNS MPO space."
        ),
    },
    {
        "name":                  "Atenolol",
        "smiles":                "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "indication":            "Hypertension — peripheral beta-blocker",
        "known_bbb_permeable":   False,
        "bbb_mechanism":         "Excluded by high polarity and P-gp efflux; intentionally peripheral",
        "expected_model_agrees": True,
        "discussion_point": (
            "Deliberately designed NOT to cross the BBB to avoid CNS side effects "
            "(fatigue, depression). High TPSA and HBD count. "
            "Good true-negative control — model should correctly predict low permeability."
        ),
    },
    {
        "name":                  "Morphine", "category": "Opioid Analgesic",
        "smiles":                "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
        "indication":            "Opioid analgesic — MOR agonist",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "Partial passive diffusion; P-gp substrate limits CNS exposure",
        "expected_model_agrees": True,
        "discussion_point": (
            "OVERCONFIDENCE CASE (P = 1.0, high confidence): The model predicts morphine "
            "as BBB-permeable with maximum probability — the highest in this entire panel. "
            "This is technically correct, but the certainty is misleading. In reality, "
            "morphine is a P-glycoprotein (P-gp) efflux substrate, meaning a significant "
            "fraction of molecules that cross the BBB are actively pumped back out, "
            "substantially limiting free CNS exposure relative to more lipophilic opioids "
            "like fentanyl or oxycodone. The BBBP dataset uses a binary permeable/not-permeable "
            "label that cannot encode this nuance — morphine is labelled permeable, the model "
            "learned that confidently, and likely reinforced it by seeing structurally related "
            "opioid scaffolds throughout the training set. A probability of 1.0 does not mean "
            "the model understands the pharmacology; it means the morphine scaffold is "
            "overrepresented or highly consistent in the training data. This is a case where "
            "model confidence and pharmacological reality diverge in a clinically relevant way."
        ),
    },
    {
        "name":                  "Gabapentin", "category": "Anticonvulsant",
        "smiles":                "NCC1(CC(=O)O)CCCCC1",
        "indication":            "Epilepsy / neuropathic pain — voltage-gated Ca²⁺ channel modulator",
        "known_bbb_permeable":   True,
        "bbb_mechanism":         "LAT1 transporter — same mechanism as levodopa",
        "expected_model_agrees": True,
        "discussion_point": (
            "Interesting contrast to levodopa: gabapentin is also a LAT1 substrate, yet "
            "the model correctly predicts BBB permeability (P ≈ 0.93, high confidence). "
            "This is likely because gabapentin's cyclohexane scaffold contributes enough "
            "lipophilic fingerprint bits to resemble passively-permeable compounds in the "
            "training set, even though its actual mechanism is active transport. "
            "The correct prediction here is for the wrong structural reason — a useful "
            "reminder that model accuracy does not imply mechanistic understanding."
        ),
    },
]
