"""
schemas.py — MetricularPro
===========================
All dataclasses and structured result types used across the pipeline.

Keeping data schemas in one file means:
- The shape of every result is visible in one place.
- Modules that only need to *read* a result don't have to import the module
  that *produces* it (avoiding circular imports).
- Type annotations elsewhere in the codebase can import from here cleanly.

Dependencies: stdlib only (dataclasses, typing) + numpy for array fields.
No RDKit, no sklearn — this file has no heavy imports.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Chemistry result schemas ──────────────────────────────────────────────────

@dataclass
class DescriptorProfile:
    """
    All standard molecular descriptors computed for one compound.

    Lipinski and Veber rule checks are derived automatically at construction
    from the raw descriptor values so callers never recompute them.

    Fields
    ------
    smiles     : str   — input SMILES (stored for traceability)
    mw         : float — molecular weight (Da)
    logp       : float — Wildman-Crippen logP (octanol-water partition)
    logd       : float — estimated logD at pH 7.4 (nitrogen-count heuristic)
    tpsa       : float — topological polar surface area (Å²)
    hbd        : int   — H-bond donors
    hba        : int   — H-bond acceptors
    rotbond    : int   — rotatable bonds
    qed        : float — quantitative estimate of drug-likeness (0–1)
    pka_basic  : float — estimated most-basic pKa (structural heuristic)
    """
    smiles:     str
    mw:         float
    logp:       float
    logd:       float
    tpsa:       float
    hbd:        int
    hba:        int
    rotbond:    int
    qed:        float
    pka_basic:  float
    # Derived rule checks — set by __post_init__, not passed by caller
    lipinski:   bool = field(init=False)
    veber:      bool = field(init=False)
    both_rules: bool = field(init=False)

    def __post_init__(self):
        self.lipinski   = (
            self.mw < 500
            and self.hbd  <= 5
            and self.hba  <= 10
            and self.logp < 5
        )
        self.veber      = self.rotbond <= 10 and self.tpsa <= 140
        self.both_rules = self.lipinski and self.veber


@dataclass
class CNSMPOResult:
    """
    Per-property CNS MPO contributions and aggregated score.

    Reference: Wager et al., ACS Chem. Neurosci. 2010, 1, 435–449.
    Score ≥ 4.0 is considered CNS-optimised per the original publication.

    Each score_* field holds the piecewise-linear desirability value (0–1)
    for that property. raw_* fields hold the actual computed descriptor value
    used as input to the desirability function, preserved for display and audit.

    Properties
    ----------
    per_property   : dict[str, float] — {property_name: contribution}
    raw_values     : dict[str, float] — {property_name: raw_descriptor_value}
    failed_properties : list[str]     — properties contributing < 1.0
    """
    smiles:        str
    score_mw:      float
    score_logp:    float
    score_logd:    float
    score_tpsa:    float
    score_hbd:     float
    score_pka:     float
    total:         float
    cns_optimised: bool
    # Raw descriptor values (audit trail)
    raw_mw:        float
    raw_logp:      float
    raw_logd:      float
    raw_tpsa:      float
    raw_hbd:       int
    raw_pka:       float

    @property
    def per_property(self) -> dict[str, float]:
        return {
            "MW":   self.score_mw,
            "logP": self.score_logp,
            "logD": self.score_logd,
            "TPSA": self.score_tpsa,
            "HBD":  self.score_hbd,
            "pKa":  self.score_pka,
        }

    @property
    def raw_values(self) -> dict[str, float | int]:
        return {
            "MW":   self.raw_mw,
            "logP": self.raw_logp,
            "logD": self.raw_logd,
            "TPSA": self.raw_tpsa,
            "HBD":  self.raw_hbd,
            "pKa":  self.raw_pka,
        }

    @property
    def failed_properties(self) -> list[str]:
        """Properties that contribute less than 1.0 (not fully optimal)."""
        return [k for k, v in self.per_property.items() if v < 1.0]


# ── ML pipeline schemas ───────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    """
    Summary statistics for one dataset after loading and fingerprint generation.
    Persisted inside ModelArtefact so the UI can display training provenance.

    Fields
    ------
    name          : str  — dataset key (e.g. 'bbbp', 'clintox')
    n_raw         : int  — rows in the raw CSV before any filtering
    n_valid       : int  — molecules with successfully parsed fingerprints
    n_skipped     : int  — molecules dropped due to invalid SMILES
    n_train       : int  — molecules in training split
    n_test        : int  — molecules in held-out test split
    class_balance : dict — {class_label: count} for the full valid set
    """
    name:          str
    n_raw:         int
    n_valid:       int
    n_skipped:     int
    n_train:       int
    n_test:        int
    class_balance: dict

    def log(self, logger) -> None:
        """Log a one-line summary using the provided logger."""
        logger.info(
            "[%s] raw=%d  valid=%d  skipped=%d  train=%d  test=%d  "
            "balance={0: %d, 1: %d}",
            self.name,
            self.n_raw, self.n_valid, self.n_skipped,
            self.n_train, self.n_test,
            self.class_balance.get(0, 0),
            self.class_balance.get(1, 0),
        )


@dataclass
class ModelArtefact:
    """
    Everything persisted to disk for one trained classifier.

    Includes the fitted model, training provenance (DatasetStats),
    full evaluation metrics on the held-out test set, and the fingerprint
    hyperparameters used at training time.

    The fp_radius and fp_nbits fields are checked at inference time
    (in ml_predict.py) to catch configuration drift between training runs.

    Fields
    ------
    dataset_name        : str                  — dataset key
    dataset_description : str                  — human-readable citation string
    model               : RandomForestClassifier
    stats               : DatasetStats
    auc                 : float                — ROC-AUC on test set
    precision           : float
    recall              : float
    f1                  : float
    fpr                 : np.ndarray           — for ROC curve plotting
    tpr                 : np.ndarray
    cm                  : np.ndarray           — 2×2 confusion matrix
    feature_importances : np.ndarray           — Gini importances (2048-dim)
    fp_radius           : int                  — Morgan radius used at training
    fp_nbits            : int                  — fingerprint length used at training
    rf_n_estimators     : int
    """
    dataset_name:        str
    dataset_description: str
    model:               object          # RandomForestClassifier; typed as object to avoid
                                         # importing sklearn here (no heavy deps in schemas)
    stats:               DatasetStats
    auc:                 float
    precision:           float
    recall:              float
    f1:                  float
    fpr:                 np.ndarray
    tpr:                 np.ndarray
    cm:                  np.ndarray
    feature_importances: np.ndarray
    fp_radius:           int = 2
    fp_nbits:            int = 2048
    rf_n_estimators:     int = 150

    def summary(self) -> str:
        return (
            f"{self.dataset_name}  "
            f"AUC={self.auc:.3f}  "
            f"P={self.precision:.3f}  "
            f"R={self.recall:.3f}  "
            f"F1={self.f1:.3f}"
        )


# ── Prediction result schema ──────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    The output of a single model inference call.

    Confidence bands
    ----------------
    'high'     : P ≥ 0.75 or P ≤ 0.25  — model is confident
    'moderate' : P ≥ 0.62 or P ≤ 0.38
    'low'      : P near 0.5             — model is uncertain; treat with caution

    Fields
    ------
    smiles      : str   — input molecule
    probability : float — P(positive class), i.e. P(permeable) or P(toxic)
    predicted   : bool  — True = positive class predicted
    confidence  : str   — 'high' | 'moderate' | 'low'
    label       : str   — human-readable verdict string
    model_name  : str   — dataset key of the model that produced this result
    """
    smiles:      str
    probability: float
    predicted:   bool
    confidence:  str
    label:       str
    model_name:  str

    @classmethod
    def from_prob(
        cls,
        smiles:     str,
        prob:       float,
        model_name: str,
        pos_label:  str,
        neg_label:  str,
    ) -> "PredictionResult":
        """
        Construct a PredictionResult from a raw probability, deriving
        the predicted class, confidence band, and label automatically.
        """
        predicted = prob >= 0.5

        if prob >= 0.75 or prob <= 0.25:
            confidence = "high"
        elif prob >= 0.62 or prob <= 0.38:
            confidence = "moderate"
        else:
            confidence = "low"

        return cls(
            smiles      = smiles,
            probability = round(prob, 4),
            predicted   = predicted,
            confidence  = confidence,
            label       = pos_label if predicted else neg_label,
            model_name  = model_name,
        )
