"""
schemas.py — Neuricular's data schemas for chemistry results and ML pipeline artefacts
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

# Values for the pka_status field on DescriptorProfile / CNSMPOResult.
PKA_MATCHED = "matched"                      # found in the IUPAC-derived lookup
PKA_NOT_FOUND = "not_found"                  # database loaded, no entry for this molecule
PKA_DB_UNAVAILABLE = "database_unavailable"  # lookup table missing/corrupt (deployment issue)


@dataclass
class PkaMatch:
    """
    A basic-pKa value found in the IUPAC Digitized pKa Dataset.

    Fields
    ------
    pka             : float — pKa of the conjugate acid of the most basic centre
    pka_type        : str   — source label of the chosen value (e.g. 'pKaH1')
    n_values        : int   — how many repeated measurements were combined (median)
    pka_min/pka_max : float — spread of those measurements
    assessment      : str   — best IUPAC reliability rating among them
                              ('Reliable' | 'Approximate' | 'Uncertain' | 'Unrated')
    source          : str   — citation label for display
    """
    pka:        float
    pka_type:   str
    n_values:   int
    pka_min:    float
    pka_max:    float
    assessment: str
    source:     str = "IUPAC Digitized pKa Dataset v2.4a"


@dataclass
class DescriptorProfile:
    """
    All standard molecular descriptors computed for one compound.

    Lipinski and Veber rule checks are derived automatically at construction
    from the raw descriptor values so callers never recompute them.

    logD and pKa depend on an experimental pKa from the IUPAC-derived
    lookup. There is no estimation fallback: when no entry is found both are
    None, and pka_status says why.

    Fields
    ------
    smiles     : str   — input SMILES (stored for traceability)
    mw         : float — molecular weight (Da)
    logp       : float — Wildman-Crippen logP (octanol-water partition)
    logd       : float | None — logD at pH 7.4 from logP and the database pKa
    tpsa       : float — topological polar surface area (Å²)
    hbd        : int   — H-bond donors
    hba        : int   — H-bond acceptors
    rotbond    : int   — rotatable bonds
    qed        : float — quantitative estimate of drug-likeness (0–1)
    pka_basic  : float | None — experimental most-basic pKa, or None
    pka_match  : PkaMatch | None — provenance of pka_basic
    pka_status : str   — PKA_MATCHED | PKA_NOT_FOUND | PKA_DB_UNAVAILABLE
    """
    smiles:     str
    mw:         float
    logp:       float
    logd:       Optional[float]
    tpsa:       float
    hbd:        int
    hba:        int
    rotbond:    int
    qed:        float
    pka_basic:  Optional[float]
    pka_match:  Optional[PkaMatch] = None
    pka_status: str = PKA_NOT_FOUND
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
    The published score is the sum of six 0–1 desirability terms, with
    >= 4.0 (of 6) considered CNS-optimised.

    logD and pKa terms are only scored when the IUPAC-derived lookup has a
    pKa for the molecule (logD needs the pKa to correct for ionisation).
    When it doesn't, both terms are left out and the score is taken over the
    four remaining properties (MW, logP, TPSA, HBD). In that case:
      - max_total is 4.0 instead of 6.0
      - cns_optimised applies the same proportion as 4.0/6 (i.e. total >= 2/3
        of max_total), so 2.67/4 plays the role of 4.0/6
    Such a score is a partial assessment and is not directly comparable to a
    full six-term score; the UI and explanations say so.

    score_* fields hold the piecewise-linear desirability value (0–1);
    raw_* fields hold the descriptor value fed into it, for display/audit.
    score_logd / score_pka / raw_logd / raw_pka are None when unavailable.

    Properties
    ----------
    per_property      : dict[str, float] — contribution per *scored* property
    raw_values        : dict[str, float] — raw value per *scored* property
    failed_properties : list[str]        — scored properties contributing < 1.0
    missing_properties: list[str]        — properties left out (e.g. logD, pKa)
    pka_available     : bool
    """
    smiles:        str
    score_mw:      float
    score_logp:    float
    score_tpsa:    float
    score_hbd:     float
    score_logd:    Optional[float]
    score_pka:     Optional[float]
    total:         float
    max_total:     float
    cns_optimised: bool
    # Raw descriptor values (audit trail)
    raw_mw:        float
    raw_logp:      float
    raw_tpsa:      float
    raw_hbd:       int
    raw_logd:      Optional[float]
    raw_pka:       Optional[float]
    pka_match:     Optional[PkaMatch] = None
    pka_status:    str = PKA_NOT_FOUND

    @property
    def pka_available(self) -> bool:
        return self.score_pka is not None

    @property
    def per_property(self) -> dict[str, float]:
        terms = {
            "MW":   self.score_mw,
            "logP": self.score_logp,
            "logD": self.score_logd,
            "TPSA": self.score_tpsa,
            "HBD":  self.score_hbd,
            "pKa":  self.score_pka,
        }
        return {k: v for k, v in terms.items() if v is not None}

    @property
    def raw_values(self) -> dict[str, float | int]:
        raws = {
            "MW":   self.raw_mw,
            "logP": self.raw_logp,
            "logD": self.raw_logd,
            "TPSA": self.raw_tpsa,
            "HBD":  self.raw_hbd,
            "pKa":  self.raw_pka,
        }
        scored = self.per_property
        return {k: v for k, v in raws.items() if k in scored}

    @property
    def failed_properties(self) -> list[str]:
        """Scored properties that contribute less than 1.0 (not fully optimal)."""
        return [k for k, v in self.per_property.items() if v < 1.0]

    @property
    def missing_properties(self) -> list[str]:
        """Properties left out of the score because they couldn't be computed."""
        return [k for k in ("logD", "pKa") if k not in self.per_property]


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
    # Decision threshold for predict >= threshold => positive class, tuned
    # per-model (see ml_model._tune_threshold) instead of assuming 0.5.
    # Severely imbalanced datasets can produce predict_proba output that
    # never crosses 0.5 for the minority class even when the underlying
    # ranking is informative — a fixed 0.5 cutoff then predicts the
    # majority class for every input. threshold captures the cutoff that
    # was actually validated for this artefact.
    threshold:           float = 0.5
    # Which candidate classifier (see ml_model._train_and_evaluate) was
    # selected for this artefact, e.g. "class_weighted_rf" or
    # "balanced_random_forest" — kept for traceability/debugging.
    model_algorithm:     str = "class_weighted_rf"

    def summary(self) -> str:
        return (
            f"{self.dataset_name}  "
            f"[{self.model_algorithm}]  "
            f"AUC={self.auc:.3f}  "
            f"P={self.precision:.3f}  "
            f"R={self.recall:.3f}  "
            f"F1={self.f1:.3f}  "
            f"thr={self.threshold:.3f}"
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
        threshold:  float = 0.5,
    ) -> "PredictionResult":
        """
        Construct a PredictionResult from a raw probability, deriving
        the predicted class, confidence band, and label automatically.

        threshold : the decision cutoff for predicted vs. not (defaults to
            0.5, but callers going through ml_predict.py pass the
            per-model tuned ModelArtefact.threshold instead — see
            ml_model._tune_threshold for why a fixed 0.5 can be wrong for
            imbalanced datasets). Note the confidence band below is still
            computed from the raw probability's distance from 0.5, not
            from `threshold` — it describes how certain the model's score
            is in absolute terms, independent of where we chose to act on it.
        """
        predicted = prob >= threshold

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
