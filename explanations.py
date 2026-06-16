"""
explanations.py — Neuricular's dynamic explanation generator for ML predictions and CNS MPO scoring
================================
Dynamic, molecule-specific explanations for ML and scoring results.

Each function receives a typed result object (from schemas.py) and returns
a structured explanation dict ready for rendering in the Streamlit UI.
No RDKit, no sklearn — pure logic operating on already-computed values.

Design principle: explanations are *specific*, not generic. They reference
the actual numerical values and flag the specific properties or confidence
levels that drove the result, so the user learns something about *this*
molecule rather than reading boilerplate.
"""

from __future__ import annotations
from schemas import CNSMPOResult, PredictionResult


# ── CNS MPO explanations ──────────────────────────────────────────────────────

def explain_cns_mpo(result: CNSMPOResult) -> dict:
    """
    Generate a molecule-specific explanation for a CNS MPO result.

    Returns
    -------
    dict with keys:
        headline    : str  — one-sentence summary
        paragraphs  : list[str] — per-property commentary (only for non-ideal props)
        optimisation: list[str] — concrete structural suggestions
        severity    : str  — 'good' | 'moderate' | 'poor'
    """
    score = result.total
    raw   = result.raw_values
    props = result.per_property

    if score >= 4.5:
        severity = "good"
        headline = (
            f"Strong CNS profile (MPO {score:.2f}/6). "
            "This molecule satisfies most criteria for CNS drug-likeness."
        )
    elif score >= 4.0:
        severity = "good"
        headline = (
            f"Acceptable CNS profile (MPO {score:.2f}/6) — just above the optimised threshold. "
            "Minor structural refinements could push the score higher."
        )
    elif score >= 2.5:
        severity = "moderate"
        headline = (
            f"Moderate CNS profile (MPO {score:.2f}/6). "
            "The molecule has notable liabilities that reduce predicted CNS optimisation."
        )
    else:
        severity = "poor"
        headline = (
            f"Poor CNS profile (MPO {score:.2f}/6). "
            "Multiple properties fall outside the CNS-optimised range — significant "
            "structural redesign would be needed for CNS drug development."
        )

    paragraphs = []
    optimisation = []

    # MW
    if props["MW"] < 1.0:
        mw = raw["MW"]
        if mw > 500:
            paragraphs.append(
                f"Molecular weight ({mw:.0f} Da) exceeds the ideal CNS ceiling of 360 Da "
                f"and even the Lipinski limit of 500 Da. High MW reduces passive membrane "
                f"permeability and increases the likelihood of P-gp recognition."
            )
            optimisation.append("Remove non-essential substituents or fragment the molecule.")
        elif mw > 360:
            paragraphs.append(
                f"Molecular weight ({mw:.0f} Da) is above the ideal CNS ceiling of 360 Da. "
                "While still within Lipinski space, CNS drugs benefit from lower MW to "
                "maximise free CNS exposure."
            )
            optimisation.append(
                "Consider bioisosteric replacements that reduce MW without losing target affinity."
            )

    # logP
    if props["logP"] < 1.0:
        lp = raw["logP"]
        if lp > 5:
            paragraphs.append(
                f"logP ({lp:.2f}) is above 5, indicating high lipophilicity. "
                "While lipophilicity aids membrane permeation, excessive logP correlates with "
                "poor aqueous solubility, high plasma protein binding, and off-target toxicity."
            )
            optimisation.append(
                "Reduce lipophilicity by replacing alkyl chains with heteroatoms, "
                "adding a polar group, or reducing aromatic ring count."
            )
        elif lp > 3:
            paragraphs.append(
                f"logP ({lp:.2f}) is moderately above the ideal CNS range (≤ 3). "
                "Slightly elevated lipophilicity increases non-specific binding risk."
            )
            optimisation.append("Minor polarity increase (e.g. fluorine substitution) may help.")
        elif lp < 0:
            paragraphs.append(
                f"logP ({lp:.2f}) is negative, indicating the molecule is very hydrophilic. "
                "Very low logP typically means poor passive membrane permeability and "
                "reliance on active transport to cross the BBB."
            )
            optimisation.append(
                "Consider prodrug strategies or lipophilic bioisosteres to improve membrane crossing."
            )

    # logD
    if props["logD"] < 1.0:
        ld = raw["logD"]
        if ld > 4:
            paragraphs.append(
                f"logD at pH 7.4 ({ld:.2f}) is high, meaning the molecule remains very "
                "lipophilic even after ionisation correction. This increases risk of "
                "hERG binding and promiscuous CNS side effects."
            )
        elif ld > 2:
            paragraphs.append(
                f"logD at pH 7.4 ({ld:.2f}) is above the ideal CNS range of ≤ 2. "
                "logD better reflects in vivo partitioning than logP for ionisable compounds."
            )
            optimisation.append(
                "Adjust pKa of basic groups (e.g. via N-methylation or ring incorporation) "
                "to lower effective logD at physiological pH."
            )
        elif ld < 0:
            paragraphs.append(
                f"logD at pH 7.4 ({ld:.2f}) is negative. At physiological pH the molecule "
                "is predominantly ionised, which sharply limits passive BBB crossing."
            )

    # TPSA
    if props["TPSA"] < 1.0:
        tpsa = raw["TPSA"]
        if tpsa > 120:
            paragraphs.append(
                f"TPSA ({tpsa:.1f} Å) is very high. TPSA above 90 Å begins to impede "
                "passive CNS penetration; above 120 Å it is strongly predictive of poor "
                "BBB permeability. High TPSA usually reflects multiple polar groups (OH, NH, C=O)."
            )
            optimisation.append(
                "Replace H-bond donors with their N-methyl or O-methyl analogues, "
                "or cyclise polar chains to reduce exposed polar surface area."
            )
        elif tpsa > 90:
            paragraphs.append(
                f"TPSA ({tpsa:.1f} Å) exceeds the ideal CNS window of 40–90 Å. "
                "Each additional polar atom above this range incrementally reduces "
                "passive transcellular permeability."
            )
            optimisation.append("Reducing HBD count is often the most effective way to lower TPSA.")
        elif tpsa < 20:
            paragraphs.append(
                f"TPSA ({tpsa:.1f} Å) is very low. While this favours membrane crossing, "
                "extremely low TPSA can indicate a molecule that is too lipophilic and "
                "may have poor aqueous solubility or high non-specific binding."
            )

    # HBD
    if props["HBD"] < 1.0:
        hbd = raw["HBD"]
        if hbd >= 3:
            paragraphs.append(
                f"H-bond donors ({hbd}) significantly exceed the ideal CNS value of ≤ 1. "
                "Each H-bond donor incurs an energetic desolvation penalty when crossing "
                "the lipid bilayer, making BBB penetration progressively harder."
            )
            optimisation.append(
                f"Capping {hbd - 1} of the {hbd} H-bond donors (e.g. OH → OMe, NH → NMe) "
                "would substantially improve predicted CNS permeability."
            )
        elif hbd == 2:
            paragraphs.append(
                f"H-bond donors ({hbd}) are above the ideal CNS threshold of ≤ 1. "
                "Two donors contributes a partial score (0.5); eliminating one would "
                "recover a full point in the MPO score."
            )
            optimisation.append("Converting one NH or OH to a non-donor bioisostere would recover full MPO credit.")

    # pKa
    if props["pKa"] < 1.0:
        pka = raw["pKa"]
        if pka > 10:
            paragraphs.append(
                f"Estimated pKa ({pka:.1f}) suggests a strongly basic amine. "
                "High basicity (pKa > 10) means the nitrogen is >99.9% protonated at "
                "physiological pH, which increases hERG channel affinity and the risk of "
                "phospholipidosis. It also lowers effective logD."
            )
            optimisation.append(
                "Reduce basicity by incorporating the nitrogen into an aromatic ring, "
                "adding electron-withdrawing groups nearby, or using a sulfonamide bioisostere."
            )
        elif pka > 8:
            paragraphs.append(
                f"Estimated pKa ({pka:.1f}) indicates moderate basicity. "
                "While tolerable, reducing pKa below 8 would recover the full MPO point "
                "and reduce hERG liability risk."
            )

    if not paragraphs:
        paragraphs.append(
            "All six CNS MPO properties are within their ideal ranges — "
            "no specific structural liabilities identified at this level of analysis."
        )

    if not optimisation:
        optimisation.append(
            "No immediate structural changes suggested — the molecule already meets "
            "CNS MPO criteria."
        )

    return {
        "headline":     headline,
        "paragraphs":   paragraphs,
        "optimisation": optimisation,
        "severity":     severity,
    }


# ── BBB permeability explanation ──────────────────────────────────────────────

def explain_bbbp(result: PredictionResult, mpo_result=None) -> dict:
    """
    Generate a molecule-specific explanation for a BBBP prediction.

    Parameters
    ----------
    result     : PredictionResult from predict_bbbp()
    mpo_result : CNSMPOResult (optional) — used to cross-reference structural factors

    Returns
    -------
    dict with keys:
        headline   : str
        body       : list[str]
        caveat     : str  — model limitation note specific to this prediction
        severity   : str  — 'good' | 'moderate' | 'poor'
    """
    p    = result.probability
    conf = result.confidence

    if result.predicted:
        severity = "good"
        if p >= 0.80:
            headline = (
                f"High confidence BBB-permeable prediction (P = {p:.1%}). "
                "The model finds strong structural evidence for CNS penetration."
            )
        else:
            headline = (
                f"Moderate confidence BBB-permeable prediction (P = {p:.1%}). "
                "The model leans toward permeability but is not highly certain."
            )
    else:
        if p <= 0.25:
            severity = "poor"
            headline = (
                f"High confidence prediction of BBB impermeability (P(perm) = {p:.1%}). "
                "The model finds strong structural features associated with CNS exclusion."
            )
        else:
            severity = "moderate"
            headline = (
                f"Weak prediction of BBB impermeability (P(perm) = {p:.1%}). "
                "The model leans toward non-permeability but the margin is narrow — "
                "treat with caution."
            )

    body = []

    # Confidence-specific commentary
    if conf == "low":
        body.append(
            f"Uncertain prediction (P = {p:.1%}, confidence: low). "
            "The model's output is close to the 0.5 decision boundary. "
            "This often occurs for molecules with structural features underrepresented "
            "in the BBBP training set (~2000 molecules), or for compounds that rely on "
            "active transport rather than passive diffusion."
        )
    elif conf == "moderate":
        body.append(
            f"Moderate confidence (P = {p:.1%}). "
            "The model has a directional preference but is not strongly committed. "
            "Consider this a weak signal rather than a definitive prediction."
        )
    else:
        body.append(
            f"High confidence (P = {p:.1%}). "
            "The Morgan fingerprint pattern of this molecule closely resembles "
            f"{'permeable' if result.predicted else 'non-permeable'} compounds "
            "in the training set."
        )

    # Cross-reference with MPO if available
    if mpo_result is not None:
        raw = mpo_result.raw_values
        tpsa = raw["TPSA"]
        hbd  = raw["HBD"]
        if not result.predicted and tpsa > 90:
            body.append(
                f"Supporting structural evidence: TPSA of {tpsa:.1f} Å is above the "
                "90 Å threshold associated with poor passive CNS permeability. "
                "This is consistent with the model's prediction."
            )
        if not result.predicted and hbd >= 3:
            body.append(
                f"Supporting structural evidence: {hbd} H-bond donors impose a high "
                "desolvation penalty for membrane crossing, supporting the non-permeable prediction."
            )
        if result.predicted and tpsa <= 60 and hbd <= 1:
            body.append(
                f"Supporting structural evidence: Low TPSA ({tpsa:.1f} Å) and "
                f"{hbd} H-bond donor(s) are consistent with efficient passive "
                "transcellular diffusion across the BBB."
            )

    # Model limitation caveat — tailored to probability
    if 0.35 <= p <= 0.65:
        caveat = (
            "Prediction uncertainty is high. This molecule may rely on active transport "
            "(e.g. LAT1, GLUT1, or other SLC transporters) which are invisible to a "
            "fingerprint-based model. Experimental P_app (PAMPA or MDCK-MDR1) assay "
            "is strongly recommended before drawing conclusions."
        )
    elif result.predicted:
        caveat = (
            "A positive prediction reflects structural similarity to known permeable "
            "compounds. It does not rule out P-gp efflux, rapid metabolism, or low "
            "free CNS exposure despite gross permeability."
        )
    else:
        caveat = (
            "A negative prediction reflects structural features associated with CNS "
            "exclusion in the training data. It does not account for active transport "
            "mechanisms — compounds like levodopa and gabapentin are falsely predicted "
            "as non-permeable by this class of model."
        )

    return {
        "headline": headline,
        "body":     body,
        "caveat":   caveat,
        "severity": severity,
    }


# ── ClinTox explanation ───────────────────────────────────────────────────────

def explain_clintox(result: PredictionResult, mpo_result=None) -> dict:
    """
    Generate a molecule-specific explanation for a ClinTox prediction.
 
    Returns
    -------
    dict with keys:
        headline  : str
        body      : list[str]
        caveat    : str
        severity  : str  — 'good' | 'moderate' | 'poor'
    """
    p    = result.probability
    conf = result.confidence
 
    if not result.predicted:
        severity = "good"
        if p <= 0.15:
            headline = (
                f"Low predicted clinical toxicity risk (P(tox) = {p:.1%}). "
                "The molecule's fingerprint pattern resembles compounds that cleared "
                "clinical trials without toxicity-driven failure."
            )
        else:
            headline = (
                f"Moderate–low predicted clinical toxicity risk (P(tox) = {p:.1%}). "
                "The model leans toward safety, but the margin is not large."
            )
    else:
        if p >= 0.75:
            severity = "poor"
            headline = (
                f"High predicted clinical toxicity risk (P(tox) = {p:.1%}). "
                "The model finds strong structural similarity to compounds that failed "
                "clinical trials due to toxicity."
            )
        else:
            severity = "moderate"
            headline = (
                f"Elevated predicted clinical toxicity risk (P(tox) = {p:.1%}). "
                "The model's prediction is positive but not highly confident."
            )
 
    body = []
 
    if conf == "low":
        body.append(
            f"**Low-confidence prediction (P = {p:.1%}).** "
            "The model is near the decision boundary. ClinTox is a small dataset "
            "(~1480 molecules, 12:1 class imbalance towards safe compounds), so "
            "low-confidence predictions should be treated with particular caution."
        )
    elif conf == "moderate":
        body.append(
            f"**Moderate-confidence prediction (P = {p:.1%}).** "
            "Treat as a weak signal; corroborate with structural toxicophore analysis "
            "and in vitro assays (hERG, Ames, cytotoxicity panel)."
        )
    else:
        body.append(
            f"**High-confidence prediction (P = {p:.1%}).** "
            "The structural fingerprint pattern "
            f"{'strongly resembles known clinical toxicants' if result.predicted else 'is dissimilar from known clinical toxicants'}."
        )
 
    # Cross-reference MPO pKa for hERG liability hint
    if mpo_result is not None:
        pka = mpo_result.raw_values["pKa"]
        lp  = mpo_result.raw_values["logP"]
        if pka > 8 and lp > 3 and result.predicted:
            body.append(
                f"**Potential hERG liability:** Basic pKa ({pka:.1f}) combined with "
                f"logP ({lp:.2f}) > 3 is a known risk factor for hERG K⁺ channel block, "
                "which is the most common mechanistic cause of cardiac toxicity-driven "
                "clinical trial failure. Experimental hERG patch-clamp assay is advisable."
            )
        if pka > 10 and result.predicted:
            body.append(
                f"**Phospholipidosis risk:** Strongly basic amines (pKa ≈ {pka:.1f}) are "
                "associated with cationic amphiphilic drug-induced phospholipidosis — "
                "a subcellular toxicity mechanism seen with some antipsychotics and "
                "antidepressants."
            )
 
    # Dataset caveat — always shown for ClinTox due to known imbalance
    body.append(
        f"**Dataset note:** ClinTox is heavily imbalanced ({'{:.0f}'.format(100 * (1 - p))}% "
        "of training molecules are safe). The model was trained with `class_weight='balanced'` "
        "to compensate, but precision for the toxic class remains limited (F1 ≈ 0.14 on this dataset). "
        "A negative prediction is more reliable than a positive one."
    )
 
    caveat = (
        "ClinTox labels reflect trial *failure due to toxicity* — not all toxic "
        "mechanisms are captured. Organ-specific toxicity (hepatotoxicity, nephrotoxicity), "
        "immunogenicity, and long-term chronic effects are not encoded in the fingerprint. "
        "This model is a coarse first filter only."
    )
 
    return {
        "headline": headline,
        "body":     body,
        "caveat":   caveat,
        "severity": severity,
    }
