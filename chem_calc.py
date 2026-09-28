"""
chem_calc.py — Neuricular's chemistry calculator module
=============================
Molecular descriptor calculations and CNS MPO scoring.

All public functions return a typed result (defined in schemas.py) or raise
a domain exception (defined in exceptions.py). Callers are expected to catch
InvalidSMILESError and present it appropriately in the UI.

pKa policy
----------
pKa comes only from the experimental IUPAC-derived lookup (pka_lookup.py).
There is no estimation fallback. When a molecule has no entry:
  - pKa and logD (which needs pKa to correct for ionisation) are None,
  - the CNS MPO is scored over the four remaining properties,
  - pKa and logD are never used as machine-learning features
    (see get_ml_descriptors).

References
----------
- CNS MPO: Wager et al., ACS Chem. Neurosci. 2010, 1, 435-449
- Lipinski Ro5: Lipinski et al., Adv. Drug Deliv. Rev. 1997, 23, 3-25
- Veber: Veber et al., J. Med. Chem. 2002, 45, 2615-2623
"""

import logging
import math
import urllib.parse
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs, QED
from rdkit.Chem import rdFingerprintGenerator

from exceptions import InvalidSMILESError, PkaDatabaseError
from pka_lookup import lookup_basic_pka
from schemas import (
    DescriptorProfile, CNSMPOResult, PkaMatch,
    PKA_MATCHED, PKA_NOT_FOUND, PKA_DB_UNAVAILABLE,
)

logger = logging.getLogger(__name__)

# Module-level Morgan fingerprint generator — instantiated once, reused everywhere.
# Replaces the deprecated GetMorganFingerprintAsBitVect (silences deprecation warnings).
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse(smiles: str) -> Chem.Mol:
    if not smiles or not smiles.strip():
        raise InvalidSMILESError(smiles)
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise InvalidSMILESError(smiles)
    return mol

def _resolve_pka(mol: Chem.Mol) -> tuple:
    """
    Look up the experimental basic pKa for `mol`.

    Returns (PkaMatch | None, status). A missing database file is reported as
    PKA_DB_UNAVAILABLE rather than raised, so the app keeps working and the
    UI can say the database is unavailable instead of claiming "no match".
    """
    try:
        match = lookup_basic_pka(mol)
    except PkaDatabaseError as exc:
        logger.error("pKa database unavailable: %s", exc)
        return None, PKA_DB_UNAVAILABLE
    if match is None:
        return None, PKA_NOT_FOUND
    return match, PKA_MATCHED


def _logd_from_pka(logp: float, pka: float, ph: float = 7.4) -> float:
    """
    logD of a monoprotic base at `ph`:  logD = logP - log10(1 + 10^(pKa - pH)).

    Assumes only the neutral species partitions into octanol, one basic
    centre (the database's most basic pKa), and no ionisable acidic group.
    """
    return round(logp - math.log10(1.0 + 10 ** (pka - ph)), 3)


# Descriptors used as machine-learning features, in feature-vector order.
# Deliberately excludes logD and pKa: both depend on the database pKa, which is
# missing for most molecules and must not leak into the models.
ML_DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "RotBonds", "QED"]

# CNS MPO (Wager et al. 2010): six 0-1 terms, optimised when the sum is >= 4.0.
_MPO_FULL_TERMS      = 6
_MPO_FULL_THRESHOLD  = 4.0


# ── Public API ────────────────────────────────────────────────────────────────

def get_descriptor_profile(smiles: str) -> DescriptorProfile:
    """
    Compute all standard descriptors and return a DescriptorProfile.
    Lipinski / Veber rule checks are derived automatically by the dataclass.

    pKa and logD are None unless the IUPAC-derived lookup has an entry for the
    molecule; profile.pka_status says which case applies.

    Raises: InvalidSMILESError
    """
    mol  = _parse(smiles)
    logp = Descriptors.MolLogP(mol)

    match, status = _resolve_pka(mol)
    pka  = match.pka if match else None
    logd = _logd_from_pka(logp, pka) if pka is not None else None

    logger.debug("Descriptor profile computed for '%s' (pKa status: %s)", smiles, status)
    return DescriptorProfile(
        smiles     = smiles,
        mw         = round(Descriptors.MolWt(mol), 2),
        logp       = round(logp, 3),
        logd       = logd,
        tpsa       = round(rdMolDescriptors.CalcTPSA(mol), 2),
        hbd        = rdMolDescriptors.CalcNumHBD(mol),
        hba        = rdMolDescriptors.CalcNumHBA(mol),
        rotbond    = Descriptors.NumRotatableBonds(mol),
        qed        = round(QED.qed(mol), 4),
        pka_basic  = pka,
        pka_match  = match,
        pka_status = status,
    )


def get_ml_descriptors(smiles: str) -> list:
    """
    The physicochemical descriptors appended to the Morgan fingerprint as ML
    features, in the order given by ML_DESCRIPTOR_NAMES:
        MW, logP, TPSA, HBD, HBA, RotBonds, QED

    Independent of any pKa lookup, so training and inference build identical
    vectors for every molecule. This is the single source of truth for the
    feature order used by ml_model.py, ml_predict.py and evaluation.py.

    Raises: InvalidSMILESError
    """
    mol = _parse(smiles)
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        QED.qed(mol),
    ]


def get_cns_mpo(smiles: str) -> CNSMPOResult:
    """
    Compute the CNS MPO score (Wager et al. 2010) and return a CNSMPOResult.
    Each property contributes 0-1.

    The logD and pKa terms are scored only when the IUPAC-derived lookup has a
    pKa for the molecule. Otherwise the score is taken over the four remaining
    properties (MW, logP, TPSA, HBD): max_total is 4.0 and cns_optimised uses
    the same proportion as the published 4.0-of-6 cut-off. Such a score is a
    partial assessment, not directly comparable to a full six-term score.

    Raises: InvalidSMILESError
    """
    mol  = _parse(smiles)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    mw   = Descriptors.MolWt(mol)

    match, status = _resolve_pka(mol)
    pka  = match.pka if match else None
    logd = _logd_from_pka(logp, pka) if pka is not None else None

    def d_mw(v):
        if v <= 360: return 1.0
        if v >= 500: return 0.0
        return round(1.0 - (v - 360) / 140, 4)

    def d_logp(v):
        if v <= 3: return 1.0
        if v >= 5: return 0.0
        return round(1.0 - (v - 3) / 2, 4)

    def d_logd(v):
        if v <= 2: return 1.0
        if v >= 4: return 0.0
        return round(1.0 - (v - 2) / 2, 4)

    def d_tpsa(v):
        if 40 <= v <= 90: return 1.0
        if v < 20 or v > 120: return 0.0
        if v < 40: return round((v - 20) / 20, 4)
        return round(1.0 - (v - 90) / 30, 4)

    def d_hbd(v):
        if v <= 1: return 1.0
        if v == 2: return 0.5
        return 0.0

    def d_pka(v):
        if v <= 8: return 1.0
        if v >= 10: return 0.0
        return round(1.0 - (v - 8) / 2, 4)

    s_mw, s_logp = d_mw(mw), d_logp(logp)
    s_tpsa, s_hbd = d_tpsa(tpsa), d_hbd(hbd)
    s_logd = d_logd(logd) if logd is not None else None
    s_pka  = d_pka(pka)   if pka  is not None else None

    scored    = [s_mw, s_logp, s_tpsa, s_hbd] + [t for t in (s_logd, s_pka) if t is not None]
    total     = round(sum(scored), 4)
    max_total = float(len(scored))
    threshold = _MPO_FULL_THRESHOLD * max_total / _MPO_FULL_TERMS

    logger.debug("CNS MPO for '%s': %.3f/%.0f (pKa status: %s)", smiles, total, max_total, status)
    return CNSMPOResult(
        smiles=smiles, score_mw=s_mw, score_logp=s_logp, score_tpsa=s_tpsa,
        score_hbd=s_hbd, score_logd=s_logd, score_pka=s_pka,
        total=total, max_total=max_total, cns_optimised=total >= threshold,
        raw_mw=round(mw, 2), raw_logp=round(logp, 3), raw_tpsa=round(tpsa, 2),
        raw_hbd=hbd, raw_logd=logd, raw_pka=pka,
        pka_match=match, pka_status=status,
    )


def get_tanimoto(smiles1: str, smiles2: str) -> float:
    """
    Tanimoto similarity via Morgan fingerprints (radius=2, nBits=2048).

    Raises: InvalidSMILESError for either invalid SMILES.
    """
    mol1 = _parse(smiles1)
    mol2 = _parse(smiles2)
    fp1  = _MORGAN_GEN.GetFingerprint(mol1)
    fp2  = _MORGAN_GEN.GetFingerprint(mol2)
    return round(DataStructs.FingerprintSimilarity(fp1, fp2), 4)


def get_morgan_fp_array(smiles: str) -> list:
    """
    Return a 2048-bit Morgan fingerprint as a list of ints.
    Used internally by ml_predict.py.

    Raises: InvalidSMILESError
    """
    mol = _parse(smiles)
    fp  = _MORGAN_GEN.GetFingerprint(mol)
    return list(fp)


def get_cns_tanimoto_panel(smiles: str, top_n: int = 10) -> list[dict]:
    """
    Compute Tanimoto similarity between the query molecule and every compound
    in the CNS drug reference database (cns_drugs.py), returning the top_n
    most similar matches sorted by descending similarity.

    Each result dict contains all fields from the database entry plus:
        similarity : float  — Tanimoto coefficient (0-1)

    Entries with invalid SMILES in the database are silently skipped.

    Raises: InvalidSMILESError if the query SMILES is invalid.
    """
    from cns_drugs import CNS_DRUG_DATABASE

    query_mol = _parse(smiles)
    query_fp  = _MORGAN_GEN.GetFingerprint(query_mol)

    results = []
    seen_names = set()   # deduplicate drugs listed under multiple categories

    for drug in CNS_DRUG_DATABASE:
        if drug["name"] in seen_names:
            continue
        try:
            ref_mol = Chem.MolFromSmiles(drug["smiles"])
            if ref_mol is None:
                logger.warning("Skipping reference drug '%s': invalid SMILES", drug["name"])
                continue
            ref_fp = _MORGAN_GEN.GetFingerprint(ref_mol)
            sim = round(DataStructs.FingerprintSimilarity(query_fp, ref_fp), 4)
            results.append({**drug, "similarity": sim})
            seen_names.add(drug["name"])
        except Exception as exc:
            logger.debug("Error computing similarity for '%s': %s", drug["name"], exc)
            continue

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]

def resolve_smiles(user_input: str) -> tuple[str, str]:

    stripped = user_input.strip()

    # Try as SMILES first
    mol = Chem.MolFromSmiles(stripped)
    if mol is not None:
        return stripped, "smiles"

    encoded = urllib.parse.quote(stripped)

    # PubChem PUG REST: compound name -> isomeric SMILES
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES/TXT"

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/plain",
            }
        )
        if resp.status_code == 200:
            smiles = resp.text.strip().split("\n")[0]
            if smiles:
                return smiles, "pubchem"
        raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        logger.warning("PubChem lookup failed for '%s': %s", stripped, exc)
        raise InvalidSMILESError(
            f"'{stripped}' could not be resolved. "
            "Try pasting the SMILES directly — find it at pubchem.ncbi.nlm.nih.gov."
        )
