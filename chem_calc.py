"""
chem_calc.py — Neuricular's chemistry calculator module
=============================
Molecular descriptor calculations and CNS MPO scoring.

All public functions return a typed result (defined in schemas.py) or raise
a domain exception (defined in exceptions.py). Callers are expected to catch
InvalidSMILESError and present it appropriately in the UI.

References
----------
- CNS MPO: Wager et al., ACS Chem. Neurosci. 2010, 1, 435-449
- Lipinski Ro5: Lipinski et al., Adv. Drug Deliv. Rev. 1997, 23, 3-25
- Veber: Veber et al., J. Med. Chem. 2002, 45, 2615-2623
"""

import logging

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs, QED
from rdkit.Chem import rdFingerprintGenerator

from exceptions import InvalidSMILESError
from schemas import DescriptorProfile, CNSMPOResult

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


def _estimate_logd(mol: Chem.Mol, logp: float) -> float:
    """
    Estimate logD at pH 7.4 using the Mannhold correction for basic amines.
    logD ≈ logP - 1.0 when a non-aromatic basic nitrogen is present.
    Limitation: overestimates for polybasic amines; use as screening value only.
    """
    basic_n_count = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7
        and not atom.GetIsAromatic()
        and atom.GetTotalValence() < 4
        and atom.GetNoImplicit() is False
    )
    return round(logp - (1.0 if basic_n_count > 0 else 0.0), 3)


def _estimate_pka_basic(mol: Chem.Mol) -> float:
    """
    Estimate the most basic pKa via nitrogen-type heuristic.
    Proper pKa requires tools like ChemAxon Marvin or Schrödinger Epik.
 
    Evaluates ALL nitrogens and returns the pKa of the most basic site.
    Uses a tier system so that a higher-tier nitrogen (e.g. aliphatic amine)
    cannot be overridden by a lower-tier one found later in iteration,
    but also so that a conjugated nitrogen correctly outranks a simple
    aromatic nitrogen even if both are present (as in tizanidine).
 
    Tiers (higher = more basic):
      0 — no basic nitrogen        → pKa 4.0
      1 — aromatic N               → pKa 5.0
      2 — conjugated non-aromatic  → pKa 7.5  (imidazoline, amidine, guanidine)
      3 — aliphatic amine          → pKa 10.5
 
    Key design note: the exocyclic NH linker in molecules like tizanidine
    is aliphatic (tier 3) but its basicity is suppressed by flanking
    electron-withdrawing groups (aromatic ring + C=N). We detect this by
    checking whether an aliphatic N is directly bonded to an aromatic ring
    or to a carbon that is itself bonded to an aromatic ring — if so, we
    downgrade it to tier 2 (pKa 7.5) to reflect the inductive withdrawal.
    """
    best_tier = 0
    best_pka  = 4.0
 
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
 
        # Exclude amide nitrogens
        is_amide = any(
            nbr.GetAtomicNum() == 6
            and any(
                b.GetBondTypeAsDouble() == 2.0
                for b in nbr.GetBonds()
                if b.GetOtherAtom(nbr).GetAtomicNum() == 8
            )
            for nbr in atom.GetNeighbors()
        )
        if is_amide:
            continue
 
        # Tier 1 — aromatic nitrogen
        if atom.GetIsAromatic():
            if best_tier < 1:
                best_tier, best_pka = 1, 5.0
            continue
 
        # Tier 2 — conjugated non-aromatic: adjacent to a C=N bond
        is_conjugated = False
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 6:
                continue
            for bond in nbr.GetBonds():
                other = bond.GetOtherAtom(nbr)
                if (other.GetIdx() != atom.GetIdx()
                        and other.GetAtomicNum() == 7
                        and bond.GetBondTypeAsDouble() == 2.0):
                    is_conjugated = True
                    break
            if is_conjugated:
                break
 
        if is_conjugated:
            if best_tier < 2:
                best_tier, best_pka = 2, 7.5
            continue
 
        # Tier 3 — aliphatic amine, but check for EW group suppression.
        # If this N is directly bonded to an aromatic atom, or bonded to a
        # carbon that bears a double bond to N (i.e. is adjacent to C=N),
        # its basicity is suppressed — treat as tier 2 (pKa 7.5).
        if atom.GetTotalValence() in (2, 3):
            is_suppressed = any(
                nbr.GetIsAromatic()
                or (nbr.GetAtomicNum() == 6 and any(
                    b.GetBondTypeAsDouble() == 2.0
                    and b.GetOtherAtom(nbr).GetAtomicNum() == 7
                    for b in nbr.GetBonds()
                    if b.GetOtherAtom(nbr).GetIdx() != atom.GetIdx()
                ))
                for nbr in atom.GetNeighbors()
            )
            if is_suppressed:
                if best_tier < 2:
                    best_tier, best_pka = 2, 7.5
            else:
                if best_tier < 3:
                    best_tier, best_pka = 3, 10.5
 
    return best_pka


# ── Public API ────────────────────────────────────────────────────────────────

def get_descriptor_profile(smiles: str) -> DescriptorProfile:
    """
    Compute all standard descriptors and return a DescriptorProfile.
    Lipinski / Veber rule checks are derived automatically by the dataclass.

    Raises: InvalidSMILESError
    """
    mol  = _parse(smiles)
    logp = Descriptors.MolLogP(mol)
    logd = _estimate_logd(mol, logp)
    pka  = _estimate_pka_basic(mol)
    logger.debug("Descriptor profile computed for '%s'", smiles)
    return DescriptorProfile(
        smiles    = smiles,
        mw        = round(Descriptors.MolWt(mol), 2),
        logp      = round(logp, 3),
        logd      = logd,
        tpsa      = round(rdMolDescriptors.CalcTPSA(mol), 2),
        hbd       = rdMolDescriptors.CalcNumHBD(mol),
        hba       = rdMolDescriptors.CalcNumHBA(mol),
        rotbond   = Descriptors.NumRotatableBonds(mol),
        qed       = round(QED.qed(mol), 4),
        pka_basic = pka,
    )


def get_cns_mpo(smiles: str) -> CNSMPOResult:
    """
    Compute the CNS MPO score (Wager et al. 2010) and return a CNSMPOResult.
    Each property contributes 0–1; score >= 4.0 = CNS-optimised.

    Raises: InvalidSMILESError
    """
    mol  = _parse(smiles)
    logp = Descriptors.MolLogP(mol)
    logd = _estimate_logd(mol, logp)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    mw   = Descriptors.MolWt(mol)
    pka  = _estimate_pka_basic(mol)

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
    s_logd, s_tpsa = d_logd(logd), d_tpsa(tpsa)
    s_hbd, s_pka  = d_hbd(hbd), d_pka(pka)
    total = round(s_mw + s_logp + s_logd + s_tpsa + s_hbd + s_pka, 4)

    logger.debug("CNS MPO for '%s': %.3f/6", smiles, total)
    return CNSMPOResult(
        smiles=smiles, score_mw=s_mw, score_logp=s_logp, score_logd=s_logd,
        score_tpsa=s_tpsa, score_hbd=s_hbd, score_pka=s_pka, total=total,
        cns_optimised=total >= 4.0,
        raw_mw=round(mw, 2), raw_logp=round(logp, 3), raw_logd=logd,
        raw_tpsa=round(tpsa, 2), raw_hbd=hbd, raw_pka=round(pka, 2),
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
        similarity : float  — Tanimoto coefficient (0–1)

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
