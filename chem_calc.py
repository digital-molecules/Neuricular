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
import math
import urllib.parse
import requests
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

def _get_basic_sites(mol):
    """
    Returns list of (atom_idx, base_pka, type)
    """

    patterns = [
        ("aliphatic_amine", Chem.MolFromSmarts("[NX3;H2,H1,H0][CX4]"), 10.5),
        ("pyridine", Chem.MolFromSmarts("n1ccccc1"), 5.2),
        ("imidazole", Chem.MolFromSmarts("n1cnc[nH]1"), 7.5),
        ("guanidine", Chem.MolFromSmarts("NC(=N)N"), 13.5),
        ("amidine", Chem.MolFromSmarts("N=C(N)N"), 11.0),
    ]

    sites = []

    for name, smarts, pka in patterns:
        if smarts is None:
            continue

        for match in mol.GetSubstructMatches(smarts):
            sites.append((match[0], pka, name))

    return sites

def _basicity_penalty(atom):
    """
    Smooth electronic penalty model (no hard tiers)
    
    Penalty is meant to deprioritize basic sites that are
    electronically deactivated by their local environment
    """

    p = 0.0

    for nbr in atom.GetNeighbors():

        if nbr.GetIsAromatic():
            p += 0.9

        if nbr.GetAtomicNum() == 6:
            for b in nbr.GetBonds():
                o = b.GetOtherAtom(nbr)
                if o.GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2.0:
                    p += 1.1 # this loop penalizes amides, esters and ketones due to conjugation

        if nbr.GetAtomicNum() in (7, 8, 9):
            p += 0.25

    return p
    
def _ionization_profile(mol):
    """
    Computes dominant + secondary protonation contributions.
    Returns sorted list of effective site pKas.

    The effective pKa is the base pKa minus the local electronic penalty.
    The score is the effective pKa minus a mild physiological smoothing factor
    (to prioritize sites that are closer to neutral at pH 7.4).
    """

    sites = _get_basic_sites(mol)

    scored_sites = []

    for atom_idx, base_pka, name in sites:
        atom = mol.GetAtomWithIdx(atom_idx)

        pka_eff = base_pka - _basicity_penalty(atom)

        # mild physiological smoothing (CNS relevance)
        score = pka_eff - abs(pka_eff - 7.4) * 0.15 # this factor is tuned to prioritize sites that are closer to neutral at pH 7.4

        scored_sites.append((score, pka_eff))

    # sort by relevance
    scored_sites.sort(reverse=True, key=lambda x: x[0])

    return scored_sites

def _estimate_pka_basic(mol: Chem.Mol) -> float:
    """
    CNS MPO effective pKa:
    dominant protonation site with improved micro-environment model
    """

    profile = _ionization_profile(mol)

    if not profile: # No basic sites found
        return 4.0

    # dominant site (but NOT raw max; environment-weighted max)
    _, pka = profile[0]

    return round(pka, 2)

def _estimate_logd(mol: Chem.Mol, logp: float) -> float:
    """
    Improved CNS-relevant logD model.
    Uses dominant + secondary ionization effects.
    """

    profile = _ionization_profile(mol)

    pH = 7.4

    if not profile:
        return round(logp, 3)

    # dominant site
    _, pka1 = profile[0]
    frac1 = 1.0 / (1.0 + 10 ** (pH - pka1))

    # secondary site dampening (important for polyamines)
    '''
    The following represents the Henderson-Hasselbalch equation,
    which describes the relationship between pH, pKa and
    the ratio of protonated to deprotonated species.
    The factor of 0.5 is an empirical adjustment to account for
    the reduced contribution of secondary sites to overall ionization.
    '''
    frac2 = 0.0
    if len(profile) > 1:
        _, pka2 = profile[1]
        frac2 = 0.5 * (1.0 / (1.0 + 10 ** (pH - pka2)))
    ionization = frac1 + frac2

    logd = logp - ionization

    return round(logd, 3)
    
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
    Each property contributes 0-1; score >= 4.0 = CNS-optimised.

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

    # Use PubChem's fast structure lookup endpoint
    # This hits a different CDN than the main PUG REST API
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
