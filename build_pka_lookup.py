"""
build_pka_lookup.py — build Neuricular's basic-pKa lookup table from the IUPAC dataset
=====================================================================================
One-off preprocessing script. Reads the IUPAC Digitized pKa Dataset and writes
a compact lookup (data/pka_IUPAC.csv) that pka_lookup.py loads at
runtime. Re-run only if the source dataset is updated.

Usage
-----
    python build_pka_lookup.py [path/to/iupac_high-confidence_v2_4.csv]

Source dataset
--------------
Zheng, J. W. and Lafontant-Joseph, O. (2026) IUPAC Digitized pKa Dataset,
v2.4a. https://doi.org/10.5281/zenodo.7236452
https://github.com/IUPAC/Dissociation-Constants
Copyright (c) 2026 International Union of Pure and Applied Chemistry (IUPAC).
Reproduced by permission of International Union of Pure and Applied
Chemistry. Licensed CC BY-NC 4.0 — non-commercial use with attribution.

Selection rules (each exists to avoid loading a wrong kind of value into the
"most basic center pKa" that CNS MPO expects)
-----------------------------------------------------------------------
1. acidity_label == 'AH'  (pKa of the conjugate acid of a base, i.e. pKaH).
   Acidic ('A'), pKb ('B', mostly high-pressure runs) and 'other'
   (unusual protonation sites) rows are dropped.
2. 20 <= T <= 30 degC. Non-numeric T ('not stated', '<25', ...) is dropped.
3. No cosolvent and no non-ambient pressure.
4. Remarks mentioning Hammett acidity functions (Ho, HR, ...) or excited
   states (Forster cycle) are dropped: those are extreme-acid-medium or
   photophysical measurements, not ordinary aqueous pKaH values.
5. assessment 'Very uncertain' / 'Unknown' is dropped.
6. Single-component structures only (no salts/mixtures in the source InChI).
7. Per molecule: median of repeated measurements per pKaH type, then the
   MAXIMUM across types. For a polybasic compound pKaH1 is the first
   dissociation of the fully protonated species (lowest value); the highest
   pKaH belongs to the last proton lost, i.e. the most basic center.
8. Amphoteric molecules are excluded: any molecule carrying BOTH 'AH' and
   'A' labels anywhere in the source. Which value belongs to the basic
   center is exactly what the source is least reliable about for these. Its
   README says that with two reported pK values the lower is simply assumed
   basic, and amino-acid-type compounds break that (levodopa's COOH value of
   2.32 is labelled as its base; the real amine pKaH is ~8.7). Compounds
   with more than two values were labelled by manual inspection, but the
   same COOH-as-base pattern still occurs (levodopa has four values). No
   label-level rule separates right from wrong, so all are dropped rather
   than trusted. This also drops some correct entries (e.g. morphine, whose
   amine value is right); that is the price of never loading an assumed
   label into the pKa field.

Matching key: first block of the InChIKey (hash of formula + connectivity +
hydrogen layer). Ignores stereochemistry, charge and isotopes, so a query
molecule matches after salt-stripping and neutralisation.
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

DEFAULT_SOURCE = Path("data/iupac_high-confidence_v2_4.csv")
OUTPUT_PATH = Path("data/pka_IUPAC.csv")

# Hammett acidity function scales, excited-state (Forster cycle) values.
_EXCLUDE_REMARKS = re.compile(
    r"\bH[o0_R]\b|H_|acidity function|excited|F[oö]rster|Hammett|\bHR\b",
    re.IGNORECASE,
)
_EXCLUDED_ASSESSMENTS = {"Very uncertain", "Unknown"}
_ASSESSMENT_RANK = {"Reliable": 0, "Approximate": 1, "Probably approximate": 1,
                    "Uncertain": 2}


def _inchikey_block1(inchi_str: str):
    """First 14 characters of the InChIKey, or None if it can't be made."""
    try:
        key = inchi.InchiToInchiKey(inchi_str)
    except Exception:
        return None
    return key[:14] if key else None


def build(source: Path = DEFAULT_SOURCE, output: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    n0 = len(df)
    steps = {"raw rows": n0}

    # Resolve keys up front: rule 8 needs the *full* per-molecule picture
    # (all labels), before any row filtering.
    df = df.assign(key=df["InChI"].map(_inchikey_block1))
    df = df[df["key"].notna()]
    steps["InChIKey resolved"] = len(df)

    ab = df[df["acidity_label"].isin(["AH", "A", "other"])]
    per_mol = ab.groupby("key").agg(
        has_ah=("acidity_label", lambda s: (s == "AH").any()),
        has_a=("acidity_label", lambda s: (s == "A").any()),
    )
    amphoteric_keys = set(per_mol.index[per_mol["has_ah"] & per_mol["has_a"]])

    df = df[df["acidity_label"] == "AH"]
    steps["acidity_label == AH"] = len(df)

    df = df.assign(
        T_num=pd.to_numeric(df["T"], errors="coerce"),
        pka=pd.to_numeric(df["pka_value"], errors="coerce"),
    )
    df = df[df["pka"].notna() & df["T_num"].between(20, 30)]
    steps["20-30 C, numeric pKa"] = len(df)

    df = df[df["cosolvent"].isna() & df["pressure"].isna()]
    steps["no cosolvent / pressure"] = len(df)

    df = df[~df["remarks"].fillna("").str.contains(_EXCLUDE_REMARKS)]
    steps["no acidity-function / excited-state"] = len(df)

    df = df[~df["assessment"].isin(_EXCLUDED_ASSESSMENTS)]
    steps["assessment ok"] = len(df)

    df = df[~df["InChI"].str.contains(r"^InChI=1S/[^/]*\.", regex=True, na=False)]
    steps["single component"] = len(df)

    n_before = df["key"].nunique()
    df = df[~df["key"].isin(amphoteric_keys)]
    steps["molecules dropped: amphoteric"] = n_before - df["key"].nunique()

    df = df.assign(assess_rank=df["assessment"].map(_ASSESSMENT_RANK).fillna(3))

    # Per (molecule, pKaH type): median over repeated measurements.
    per_type = (
        df.groupby(["key", "pka_type"])
        .agg(
            pka=("pka", "median"),
            n_values=("pka", "size"),
            pka_min=("pka", "min"),
            pka_max=("pka", "max"),
            best_assessment=("assess_rank", "min"),
            smiles=("SMILES", "first"),
        )
        .reset_index()
    )
    # Most basic center = highest pKaH across types.
    idx = per_type.groupby("key")["pka"].idxmax()
    out = per_type.loc[idx].copy()
    rank_to_label = {0: "Reliable", 1: "Approximate", 2: "Uncertain", 3: "Unrated"}
    out["assessment"] = out["best_assessment"].map(rank_to_label)
    out["pka"] = out["pka"].round(2)
    out = out[["key", "pka", "pka_type", "n_values", "pka_min", "pka_max",
               "assessment", "smiles"]].sort_values("key")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    steps["molecules in lookup"] = len(out)
    for name, n in steps.items():
        print(f"{name:40s} {n:>7d}")
    print(f"\nWrote {output}")
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    build(src)
