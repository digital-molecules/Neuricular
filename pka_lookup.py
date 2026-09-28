"""
pka_lookup.py — Neuricular's experimental basic-pKa lookup
==========================================================
The only source of pKa in Neuricular. There is deliberately no estimation
fallback: if a molecule has no entry, its pKa is reported as unavailable
rather than guessed.

Data comes from the IUPAC Digitized pKa Dataset (v2.4a), preprocessed by
build_pka_lookup.py into data/pka_IUPAC.csv. See that script's
docstring for exactly which rows are kept and why.

    Zheng, J. W. and Lafontant-Joseph, O. (2026) IUPAC Digitized pKa Dataset,
    v2.4a. https://doi.org/10.5281/zenodo.7236452
    Copyright (c) 2026 International Union of Pure and Applied Chemistry
    (IUPAC). Reproduced by permission of International Union of Pure and
    Applied Chemistry. Licensed CC BY-NC 4.0 (non-commercial use with
    attribution).

Matching
--------
The query molecule is reduced to its largest fragment (drops counter-ions and
solvents), neutralised, and converted to an InChIKey. The first 14 characters
of that key (formula + connectivity + hydrogens) are the lookup key. That
ignores stereochemistry, charge and isotopes, so salts and stereoisomers of a
listed compound still match. A different constitution never matches.

The dataset is built from 1965–1979 reference works, so coverage of modern
drugs is low. A miss is an ordinary outcome, not an error.

No dependency on chem_calc.py (which imports this module); it takes an RDKit
Mol that the caller has already parsed.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from exceptions import PkaDatabaseError
from schemas import PkaMatch

logger = logging.getLogger(__name__)

PKA_LOOKUP_PATH = Path(__file__).resolve().parent / "data" / "pka_IUPAC.csv"

_REQUIRED_COLUMNS = {"key", "pka", "pka_type", "n_values", "pka_min",
                     "pka_max", "assessment"}

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER        = rdMolStandardize.Uncharger()

_TABLE: Optional[dict] = None   # loaded lazily, once per process


def load_pka_table(path: Path = PKA_LOOKUP_PATH) -> dict:
    """
    Load (and cache) the lookup table as {inchikey_block1: PkaMatch}.

    Raises: PkaDatabaseError if the file is missing or malformed.
    """
    global _TABLE
    if _TABLE is not None:
        return _TABLE

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise PkaDatabaseError(
                    f"pKa lookup '{path}' is missing columns {sorted(missing)}. "
                    "Re-run 'python build_pka_lookup.py' to regenerate it."
                )
            table = {
                row["key"]: PkaMatch(
                    pka        = float(row["pka"]),
                    pka_type   = row["pka_type"],
                    n_values   = int(row["n_values"]),
                    pka_min    = float(row["pka_min"]),
                    pka_max    = float(row["pka_max"]),
                    assessment = row["assessment"],
                )
                for row in reader
            }
    except FileNotFoundError:
        raise PkaDatabaseError(
            f"pKa lookup table '{path}' not found. Download the IUPAC dataset "
            "into data/ and run 'python build_pka_lookup.py' to create it."
        )
    except (ValueError, csv.Error) as exc:
        raise PkaDatabaseError(
            f"pKa lookup table '{path}' could not be read: {exc}. "
            "Re-run 'python build_pka_lookup.py' to regenerate it."
        ) from exc

    if not table:
        raise PkaDatabaseError(
            f"pKa lookup table '{path}' is empty. "
            "Re-run 'python build_pka_lookup.py'."
        )

    logger.info("Loaded %d experimental pKa entries from %s", len(table), path)
    _TABLE = table
    return _TABLE


def _query_key(mol: Chem.Mol) -> Optional[str]:
    """InChIKey connectivity block of the neutralised largest fragment, or None."""
    try:
        neutral = _UNCHARGER.uncharge(_LARGEST_FRAGMENT.choose(mol))
        key = Chem.MolToInchiKey(neutral)
    except Exception as exc:
        logger.debug("Could not build InChIKey for pKa lookup: %s", exc)
        return None
    return key[:14] if key else None


def lookup_basic_pka(mol: Chem.Mol) -> Optional[PkaMatch]:
    """
    Return the database basic-pKa entry for `mol`, or None if there isn't one.

    Raises: PkaDatabaseError if the lookup table can't be loaded.
    """
    table = load_pka_table()
    key = _query_key(mol)
    return table.get(key) if key else None
