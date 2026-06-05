"""
exceptions.py — Neuricular's custom exceptions module
==============================
All custom exceptions for the Neuricular pipeline, in one place.

Organising exceptions centrally means:
- Every module imports from here rather than defining its own error types.
- Callers can catch a specific exception without importing the module that raises it.
- Adding a new exception type does not require touching any computation module.

Hierarchy
---------
Chemistry errors (ValueError subclasses — represent bad user input):
    InvalidSMILESError
    InsufficientMoleculeError

ML pipeline errors (RuntimeError subclasses — represent infrastructure/data failures):
    DatasetLoadError
    InsufficientDataError
    ModelTrainingError
    ModelLoadError
    PredictionError
"""


# ── Chemistry / input errors ──────────────────────────────────────────────────

class InvalidSMILESError(ValueError):
    """
    Raised when a SMILES string cannot be parsed into a valid RDKit molecule.

    This is a user-input error: the string may be malformed, contain
    unsupported syntax, or represent a molecule RDKit cannot kekulize.

    Attributes
    ----------
    smiles : str
        The offending SMILES string, preserved for logging and UI display.
    """
    def __init__(self, smiles: str):
        self.smiles = smiles
        super().__init__(
            f"Could not parse SMILES '{smiles}'. "
            "Please verify the string is a valid, canonical SMILES. "
            "Tools like PubChem Sketcher or RDKit's MolFromSmiles can help validate."
        )


class InsufficientMoleculeError(ValueError):
    """
    Raised when a structurally valid molecule lacks the features required
    for a specific calculation.

    Example: requesting a pKa estimate for a molecule with no nitrogen atoms.

    Attributes
    ----------
    smiles : str
        The molecule's SMILES.
    operation : str
        The calculation that could not proceed.
    """
    def __init__(self, smiles: str, operation: str, reason: str):
        self.smiles    = smiles
        self.operation = operation
        super().__init__(
            f"Cannot compute '{operation}' for '{smiles}': {reason}"
        )


# ── ML pipeline / infrastructure errors ──────────────────────────────────────

class DatasetLoadError(RuntimeError):
    """
    Raised when a dataset cannot be fetched from any of its configured URLs
    and no local fallback file is present.

    This is an infrastructure error, not a user-input error.
    The message includes actionable instructions for manual resolution.
    """


class InsufficientDataError(RuntimeError):
    """
    Raised when a dataset yields too few valid molecules to train a reliable
    binary classifier (e.g. after SMILES filtering or class-label validation).

    Class attribute
    ---------------
    MIN_REQUIRED : int
        Minimum number of valid molecules required to proceed with training.
    """
    MIN_REQUIRED: int = 100


class ModelTrainingError(RuntimeError):
    """
    Raised when the scikit-learn classifier fails during fitting or when
    evaluation metrics cannot be computed (e.g. test set contains only one class).
    """


class ModelLoadError(RuntimeError):
    """
    Raised when a model pickle file is missing, corrupt, or was created
    by an incompatible version of scikit-learn or Neuricular.

    The message always includes the instruction to re-run ml_model.py.
    """


class PredictionError(RuntimeError):
    """
    Raised when inference fails for a structurally valid molecule.

    Distinct from InvalidSMILESError: the molecule parsed correctly,
    but something went wrong during fingerprint generation or model inference
    (e.g. fingerprint length mismatch between training and runtime config).
    """
