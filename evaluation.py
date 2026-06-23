"""
evaluation.py - Neuricular's model evaluation and performance analysis module
=============================

Outputs:
    results/
        performance_summary.csv
        roc_curves.svg
        confusion_matrices.svg
        feature_importances.svg
        dataset_distribution.svg
        cns_reference_validation.csv
"""

import os
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ml_predict import (
    predict_bbbp,
    CNS_REFERENCE_DRUGS,
    predict_clintox,
)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_artefact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------
# Load models
# --------------------------------------------------

bbbp = load_artefact("bbbp_model.pkl")
clintox = load_artefact("clintox_model.pkl")

artefacts = [bbbp, clintox]

# --------------------------------------------------
# Performance table
# --------------------------------------------------

summary = pd.DataFrame([
    {
        "Dataset": a.dataset_name,
        "AUC": a.auc,
        "Precision": a.precision,
        "Recall": a.recall,
        "F1": a.f1,
    }
    for a in artefacts
])

summary.to_csv(
    f"{OUTPUT_DIR}/performance_summary.csv",
    index=False
)

print(summary)

# --------------------------------------------------
# ROC Curves
# --------------------------------------------------

plt.figure(figsize=(7, 6))

for a in artefacts:
    plt.plot(
        a.fpr,
        a.tpr,
        label=f"{a.dataset_name.upper()} (AUC={a.auc:.3f})"
    )

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/roc_curves.svg",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Confusion Matrices
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, a in zip(axes, artefacts):

    sns.heatmap(
        a.cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(a.dataset_name.upper())
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrices.svg",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, a in zip(axes, artefacts):

    idx = np.argsort(a.feature_importances)[::-1][:20]

    vals = a.feature_importances[idx]

    ax.barh(
        range(len(idx)),
        vals
    )

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(idx)

    ax.invert_yaxis()

    ax.set_title(
        f"{a.dataset_name.upper()} Top Fingerprints"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/feature_importances.svg",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Feature Importance — Physicochemical Descriptors
# --------------------------------------------------
# The feature vector is [2048 Morgan fingerprint bits] + [8 descriptors].
# Descriptor indices 2048-2055 correspond to:
DESCRIPTOR_NAMES = ["MW", "logP", "logD", "TPSA", "HBD", "HBA", "RotBonds", "QED"]
DESCRIPTOR_START = 2048

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, a in zip(axes, artefacts):
    desc_importances = a.feature_importances[DESCRIPTOR_START:]
    bars = ax.barh(DESCRIPTOR_NAMES[::-1], desc_importances[::-1])
    ax.set_xlabel("Feature importance (Gini)")
    ax.set_title(f"{a.dataset_name.upper()} — Descriptor Feature Importances")
    ax.set_xlim(0, max(desc_importances) * 1.25)
    for bar, val in zip(bars, desc_importances[::-1]):
        ax.text(
            val + max(desc_importances) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}",
            va="center", fontsize=8
        )
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/descriptor_importances.svg",
    dpi=300
)
plt.close()

# --------------------------------------------------
# Dataset Distribution
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, a in zip(axes, artefacts):

    labels = list(a.stats.class_balance.keys())
    counts = list(a.stats.class_balance.values())

    ax.bar(labels, counts)

    ax.set_title(a.dataset_name.upper())
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/dataset_distribution.svg",
    dpi=300
)

plt.close()

# --------------------------------------------------
# CNS Validation (BBB + Toxicity combined)
# --------------------------------------------------

print("\n--- CNS DEBUG RUN ---\n")

rows = []

for drug in CNS_REFERENCE_DRUGS:
    name = drug["name"]
    smiles = drug["smiles"]

    try:
        # BBB prediction
        bbb_pred = predict_bbbp(smiles, bbbp)

        # Toxicity prediction (ClinTox model)
        tox_pred = predict_clintox(smiles, clintox)

        print(name)
        print("BBB:", bbb_pred)
        print("TOX:", tox_pred)
        print("OK\n")

        rows.append({
            # Identity
            "Drug": name,

            # BBB ground truth + prediction
            "Known BBB": drug["known_bbb_permeable"],
            "Predicted BBB": bbb_pred.predicted,
            "BBB Probability": bbb_pred.probability,
            "BBB Confidence": bbb_pred.confidence,

            # Toxicity prediction (NEW)
            "Predicted Toxicity": tox_pred.predicted,
            "Toxicity Probability": tox_pred.probability,
            "Toxicity Confidence": tox_pred.confidence,

        })

    except Exception as e:
        print(name)
        print(type(e).__name__)
        print(e)
        print("FAILED\n")

        rows.append({
            "Drug": name,

            "Known BBB": drug["known_bbb_permeable"],
            "Predicted BBB": "ERROR",
            "BBB Probability": np.nan,
            "BBB Confidence": "ERROR",

            "Predicted Toxicity": "ERROR",
            "Toxicity Probability": np.nan,
            "Toxicity Confidence": "ERROR",
        })


cns_df = pd.DataFrame(rows)

cns_df.to_csv(
    f"{OUTPUT_DIR}/cns_reference_validation.csv",
    index=False
)
