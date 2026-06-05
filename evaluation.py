"""
evaluation.py - Neuricular's model evaluation and performance analysis module
=============================

Outputs:
    results/
        performance_summary.csv
        roc_curves.png
        confusion_matrices.png
        feature_importances.png
        dataset_distribution.png
        cns_reference_validation.csv
"""

import os
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ml_model import DATASETS, _load_dataset
from ml_predict import (
    load_model,
    predict_bbbp,
    CNS_REFERENCE_DRUGS,
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
    f"{OUTPUT_DIR}/roc_curves.png",
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
    f"{OUTPUT_DIR}/confusion_matrices.png",
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
    f"{OUTPUT_DIR}/feature_importances.png",
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
    f"{OUTPUT_DIR}/dataset_distribution.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# CNS Validation
# --------------------------------------------------

print("\n--- CNS DEBUG RUN ---\n")
rows = []
for drug in CNS_REFERENCE_DRUGS:
    try:
        pred = predict_bbbp(
            drug["smiles"],
            bbbp
        )

        print(drug["name"])
        print(pred)
        print("OK\n")

        rows.append({
            "Drug": drug["name"],
            "Known BBB": drug["known_bbb_permeable"],
            "Predicted BBB": pred.predicted,
            "Probability": pred.probability,
            "Confidence": pred.confidence,
        })

    except Exception as e:
        print(drug["name"])
        print(type(e).__name__)
        print(e)
        print("FAILED\n")

        rows.append({
            "Drug": drug["name"],
            "Known BBB": drug["known_bbb_permeable"],
            "Predicted BBB": "ERROR",
            "Probability": np.nan,
        })
cns_df = pd.DataFrame(rows)

cns_df.to_csv(
    f"{OUTPUT_DIR}/cns_reference_validation.csv",
    index=False
)

print(
    f"\nAll outputs saved to '{OUTPUT_DIR}'"
)