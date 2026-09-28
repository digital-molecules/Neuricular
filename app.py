"""
app.py  - Neuricular's frontend Streamlit application
======================
CNS Drug Candidate Screening Pipeline

Tabs:
  1. Descriptors  - standard MW/logP/TPSA/QED panel + radar chart + Tanimoto
  2. CNS MPO      - Wager 2010 score with per-property breakdown
  3. ML           - BBB permeability + clinical toxicity predictions + combined verdict
  4. Reference    - curated CNS drug panel with model agreement analysis

Run:
    streamlit run app.py
"""

import logging
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from chem_calc import get_descriptor_profile, get_cns_mpo, get_tanimoto, get_cns_tanimoto_panel, resolve_smiles
from ml_predict import load_model, predict_bbbp, predict_clintox, get_top_features, CNS_REFERENCE_DRUGS
from exceptions import InvalidSMILESError, ModelLoadError, PredictionError, PkaDatabaseError
from pka_lookup import load_pka_table
from schemas import PKA_MATCHED, PKA_NOT_FOUND, PKA_DB_UNAVAILABLE
from explanations import explain_cns_mpo, explain_bbbp, explain_clintox

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuricular",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #F5E6EE;
    color: #2D4A6B;
}
.stApp { background-color: #F5E6EE; }
header[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background-color: #EDD3DF;
    border-right: 1px solid #C9A8BB;
}
.stTextInput > div > div > input {
    background-color: #fffcfe !important;
    border: 1px solid #5d6987 !important;
    border-radius: 4px !important;
    color: #2D4A6B !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3D6A99 !important;
    box-shadow: 0 0 0 2px rgba(74,158,255,0.15) !important;
}
[data-testid="metric-container"] {
    background-color: #fffcfe;
    border: 1px solid #C9A8BB;
    border-radius: 6px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6a7a90 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.3rem !important;
    color: #2D4A6B !important;
}
[data-testid="stExpander"] {
    background-color: #EDD3DF;
    border: 1px solid #C9A8BB !important;
    border-radius: 6px;
}
[data-baseweb="tab-list"] {
    background-color: #EDD3DF;
    border-bottom: 1px solid #C9A8BB;
    gap: 0;
}
[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6a7a90;
    padding: 0.7rem 1.4rem;
}
[aria-selected="true"] {
    color: #3D6A99 !important;
    background: transparent !important;
}
hr { border-color: #C9A8BB; }
[data-testid="stCaptionContainer"] p {
    color: #7A9ABF !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}
div[data-testid="stDataFrame"] { border: 1px solid #C9A8BB; border-radius: 6px; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F5E6EE; }
::-webkit-scrollbar-thumb { background: #5d6987; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor": "#F5E6EE",
    "axes.facecolor":   "#EDD3DF",
    "axes.edgecolor":   "#5d6987",
    "axes.labelcolor":  "#496D99",
    "xtick.color":      "#6a7a90",
    "ytick.color":      "#6a7a90",
    "text.color":       "#2D4A6B",
    "grid.color":       "#E4C5D4",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
    "axes.titlesize":   11,
    "axes.labelsize":   9,
})

# ── Model loading (cached, fails gracefully) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models…")
def _load_models():
    errors = {}
    bbbp, clintox = None, None
    try:
        bbbp = load_model("bbbp_model.pkl")
    except ModelLoadError as e:
        errors["bbbp"] = str(e)
    try:
        clintox = load_model("clintox_model.pkl")
    except ModelLoadError as e:
        errors["clintox"] = str(e)
    return bbbp, clintox, errors

bbbp_model, clintox_model, model_errors = _load_models()


@st.cache_resource(show_spinner=False)
def _check_pka_database():
    """(entry_count, error_message). The lookup is the only source of pKa."""
    try:
        return len(load_pka_table()), None
    except PkaDatabaseError as e:
        return 0, str(e)

pka_db_size, pka_db_error = _check_pka_database()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_opt(value, na: str = "n/a") -> str:
    """Format an optional number; None (e.g. no database pKa) shows as n/a."""
    return na if value is None else f"{value}"


def _label(text: str) -> str:
    """Render a small monospace section label."""
    return (
        f"<div style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; "
        f"text-transform:uppercase; letter-spacing:0.1em; color:#7A9ABF; "
        f"margin-bottom:0.6rem;'>{text}</div>"
    )


def _card(label: str, value: str, color: str = "#2D4A6B") -> str:
    return f"""
    <div style="background:#fffcfe; border:1px solid #C9A8BB; border-radius:6px;
                padding:0.9rem 1rem; margin-bottom:0.4rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    text-transform:uppercase; letter-spacing:0.08em; color:#7A9ABF;">
            {label}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.25rem;
                    color:{color}; margin-top:0.2rem; font-weight:500;">
            {value}
        </div>
    </div>
    """


def _verdict_card(label: str, passed: bool, detail: str) -> str:
    color  = "#2D7A4F" if passed else "#9B2335"
    symbol = "✓" if passed else "✗"
    return f"""
    <div style="background:#EDD3DF; border:1px solid #C9A8BB;
                border-left:3px solid {color}; border-radius:4px;
                padding:0.9rem 1rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    color:#7A9ABF; text-transform:uppercase; letter-spacing:0.08em;">
            {label}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.2rem;
                    color:{color}; margin-top:0.3rem;">
            {symbol} <span style="font-size:0.82rem; color:#496D99;">{detail}</span>
        </div>
    </div>
    """


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 1.5rem 0; border-bottom:1px solid #C9A8BB; margin-bottom:2rem;">
    <div style="display:flex; align-items:baseline; gap:0.75rem; flex-wrap:wrap;">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
                     font-weight:500; color:#2D4A6B; letter-spacing:-0.02em;">
            Neuricular
        </span>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                     color:#3D6A99; letter-spacing:0.12em; text-transform:uppercase;
                     border:1px solid #1A3A5C; padding:2px 8px; border-radius:3px;">
            CNS Pipeline
        </span>
    </div>
    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem;
                color:#7A9ABF; margin-top:0.4rem; font-weight:300; letter-spacing:0.02em;">
        Descriptors · CNS MPO · BBB permeability · Clinical toxicity · Reference drug analysis
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model load warnings ───────────────────────────────────────────────────────
if model_errors:
    for model_name, err_msg in model_errors.items():
        st.warning(
            f"{model_name.upper()} model not loaded. {err_msg}",
            icon="⚠️"
        )

if pka_db_error:
    st.warning(
        f"pKa database not loaded, so pKa and logD will be unavailable and CNS MPO "
        f"will be scored over four properties. {pka_db_error}",
        icon="⚠️",
    )

# ── SMILES input ──────────────────────────────────────────────────────────────
col_in, col_hint = st.columns([3, 2])
with col_in:
    raw_smiles = st.text_input(
        "smiles_input",
        placeholder="Paste a SMILES string here…",
        label_visibility="collapsed",
    )

with col_hint:
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace; font-size:0.7rem; "
        "color:#7A9ABF; padding-top:0.65rem;'>"
        "Enter a SMILES <em>or</em> a generic name (e.g. "
        "<span style='color:#3D6A99;'>edaravone</span> or "
        "<span style='color:#3D6A99;'>CC1=NN(C(=O)C1)C2=CC=CC=C2</span>)"
        "</div>",
        unsafe_allow_html=True,
    )

smiles = None
source = None
if raw_smiles:
    try:
        smiles, source = resolve_smiles(raw_smiles)
    except InvalidSMILESError:
        st.error(
            f"**Could not resolve** `{raw_smiles}`. "
            "Enter a valid SMILES string or a recognised drug name (looked up via PubChem)."
        )

if smiles and source == "pubchem":
    st.caption(f"Resolved via PubChem → `{smiles}`")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Descriptors  ",
    "  CNS MPO  ",
    "  ML Predictions  ",
    "  Reference Drugs  ",
])

EMPTY_MSG = (
    "<div style='color:#7A9ABF; font-family:IBM Plex Mono,monospace; "
    "font-size:0.85rem; padding:2rem 0;'>Enter a SMILES above to continue.</div>"
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not smiles:
        st.markdown(EMPTY_MSG, unsafe_allow_html=True)
    else:
        try:
            profile = get_descriptor_profile(smiles)
        except InvalidSMILESError as e:
            st.error(str(e))
            profile = None

        if profile:
            # Descriptor metrics
            cols = st.columns(8)
            metrics = [
                ("MW (Da)",    f"{profile.mw}"),
                ("logP",       f"{profile.logp}"),
                ("logD (7.4)", _fmt_opt(profile.logd)),
                ("TPSA (Å²)",  f"{profile.tpsa}"),
                ("HBD",        f"{profile.hbd}"),
                ("HBA",        f"{profile.hba}"),
                ("QED",        f"{profile.qed}"),
                ("Rotatable Bonds",        f"{profile.rotbond}")
            ]
            for col, (label, val) in zip(cols, metrics):
                col.metric(label, val)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            if profile.pka_status == PKA_MATCHED:
                m = profile.pka_match
                st.info(
                    f"pKa {m.pka:g} (most basic centre) is an experimental value from the "
                    f"{m.source} (IUPAC rating: {m.assessment}). logD is calculated from "
                    "logP and this pKa."
                )
            elif profile.pka_status == PKA_DB_UNAVAILABLE:
                st.warning(
                    "The pKa lookup table could not be loaded, so pKa and logD are "
                    "unavailable."
                )
            else:
                st.info(
                    "No pKa match was found for this molecule in the IUPAC pKa dataset "
                    "(compiled from 1965-1979 reference works, so newer compounds are "
                    "often absent). pKa is not estimated as a substitute, and logD, which "
                    "is calculated from pKa, is therefore not available."
                )
            # Rule badges
            c1, c2 = st.columns(2)
            with c1:
                (st.success if profile.lipinski else st.warning)(
                    "✓ Lipinski Ro5 passed" if profile.lipinski else "✗ Lipinski Ro5 failed"
                )
            with c2:
                (st.success if profile.veber else st.warning)(
                    "✓ Veber's Rule passed" if profile.veber else "✗ Veber's Rule failed"
                )

            st.markdown("<hr>", unsafe_allow_html=True)

            # Radar chart - Lipinski space
            st.markdown(_label("Lipinski Space Radar"), unsafe_allow_html=True)

            limits = {"MW": 500, "logP": 5, "HBD": 5, "HBA": 10, "TPSA": 140, "RotBonds": 10}
            raw_vals = [
                profile.mw, profile.logp, profile.hbd,
                profile.hba, profile.tpsa, profile.rotbond,
            ]
            norms  = [min(v / l, 1.35) for v, l in zip(raw_vals, limits.values())]
            labels = list(limits.keys())
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            norms_c  = norms  + norms[:1]
            angles_c = angles + angles[:1]

            fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
            ax.set_facecolor("#F5E6EE")
            fig.patch.set_facecolor("#F5E6EE")
            ax.plot(angles_c, norms_c, color="#3D6A99", lw=1.8)
            ax.fill(angles_c, norms_c, color="#3D6A99", alpha=0.12)
            # Ro5 boundary
            ax.plot(angles_c, [1.0] * len(angles_c),
                    color="#9B2335", lw=0.9, linestyle="--", alpha=0.6, label="Ro5 limit")
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, size=8.5, color="#496D99")
            ax.set_yticklabels([])
            ax.set_ylim(0, 1.4)
            ax.spines["polar"].set_color("#5d6987")
            ax.grid(color="#E4C5D4", linewidth=0.7)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
            plt.tight_layout()

            rc, rc2 = st.columns([1, 2])
            with rc:
                st.pyplot(fig)
            with rc2:
                st.markdown("""
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem;
                            color:#6a7a90; padding-top:1.8rem; line-height:1.75;">
                Each axis is normalised to its Lipinski upper limit.<br>
                The <span style='color:#9B2335;'>dashed red boundary</span> represents
                full Ro5 compliance. Values extending beyond it indicate that property
                exceeds its Lipinski threshold.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # Tanimoto
            st.markdown(_label("Tanimoto Similarity"), unsafe_allow_html=True)
            smiles2 = st.text_input(
                "tanimoto_input",
                placeholder="Enter a second SMILES to compare…",
                label_visibility="collapsed",
                key="tanimoto_input",
            )
            if smiles2:
                try:
                    sim = get_tanimoto(smiles, smiles2)
                    st.metric("Tanimoto Index (Morgan r=2, 2048 bits)", f"{sim:.4f}")
                    st.progress(float(sim))
                    if sim >= 0.85:
                        st.success("Very high similarity, likely same or closely related scaffold.")
                    elif sim >= 0.60:
                        st.info("Moderate similarity, shared structural features.")
                    elif sim >= 0.40:
                        st.warning("Low similarity, structurally distinct molecules.")
                    else:
                        st.warning("Very low similarity, essentially unrelated structures.")
                except InvalidSMILESError as e:
                    st.error(str(e))

        # ── CNS Drug Similarity Panel ─────────────────────────────────────────
        if smiles:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(_label("Automated CNS Drug Similarity Panel"), unsafe_allow_html=True)
            st.markdown(
                "<div style='font-family:IBM Plex Sans,sans-serif; font-size:0.82rem; "
                "color:#6a7a90; margin-bottom:1rem;'>"
                "Tanimoto similarity against a curated library of nearly 90 approved CNS drugs "
                "spanning psychiatry, neurology, pain, anaesthesia and addiction medicine. "
                "Identifies the most structurally related known CNS compounds.</div>",
                unsafe_allow_html=True,
            )
            try:
                panel = get_cns_tanimoto_panel(smiles, top_n=10)

                # Colour-coded similarity bar chart
                names_p = [f"{d['name']} ({d['category']})" for d in panel]
                sims_p  = [d["similarity"] for d in panel]
                bar_c_p = [
                    "#2D7A4F" if s >= 0.6 else "#8B6914" if s >= 0.35 else "#7A9ABF"
                    for s in sims_p
                ]
                fig, ax = plt.subplots(figsize=(8, 3.8))
                bars = ax.barh(names_p[::-1], sims_p[::-1], color=bar_c_p[::-1],
                               height=0.55, zorder=2)
                ax.axvline(0.85, color="#2D7A4F", linestyle="--", lw=0.8, alpha=0.6,
                           label="≥ 0.85 very high")
                ax.axvline(0.60, color="#8B6914", linestyle="--", lw=0.8, alpha=0.6,
                           label="≥ 0.60 moderate")
                ax.axvline(0.35, color="#7A9ABF", linestyle="--", lw=0.8, alpha=0.5,
                           label="< 0.35 low")
                ax.set_xlim(0, 1.05)
                ax.set_xlabel("Tanimoto similarity")
                ax.set_title("Top 10 most similar CNS drugs")
                ax.grid(axis="x", zorder=0)
                ax.legend(fontsize=7, loc="lower right")
                for bar, s in zip(bars, sims_p[::-1]):
                    ax.text(s + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{s:.3f}", va="center", fontsize=8, color="#496D99")
                plt.tight_layout()
                st.pyplot(fig)

                # Detail table
                rows_p = []
                for d in panel:
                    s = d["similarity"]
                    if s >= 0.85:
                        interp = "Very high, likely same scaffold"
                    elif s >= 0.60:
                        interp = "Moderate, shared structural features"
                    elif s >= 0.35:
                        interp = "Low, distant structural relationship"
                    else:
                        interp = "Very low, structurally distinct"
                    rows_p.append({
                        "Drug":        d["name"],
                        "Category":    d["category"],
                        "Indication":  d["indication"],
                        "MOA":         d["moa"],
                        "BBB mechanism": d["bbb_mechanism"],
                        "Similarity":  f"{s:.4f}",
                        "Interpretation": interp,
                    })
                st.dataframe(
                    pd.DataFrame(rows_p),
                    use_container_width=True,
                    hide_index=True,
                )

                # Highlight any very high matches
                top_match = panel[0]
                if top_match["similarity"] >= 0.85:
                    st.success(
                        f"Very high structural similarity to {top_match['name']} "
                        f"(Tanimoto = {top_match['similarity']:.3f}). "
                        f"This molecule closely resembles a known {top_match['category'].lower()}; "
                        f"{top_match['moa']}."
                    )
                elif top_match["similarity"] >= 0.60:
                    st.info(
                        f"Moderate similarity to {top_match['name']} "
                        f"(Tanimoto = {top_match['similarity']:.3f}), "
                        f"a {top_match['category'].lower()} ({top_match['moa']})."
                    )
                elif top_match["similarity"] < 0.35:
                    st.warning(
                        f"Low similarity to all reference CNS drugs (highest: "
                        f"{top_match['name']} at {top_match['similarity']:.3f}). "
                        "This molecule may be structurally novel relative to approved CNS drugs."
                    )

            except InvalidSMILESError as e:
                st.error(str(e))

        with st.expander("Reference / descriptor definitions"):
            st.markdown("""
            | Descriptor | Definition | Lipinski / Veber threshold |
            |---|---|---|
            | MW | Molecular weight (Da) | < 500 (Lipinski) |
            | logP | Octanol-water partition coefficient (lipophilicity) | < 5 (Lipinski) |
            | logD | logP corrected for ionisation at pH 7.4, from the database pKa (n/a without one) | ≤ 2 for CNS (MPO) |
            | TPSA | Topological polar surface area (Å²) | ≤ 140 (Veber) |
            | HBD | H-bond donors | ≤ 5 (Lipinski) |
            | HBA | H-bond acceptors | ≤ 10 (Lipinski) |
            | QED | Quantitative estimate of drug-likeness (0-1) | > 0.5 considered drug-like |
            | RotBonds | Rotatable bonds | ≤ 10 (Veber) |
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - CNS MPO
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        _label("Wager et al., ACS Chem. Neurosci. 2010, 1, 435-449"),
        unsafe_allow_html=True,
    )

    if not smiles:
        st.markdown(EMPTY_MSG, unsafe_allow_html=True)
    else:
        try:
            mpo = get_cns_mpo(smiles)
        except InvalidSMILESError as e:
            st.error(str(e))
            mpo = None

        if mpo:
            mpo_frac    = mpo.total / mpo.max_total
            mpo_partial = mpo.max_total < 6
            mpo_cutoff  = 4.0 * mpo.max_total / 6.0   # 4.0 of 6, scaled to the terms scored
            score_color = (
                "#2D7A4F" if mpo.cns_optimised
                else "#8B6914" if mpo_frac >= 2.5 / 6
                else "#9B2335"
            )

            # Headline score
            st.markdown(f"""
            <div style="display:flex; align-items:baseline; gap:1.5rem; margin-bottom:1.5rem; flex-wrap:wrap;">
                <div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                                text-transform:uppercase; letter-spacing:0.1em; color:#7A9ABF;">
                        CNS MPO Score
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:3rem;
                                font-weight:500; color:{score_color}; line-height:1.05;">
                        {mpo.total:.2f}
                        <span style="font-size:1.1rem; color:#7A9ABF;"> / {mpo.max_total:.2f}</span>
                    </div>
                </div>
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.92rem;
                            color:{'#2D7A4F' if mpo.cns_optimised else '#9B2335'};
                            align-self:center;">
                    {f'✓ CNS-optimised (threshold ≥ {mpo_cutoff:.2f})' if mpo.cns_optimised
                     else f'✗ Not CNS-optimised (below threshold of {mpo_cutoff:.2f})'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Per-property bar chart
            props      = list(mpo.per_property.keys())
            scores     = list(mpo.per_property.values())
            raws       = list(mpo.raw_values.values())
            bar_colors = [
                "#2D7A4F" if s == 1.0
                else "#8B6914" if s >= 0.5
                else "#9B2335"
                for s in scores
            ]

            fig, ax = plt.subplots(figsize=(7.5, 3.4))
            bars = ax.barh(props, scores, color=bar_colors, height=0.52, zorder=2)
            ax.set_xlim(0, 1.28)
            ax.set_xlabel("Contribution (0 → 1)")
            ax.set_title("CNS MPO - Per-property Contributions")
            ax.axvline(1.0, color="#5d6987", linestyle="--", linewidth=0.9)
            ax.grid(axis="x", zorder=0)
            for bar, s, rv in zip(bars, scores, raws):
                ax.text(
                    s + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{s:.3f}   (raw: {rv})",
                    va="center", fontsize=8.5, color="#496D99",
                )
            plt.tight_layout()
            st.pyplot(fig)

            if mpo.failed_properties:
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; "
                    f"color:#8B6914; margin-top:0.4rem;'>"
                    f"Sub-optimal properties: {', '.join(mpo.failed_properties)}</div>",
                    unsafe_allow_html=True,
                )

            if mpo_partial:
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; "
                    f"color:#7A9ABF; margin-top:0.3rem;'>"
                    f"Not scored: {', '.join(mpo.missing_properties)} "
                    f"(partial MPO over {int(mpo.max_total)} of 6 properties)</div>",
                    unsafe_allow_html=True,
                )

            # ── Dynamic explanation ───────────────────────────────────────────
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            st.markdown(_label("Structural interpretation"), unsafe_allow_html=True)
            expl = explain_cns_mpo(mpo)
            sev_color = {"good": "#2D7A4F", "moderate": "#8B6914", "poor": "#9B2335"}[expl["severity"]]
            st.markdown(
                f"<div style='font-family:IBM Plex Sans,sans-serif; font-size:0.9rem; "
                f"color:{sev_color}; margin-bottom:0.8rem;'>{expl['headline']}</div>",
                unsafe_allow_html=True,
            )
            if not expl["all_optimal"]:
                for para in expl["paragraphs"]:
                    st.markdown(
                        f"<div style='background:#EDD3DF; border:1px solid #C9A8BB; "
                        f"border-left:3px solid {sev_color}; border-radius:4px; "
                        f"padding:0.8rem 1rem; margin-bottom:0.5rem; "
                        f"font-family:IBM Plex Sans,sans-serif; font-size:0.83rem; "
                        f"color:#496D99; line-height:1.7;'>{para}</div>",
                        unsafe_allow_html=True,
                    )
            if expl["optimisation"] and not expl["all_optimal"]:
                with st.expander("💡 Structural optimisation suggestions"):
                    for sug in expl["optimisation"]:
                        st.markdown(f"- {sug}")

            for note in expl["notes"]:
                st.info(note)

        with st.expander("Reference / CNS MPO desirability functions"):
            st.markdown("""
            Each property contributes 0-1 via a piecewise-linear desirability function:

            | Property | Ideal (score=1) | Penalised zone | Score=0 |
            |---|---|---|---|
            | MW | ≤ 360 Da | 360-500 | > 500 |
            | logP | ≤ 3 | 3-5 | > 5 |
            | logD (pH 7.4) | ≤ 2 | 2-4 | > 4 |
            | TPSA | 40-90 Å² | 20-40 / 90-120 | < 20 or > 120 |
            | HBD | 0-1 | 2 (= 0.5) | ≥ 3 |
            | pKa (most basic) | ≤ 8 | 8-10 | > 10 |

            > Note on pKa and logD: pKa is taken only from the IUPAC Digitized pKa
            > Dataset (experimental values, matched by structure); it is never
            > estimated. logD is calculated from logP and that pKa. When the
            > dataset has no entry for a molecule, both terms are left out and the
            > score is taken over MW, logP, TPSA and HBD, with the optimised
            > threshold scaled to the same proportion (2.67 of 4 in place of 4.0 of 6).
            > A partial score is not directly comparable to a six-property one.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    if not smiles:
        st.markdown(EMPTY_MSG, unsafe_allow_html=True)
    else:
        # ── BBB + ClinTox side by side ────────────────────────────────────────
        col_bbb, col_tox = st.columns(2)

        bbb_result = tox_result = None

        with col_bbb:
            st.markdown(_label("BBB Permeability - BBBP dataset"), unsafe_allow_html=True)
            if bbbp_model is None:
                st.error("BBBP model not available. Run `python ml_model.py` first.")
            else:
                try:
                    bbb_result = predict_bbbp(smiles, bbbp_model)
                    p = bbb_result.probability
                    color = "#2D7A4F" if bbb_result.predicted else "#9B2335"
                    st.markdown(f"""
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
                                font-weight:500; color:{color};">{p:.1%}</div>
                    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.88rem;
                                color:{color}; margin-bottom:0.8rem;">
                        {bbb_result.label}
                        <span style="color:#7A9ABF; font-size:0.75rem;">
                        &nbsp;· confidence: {bbb_result.confidence}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(p, text=f"P(BBB-permeable) = {p:.4f}")
                    # Dynamic explanation
                    mpo_for_expl = None
                    try:
                        mpo_for_expl = get_cns_mpo(smiles)
                    except Exception:
                        pass
                    bbb_expl = explain_bbbp(bbb_result, mpo_for_expl)
                    bbb_sev_color = {"good": "#2D7A4F", "moderate": "#8B6914", "poor": "#9B2335"}[bbb_expl["severity"]]
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    for line in bbb_expl["body"]:
                        st.markdown(
                            f"<div style='background:#EDD3DF; border:1px solid #C9A8BB; "
                            f"border-left:3px solid {bbb_sev_color}; border-radius:4px; "
                            f"padding:0.7rem 0.9rem; margin-bottom:0.4rem; "
                            f"font-family:IBM Plex Sans,sans-serif; font-size:0.8rem; "
                            f"color:#496D99; line-height:1.65;'>{line}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"<div style='font-family:IBM Plex Mono,monospace; font-size:0.7rem; "
                        f"color:#7A9ABF; margin-top:0.4rem; font-style:italic;'>"
                        f"⚠ {bbb_expl['caveat']}</div>",
                        unsafe_allow_html=True,
                    )
                except (InvalidSMILESError, PredictionError) as e:
                    st.error(f"Prediction failed: {e}")

        with col_tox:
            st.markdown(_label("Clinical Toxicity - ClinTox dataset"), unsafe_allow_html=True)
            if clintox_model is None:
                st.error("ClinTox model not available. Run `python ml_model.py` first.")
            else:
                try:
                    tox_result = predict_clintox(smiles, clintox_model)
                    p = tox_result.probability
                    color = "#9B2335" if tox_result.predicted else "#2D7A4F"
                    st.markdown(f"""
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
                                font-weight:500; color:{color};">{p:.1%}</div>
                    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.88rem;
                                color:{color}; margin-bottom:0.8rem;">
                        {tox_result.label}
                        <span style="color:#7A9ABF; font-size:0.75rem;">
                        &nbsp;· confidence: {tox_result.confidence}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(p, text=f"P(clinical toxicity) = {p:.4f}")
                    # Dynamic explanation
                    mpo_for_tox = None
                    try:
                        mpo_for_tox = get_cns_mpo(smiles)
                    except Exception:
                        pass
                    tox_expl = explain_clintox(tox_result, mpo_for_tox, clintox_model)
                    tox_sev_color = {"good": "#2D7A4F", "moderate": "#8B6914", "poor": "#9B2335"}[tox_expl["severity"]]
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    for line in tox_expl["body"]:
                        st.markdown(
                            f"<div style='background:#EDD3DF; border:1px solid #C9A8BB; "
                            f"border-left:3px solid {tox_sev_color}; border-radius:4px; "
                            f"padding:0.7rem 0.9rem; margin-bottom:0.4rem; "
                            f"font-family:IBM Plex Sans,sans-serif; font-size:0.8rem; "
                            f"color:#496D99; line-height:1.65;'>{line}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"<div style='font-family:IBM Plex Mono,monospace; font-size:0.7rem; "
                        f"color:#7A9ABF; margin-top:0.4rem; font-style:italic;'>"
                        f"⚠ {tox_expl['caveat']}</div>",
                        unsafe_allow_html=True,
                    )
                except (InvalidSMILESError, PredictionError) as e:
                    st.error(f"Prediction failed: {e}")

        # ── Combined CNS verdict ──────────────────────────────────────────────
        if bbb_result or tox_result:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(_label("Combined CNS Candidate Assessment"), unsafe_allow_html=True)

            try:
                mpo_v = get_cns_mpo(smiles)
            except InvalidSMILESError:
                mpo_v = None

            bbb_pass = bbb_result.predicted if bbb_result else False
            tox_pass = (not tox_result.predicted) if tox_result else False
            mpo_pass = mpo_v.cns_optimised if mpo_v else False

            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                detail = f"{bbb_result.probability:.1%}" if bbb_result else "N/A"
                st.markdown(_verdict_card("BBB Permeable", bbb_pass, detail), unsafe_allow_html=True)
            with vc2:
                detail = f"P(tox)={tox_result.probability:.1%}" if tox_result else "N/A"
                st.markdown(_verdict_card("Low Toxicity Risk", tox_pass, detail), unsafe_allow_html=True)
            with vc3:
                detail = f"MPO={mpo_v.total:.2f}/{mpo_v.max_total:.0f}" if mpo_v else "N/A"
                mpo_label = "CNS MPO ≥ 4.0"
                if mpo_v and mpo_v.max_total < 6:
                    mpo_label = f"CNS MPO ≥ {4.0 * mpo_v.max_total / 6:.2f} (partial)"
                st.markdown(_verdict_card(mpo_label, mpo_pass, detail), unsafe_allow_html=True)

            n_pass = sum([bbb_pass, tox_pass, mpo_pass])
            verdict_map = {
                3: ("Promising CNS candidate, all three criteria met.", "#2D7A4F"),
                2: ("Moderate CNS profile, passes 2/3 criteria; consider optimisation.", "#8B6914"),
                1: ("Weak CNS profile, significant structural liabilities present.", "#B07040"),
                0: ("Poor CNS candidate, fails all three criteria.", "#9B2335"),
            }
            vtext, vcolor = verdict_map[n_pass]
            st.markdown(
                f"<div style='margin-top:1rem; font-family:IBM Plex Sans,sans-serif; "
                f"font-size:0.9rem; color:{vcolor};'>{n_pass}/3 - {vtext}</div>",
                unsafe_allow_html=True,
            )
            if mpo_v and mpo_v.max_total < 6:
                st.caption(
                    "CNS MPO here is partial (no pKa match in the IUPAC dataset, so logD "
                    "and pKa were not scored). Read the third criterion with that in mind."
                )

        with st.expander("Methodology (how the ML models work)"):
            st.markdown("""
            Both models follow the same pipeline:

            1. Input: SMILES → RDKit Morgan fingerprint (radius=2, 2048 bits)  
            2. Model: `sklearn.RandomForestClassifier` (150 trees, `class_weight='balanced'`)  
            3. Split: 80/20 stratified random split (seed=42)  
            4. Evaluation: ROC-AUC, precision, recall, F1 on held-out test set  

            BBBP dataset (Martins et al., J. Chem. Inf. Model. 2012): ~2000 molecules  
            with binary BBB permeability labels derived from in vivo and in vitro experiments.  

            ClinTox dataset (Gayvert et al., Cell Chem. Biol. 2016): ~1500 drug compounds  
            labelled by FDA approval status and clinical trial toxicity outcome.  

            Limitation: Both models use 2D structural fingerprints only. They cannot encode  
            active transport mechanisms, P-gp efflux, protein binding, or metabolic activation.  
            Predictions should be treated as rapid computational filters, not definitive assessments.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - REFERENCE DRUGS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        _label("Curated CNS drug panel (model agreement vs. known pharmacology)"),
        unsafe_allow_html=True,
    )

    rows = []
    mpo_frac_by_drug = {}   # name -> (fraction of max score, is_partial)
    for drug in CNS_REFERENCE_DRUGS:
        bbbp_r = clintox_r = mpo_r = None

        try:
            if bbbp_model:
                bbbp_r = predict_bbbp(drug["smiles"], bbbp_model)
        except (InvalidSMILESError, PredictionError) as e:
            logger.warning("Reference drug '%s' BBB failed: %s", drug["name"], e)

        try:
            if clintox_model:
                clintox_r = predict_clintox(drug["smiles"], clintox_model)
        except (InvalidSMILESError, PredictionError) as e:
            logger.warning("Reference drug '%s' ClinTox failed: %s", drug["name"], e)

        try:
            mpo_r = get_cns_mpo(drug["smiles"])
        except InvalidSMILESError as e:
            logger.warning("Reference drug '%s' CNS MPO failed: %s", drug["name"], e)

        known = drug["known_bbb_permeable"]
        pred  = bbbp_r.predicted if bbbp_r else None
        exp   = drug["expected_model_agrees"]

        if exp is None:
            agrees_str = "Ambiguous"
        elif pred is None:
            agrees_str = "N/A"
        elif pred == known:
            agrees_str = "✓ Agrees"
        else:
            agrees_str = "✗ Disagrees"

        rows.append({
            "Drug":           drug["name"],
            "Indication":     drug["indication"],
            "Known BBB":      "Yes" if known else "No",
            "BBB mechanism":  drug["bbb_mechanism"],
            "Predicted BBB":  f"{bbbp_r.probability:.2f}" if bbbp_r else "-",
            "Model agrees":   agrees_str,
            "Tox P(tox)":     f"{clintox_r.probability:.2f}" if clintox_r else "-",
            "CNS MPO":        f"{mpo_r.total:.2f}/{mpo_r.max_total:.0f}" if mpo_r else "-",
            "pKa (IUPAC)":    (f"{mpo_r.raw_pka:g}" if mpo_r and mpo_r.raw_pka is not None
                               else "not found"),
        })
        mpo_frac_by_drug[drug["name"]] = (
            (mpo_r.total / mpo_r.max_total, mpo_r.max_total < 6) if mpo_r else (0.0, False)
        )

    df_ref = pd.DataFrame(rows)
    st.dataframe(df_ref, use_container_width=True, hide_index=True)

    # CNS MPO bar chart for reference panel
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(_label("CNS MPO scores / reference drug panel"), unsafe_allow_html=True)

    names_r  = [r["Drug"] for r in rows]
    fracs    = [mpo_frac_by_drug[n][0] for n in names_r]
    partial  = [mpo_frac_by_drug[n][1] for n in names_r]
    labels_r = [r["CNS MPO"] for r in rows]

    # Scores are plotted as a fraction of each molecule's maximum so 4-term
    # (no pKa match) and 6-term scores share one axis. The optimised cut-off
    # is 4.0/6 = 2/3 of the maximum either way. Partial scores are hatched.
    bar_cols = [
        "#2D7A4F" if f >= 4 / 6 else "#8B6914" if f >= 2.5 / 6 else "#9B2335"
        for f in fracs
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names_r, fracs, color=bar_cols, width=0.6, zorder=2)
    for bar, is_partial in zip(bars, partial):
        if is_partial:
            bar.set_hatch("//")
            bar.set_edgecolor("white")
    ax.axhline(4 / 6, color="#9B2335", linestyle="--", lw=1.2, alpha=0.7,
               label="CNS-optimised threshold (≥ 4.0 of 6, i.e. ≥ 67% of max)")
    ax.set_ylabel("CNS MPO (fraction of max score)")
    ax.set_title("CNS MPO - Reference Drug Panel (hatched = partial, no pKa match)")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=8)
    plt.xticks(rotation=22, ha="right", fontsize=9)
    for bar, f, lab in zip(bars, fracs, labels_r):
        ax.text(
            bar.get_x() + bar.get_width() / 2, f + 0.01,
            lab, ha="center", va="bottom", fontsize=8.5, color="#496D99",
        )
    plt.tight_layout()
    st.pyplot(fig)

    # Key mechanistic discussion notes
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(_label("Mechanistic discussion points"), unsafe_allow_html=True)

    for drug in CNS_REFERENCE_DRUGS:
        if drug["expected_model_agrees"] is False:
            color = "#8B6914"
            icon  = "⚠️"
        elif drug["expected_model_agrees"] is None:
            color = "#6a7a90"
            icon  = "🔬"
        else:
            continue  # only show interesting cases

        st.markdown(f"""
        <div style="background:#EDD3DF; border:1px solid #C9A8BB;
                    border-left:3px solid {color}; border-radius:4px;
                    padding:0.9rem 1.1rem; margin-bottom:0.6rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                        color:{color}; margin-bottom:0.3rem;">
                {icon} {drug['name']} - {drug['indication']}
            </div>
            <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem;
                        color:#496D99; line-height:1.7;">
                <strong style="color:#2D4A6B;">Mechanism:</strong> {drug['bbb_mechanism']}<br>
                {drug['discussion_point']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Why include known drugs in the analysis?"):
        st.markdown("""
        Benchmarking against molecules with established clinical profiles serves two purposes:

        **1. Sanity-checking the model**: drugs like donepezil, sertraline and caffeine are
        textbook CNS compounds predicted correctly with high confidence. If the model failed
        these, it would signal a data or training problem.

        **2. Revealing mechanistic blind spots**: levodopa is the clearest failure in this panel.
        It crosses the BBB via the LAT1 large amino acid transporter, yet the model predicts
        low permeability (P ≈ 0.3085) because its polar, zwitterionic catechol-amino acid scaffold
        carries fingerprint bits associated with non-permeable compounds. Gabapentin is also a
        LAT1 substrate but is correctly predicted as permeable (P ≈ 0.93), likely because its
        cyclohexane ring contributes enough lipophilic bits to superficially resemble a passively
        diffusing molecule. The model gets the right answer for the wrong structural reason.
        Levodopa has no such compensating features, making it the purest example of a
        mechanism-invisible failure in this dataset.

        **3. Demonstrating scientific judgement**: knowing when not to trust your model is
        as important as knowing how to build it.
        """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1.2rem; border-top:1px solid #C9A8BB;
            font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#5d6987;
            display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
    <span>By Efthalia Arvanitidou</span>
</div>
""", unsafe_allow_html=True)
