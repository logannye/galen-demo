"""SCM Off-Target Safety — Streamlit demo.

Launch:
  cd ~/Desktop/SCM\ Off-Target\ Safety
  streamlit run scripts/demo/streamlit_app.py

Biopharma use case:
  A medicinal chemist or computational biologist evaluating a preclinical
  drug candidate. Inputs (in priority order):
    1. SMILES of the candidate molecule (most common)
    2. Drug name (for reference/validation against known drugs)
    3. Explicit binding profile (for compounds with panel-screen data)

  Outputs:
    - Ranked side-effect predictions
    - Per-target attribution: which binding targets are driving each
      predicted side effect (so the medicinal chemist can engineer out
      the worst offenders)
    - Nearest known drugs in chemistry space (anchoring the prediction)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root
WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

import pandas as pd
import streamlit as st

from scripts.demo.predict import PredictionEngine, PredictionResult


@st.cache_resource(show_spinner="Loading SCM Off-Target Safety engine...")
def get_engine() -> PredictionEngine:
    return PredictionEngine()


def render_binding_profile(bp: list[dict], is_inferred: bool) -> None:
    if not bp:
        st.warning("No binding profile available.")
        return
    header = ("Inferred binding profile" if is_inferred
              else "Measured binding profile (ChEMBL)")
    st.subheader(header)
    rows = []
    for t in bp[:30]:
        row = {
            "Gene": t.get("gene_symbol", t.get("uniprot", "?")),
            "UniProt": t.get("uniprot", ""),
            "Target": (t.get("target_pref_name", "") or "")[:50],
            "Affinity (nM)": f"{t.get('standard_value_nm', 0):.2f}",
            "Type": t.get("standard_type", "?"),
        }
        if is_inferred:
            row["Support (NN)"] = t.get("n_supporting_neighbors", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_nearest_drugs(nn: list[dict]) -> None:
    if not nn:
        return
    st.subheader("Structural neighbors (chemistry anchor)")
    st.caption(
        "These are the most-similar drugs in the catalog by ECFP4 Tanimoto. "
        "Your compound's predicted profile is a similarity-weighted aggregate "
        "of their binding."
    )
    df = pd.DataFrame([
        {
            "Drug": d["drug_name"],
            "CID": d["cid"],
            "Tanimoto": f"{d['tanimoto']:.3f}",
            "# Targets": d.get("n_targets", ""),
            "# Side effects": d.get("n_side_effects", ""),
        }
        for d in nn
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_explanations(result: PredictionResult) -> None:
    if not result.explanations:
        st.warning("No predictions returned.")
        return
    st.subheader(f"Top {len(result.explanations)} predicted side effects with per-target attribution")
    st.caption(
        "Each predicted side effect is decomposed into the percent contribution "
        "from each binding target. This shows WHICH targets are driving each "
        "prediction — useful for SAR optimization or off-target engineering."
    )

    # Summary table
    rows = []
    for e in result.explanations:
        row = {
            "Rank": e.rank,
            "Side effect": e.side_effect_name,
            "UMLS": e.side_effect_id,
            "SCM score": f"{e.scm_score:.3f}",
            "Top target": e.top_targets[0].gene_symbol if e.top_targets else "—",
            "Top target %": (f"{e.top_targets[0].contribution_pct:.0%}"
                              if e.top_targets else "—"),
        }
        if e.in_sider_gold is True:
            row["Match"] = "✓ in SIDER"
        elif e.in_sider_gold is False:
            row["Match"] = "novel"
        else:
            row["Match"] = "—"
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Detailed per-side-effect attribution
    with st.expander("Per-side-effect target attribution (drill-down)"):
        for e in result.explanations:
            st.markdown(
                f"**#{e.rank} — {e.side_effect_name}** "
                f"(UMLS `{e.side_effect_id}`, score={e.scm_score:.3f})"
            )
            if e.in_sider_gold is True:
                st.success("This side effect IS documented in SIDER for this drug.")
            elif e.in_sider_gold is False:
                st.info("Predicted but NOT in this drug's SIDER label.")
            attribution_df = pd.DataFrame([
                {
                    "Gene": t.gene_symbol,
                    "UniProt": t.uniprot,
                    "Contribution %": t.contribution_pct,
                    "α (training)": t.alpha,
                    "Affinity (nM)": t.standard_value_nm,
                }
                for t in e.top_targets
            ])
            if not attribution_df.empty:
                st.bar_chart(
                    attribution_df.set_index("Gene")["Contribution %"],
                    use_container_width=True,
                )
                st.dataframe(
                    attribution_df.style.format({
                        "Contribution %": "{:.0%}", "α (training)": "{:.3f}",
                        "Affinity (nM)": "{:.2f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
            st.divider()


def main() -> None:
    st.set_page_config(
        page_title="SCM Off-Target Safety",
        page_icon="💊", layout="wide",
    )
    st.title("💊 SCM Off-Target Safety")
    st.markdown(
        "Mechanism-interpretable side-effect prediction from polypharmacology "
        "binding profiles. Built on a structural causal model trained on "
        "SIDER 4.1 ∩ ChEMBL 36 (n=247 training drugs, 833 targets, 500 "
        "side-effect vocabulary)."
    )

    with st.sidebar:
        st.header("Configuration")
        top_k_se = st.slider(
            "Top-K side effects to display", min_value=5, max_value=50,
            value=15, step=5,
        )
        top_k_targets = st.slider(
            "Top-K contributing targets per side effect", min_value=3,
            max_value=10, value=5, step=1,
        )
        st.markdown("---")
        st.header("Performance benchmarks")
        st.markdown(
            "**Per pre-registered Sprint 1+2 results (n=200):**\n"
            "- SCM AP: 0.571 (matches RF-ECFP chemistry SOTA)\n"
            "- SCM P@10: 0.808\n"
            "- vs LLM-drug-blind: SCM +0.358 AP, +0.090 P@10 (p≈0)\n"
            "- vs LLM-with-name: SCM matches at P@10\n"
            "- Robust to target-disjointness (p=6.3×10⁻⁷)\n\n"
            "**Architectural moat:**\n"
            "- Per-target attribution (no black-box ML method has this)\n"
            "- Statistical mechanism interpolation\n"
            "- 30%+ relative outperformance over LLMs"
        )

    engine = get_engine()

    tab_name, tab_smiles, tab_profile = st.tabs([
        "🏷️ Drug name", "🧬 SMILES (novel compound)",
        "🎯 Custom binding profile",
    ])

    result: PredictionResult | None = None

    with tab_name:
        st.markdown("Look up a drug by name (must be in SIDER ∩ ChEMBL catalog).")
        drug_name = st.text_input(
            "Drug name", value="atorvastatin", key="drug_name_input",
        )
        if st.button("Predict", key="predict_name", type="primary"):
            with st.spinner(f"Looking up {drug_name}..."):
                result = engine.predict_from_drug_name(
                    drug_name, top_k_se=top_k_se,
                    top_k_targets=top_k_targets,
                )

    with tab_smiles:
        st.markdown(
            "Paste a SMILES string for a novel compound. The binding profile "
            "will be inferred via Tanimoto nearest-neighbors against the "
            "training catalog."
        )
        st.caption(
            "*The chemist's question*: given my new lead compound's structure, "
            "what side effects would the SCM predict, and which targets in its "
            "inferred binding profile are driving each prediction?"
        )
        smiles = st.text_area(
            "SMILES",
            value="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # albuterol
            height=80, key="smiles_input",
        )
        col_k, col_t = st.columns(2)
        with col_k:
            k_neighbors = st.slider("Number of structural neighbors",
                                       min_value=1, max_value=20, value=5)
        with col_t:
            min_tanimoto = st.slider("Minimum Tanimoto similarity",
                                       min_value=0.10, max_value=0.90,
                                       value=0.30, step=0.05)
        if st.button("Predict", key="predict_smiles", type="primary"):
            with st.spinner("Computing fingerprint + nearest neighbors..."):
                try:
                    result = engine.predict_from_smiles(
                        smiles, k_neighbors=k_neighbors,
                        min_tanimoto=min_tanimoto,
                        top_k_se=top_k_se, top_k_targets=top_k_targets,
                    )
                except ValueError as e:
                    st.error(str(e))

    with tab_profile:
        st.markdown(
            "Paste a custom binding profile (e.g., from a Eurofins / DiscoverX "
            "safety panel screen). Format: one target per line, "
            "`UNIPROT,GENE,Ki_nM`."
        )
        default_profile = (
            "P04035,HMGCR,0.5\n"
            "Q9Y6L6,SLCO1B1,800\n"
            "Q9UNQ0,ABCG2,5000\n"
            "P08684,CYP3A4,2000"
        )
        profile_text = st.text_area("Binding profile", value=default_profile,
                                       height=140, key="profile_input")
        if st.button("Predict", key="predict_profile", type="primary"):
            bp = []
            for line in profile_text.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3 and parts[0]:
                    try:
                        bp.append({
                            "uniprot": parts[0],
                            "gene_symbol": parts[1] if len(parts) > 1 else parts[0],
                            "target_pref_name": "",
                            "standard_type": "Ki",
                            "standard_value_nm": float(parts[2]),
                        })
                    except ValueError:
                        st.warning(f"skipping malformed line: {line}")
            if bp:
                result = engine.predict_from_binding_profile(
                    bp, top_k_se=top_k_se, top_k_targets=top_k_targets,
                )
            else:
                st.error("No valid targets parsed.")

    if result is not None:
        st.markdown("---")
        st.info(result.note)
        if result.is_inferred and result.nearest_drugs:
            render_nearest_drugs(result.nearest_drugs)
        render_binding_profile(result.binding_profile, result.is_inferred)
        render_explanations(result)


if __name__ == "__main__":
    main()
