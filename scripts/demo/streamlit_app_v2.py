"""SCM Off-Target Safety — Clinical Safety Prediction Demo (v2, Sprint 6).

Launch:
  cd ~/Desktop/SCM\\ Off-Target\\ Safety
  streamlit run scripts/demo/streamlit_app_v2.py

The biopharma-ready demo of the Hybrid SCM+LLM clinical-safety prediction
system. Optimized for BD conversations and medicinal-chemistry workflows.

Primary mode (Clinical Safety): Hybrid SCM+LLM ranks top-20 most-likely
safety liabilities for a novel preclinical compound with:
  - per-prediction mechanism rationale (from LLM reasoning)
  - per-prediction top-3 binding-target attribution (from SCM)
  - per (target, side-effect) multi-source evidence trace (CTD, OpenTargets
    FAERS, PharmGKB, AOP-Wiki, SIDER)

Validated on n=51 clinical-safety benchmark (Sprint 5):
  - hit@3: 47% (vs LLM-with-name 31%, LLM-drug-blind 27%)
  - hit@10: 57% (McNemar p=0.006 vs LLM-drug-blind)
  - Beats LLM-with-name despite drug-blind operation.
"""
from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

import pandas as pd
import streamlit as st

from scripts.demo.predict_hybrid import ClinicalSafetyEngine, ClinicalSafetyResult


@st.cache_resource(show_spinner="Loading SCM Off-Target Safety engine (may take ~10s on first run)...")
def get_engine() -> ClinicalSafetyEngine:
    return ClinicalSafetyEngine()


_SOURCE_BADGES = {
    "SIDER":              ("🔵", "blue"),
    "CTD":                ("🟢", "green"),
    "OpenTargets (FAERS)": ("🟣", "violet"),
    "PharmGKB":           ("🟠", "orange"),
    "AOP-Wiki":           ("🔴", "red"),
}


def render_binding_profile(bp: list[dict], is_inferred: bool) -> None:
    header = ("Inferred binding profile (Tanimoto NN)" if is_inferred
              else "Measured binding profile (ChEMBL)")
    st.markdown(f"#### {header}  ·  {len(bp)} targets")
    if not bp:
        st.warning("No binding profile available.")
        return
    rows = []
    for t in bp[:30]:
        rows.append({
            "Gene": t.get("gene_symbol", t.get("uniprot", "?")),
            "UniProt": t.get("uniprot", ""),
            "Target": (t.get("target_pref_name", "") or "")[:50],
            "Affinity (nM)": f"{t.get('standard_value_nm', 0):.2f}",
            "Type": t.get("standard_type", "?"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_nearest_drugs(nn: list[dict]) -> None:
    if not nn:
        return
    st.markdown("#### Chemistry-space nearest neighbors")
    st.caption(
        "ECFP4 Tanimoto similarity to the training catalog. Your compound's "
        "predicted binding profile is a similarity-weighted aggregate of these."
    )
    df = pd.DataFrame([
        {"Drug": d["drug_name"], "Tanimoto": f"{d['tanimoto']:.3f}",
         "# Targets": d.get("n_targets", ""),
         "# Side effects": d.get("n_side_effects", "")}
        for d in nn
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_prediction(p, idx: int) -> None:
    badge_emojis = []
    for ev in p.edge_evidences[:3]:
        for src in ev.sources:
            emoji, _ = _SOURCE_BADGES.get(src.source, ("⚪", "gray"))
            if emoji not in badge_emojis:
                badge_emojis.append(emoji)
    badges = " ".join(badge_emojis)

    promoted = "" if p.scm_top100 else "  🔥 LLM-promoted"
    title = (f"**#{p.rank}  ·  {p.side_effect_name}**  {badges}{promoted}  "
              f"`UMLS {p.side_effect_umls}`")

    with st.expander(title, expanded=(idx < 5)):
        # Mechanism rationale
        st.markdown(f"**Mechanism rationale**: {p.mechanism_rationale}")

        # Top-target attribution
        if p.top_targets:
            attr_df = pd.DataFrame([
                {"Gene": t.gene_symbol, "UniProt": t.uniprot,
                 "Contribution %": t.contribution_pct,
                 "α (blended)": t.alpha,
                 "Affinity (nM)": t.standard_value_nm}
                for t in p.top_targets[:5]
            ])
            if not attr_df.empty:
                col_chart, col_table = st.columns([1, 2])
                with col_chart:
                    st.bar_chart(
                        attr_df.set_index("Gene")["Contribution %"],
                        use_container_width=True,
                    )
                with col_table:
                    st.dataframe(
                        attr_df.style.format({
                            "Contribution %": "{:.0%}", "α (blended)": "{:.3f}",
                            "Affinity (nM)": "{:.2f}",
                        }),
                        use_container_width=True, hide_index=True,
                    )

        # Multi-source evidence per top target
        st.markdown("**Multi-source evidence trace**")
        for ev in p.edge_evidences[:3]:
            st.markdown(f"  - **{ev.gene_symbol}** → {ev.side_effect_name}:")
            for src in ev.sources:
                emoji, color = _SOURCE_BADGES.get(src.source, ("⚪", "gray"))
                st.markdown(f"    {emoji} **{src.source}**: {src.detail}")


def render_predictions(result: ClinicalSafetyResult) -> None:
    if not result.predictions:
        st.error("No predictions returned. Check binding profile and try again.")
        return
    st.markdown(f"### Top {len(result.predictions)} clinical-safety predictions")
    confidence_color = {"high": "green", "medium": "orange",
                         "low": "red"}.get(result.confidence, "gray")
    st.markdown(
        f"Confidence: :{confidence_color}[{result.confidence.upper()}]  ·  "
        f"Drug-blind: {'yes' if result.query_type == 'smiles' or result.is_inferred else 'no'}"
    )
    st.caption(
        "🔥 LLM-promoted = prediction promoted from outside SCM's top-100 by "
        "LLM mechanism reasoning. Multi-source badges: "
        "🔵 SIDER · 🟢 CTD · 🟣 OpenTargets FAERS · 🟠 PharmGKB · 🔴 AOP-Wiki"
    )
    for i, p in enumerate(result.predictions):
        render_prediction(p, i)


def main() -> None:
    st.set_page_config(
        page_title="SCM Off-Target Safety — Clinical-Safety Prediction",
        page_icon="💊", layout="wide",
    )

    st.title("💊 SCM Off-Target Safety")
    st.markdown(
        "**Mechanism-grounded clinical-safety prediction for preclinical "
        "drug candidates.** Drug-blind operation • per-target attribution • "
        "multi-source evidence trace for FDA-grade regulatory support."
    )

    with st.sidebar:
        st.markdown("### Performance benchmarks")
        st.markdown(
            "Validated on n=51 clinical-safety benchmark (Sprint 5, "
            "pre-registered):\n\n"
            "- **Top-3 hit rate: 47%** vs LLM-with-name 31%, LLM-drug-blind 27%\n"
            "- **Top-10 hit rate: 57%** vs LLM-drug-blind 39% (McNemar p=0.006)\n"
            "- Beats memorization-capable LLM **despite operating drug-blind**\n"
            "- Architecture: SCM-Blended (5 curated causal databases) + "
            "LLM mechanism reasoning"
        )
        st.markdown("---")
        st.markdown("### Canonical safety liabilities caught at rank 1-3")
        st.markdown(
            "- All hERG → QT prolongation drugs (11 of 11)\n"
            "- All statins → rhabdomyolysis (4 of 4)\n"
            "- All VEGFR-TKI → hypertension/cardiotox (3 of 3)\n"
            "- Cerivastatin SLCO1B1 → rhabdo (PharmGKB Level 1A)\n"
            "- Anthracycline cardiotoxicity, TCA arrhythmias\n"
            "- COX-2 selective → MI (rofecoxib class)"
        )
        st.markdown("---")
        st.markdown("### Causal evidence sources")
        st.markdown(
            "- 🔵 **SIDER** — drug-AE FDA label co-occurrence\n"
            "- 🟢 **CTD** — chemical-gene-disease curated mechanism (1,622 edges)\n"
            "- 🟣 **OpenTargets FAERS** — statistically-significant target-AE "
            "  associations (LLR test; 3,837 edges)\n"
            "- 🟠 **PharmGKB** — clinical annotations evidence levels 1A-4 (248 edges)\n"
            "- 🔴 **AOP-Wiki** — formalized adverse outcome pathways (14 edges)"
        )

    tab_smiles, tab_name, tab_profile = st.tabs([
        "🧬 Novel compound (SMILES)",
        "🏷️ Known drug (validation)",
        "🎯 Panel-screen binding profile",
    ])

    engine = get_engine()
    result: ClinicalSafetyResult | None = None

    with tab_smiles:
        st.markdown(
            "**The production use case**: paste a SMILES for a preclinical "
            "compound. Binding profile inferred via Tanimoto NN against the "
            "training catalog; Hybrid SCM+LLM predicts top-20 clinical-safety "
            "liabilities with mechanism rationale + multi-source evidence."
        )
        smiles = st.text_area(
            "SMILES", value="CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
            height=80, key="v2_smiles",
        )
        col1, col2 = st.columns(2)
        with col1:
            k_nbr = st.slider("Number of structural neighbors", 1, 20, 5)
        with col2:
            min_tan = st.slider("Minimum Tanimoto", 0.10, 0.90, 0.30, 0.05)
        if st.button("Predict clinical-safety liabilities", key="smiles_btn",
                       type="primary", use_container_width=True):
            with st.spinner("SCM scoring → Hybrid LLM re-ranking → multi-source evidence trace... (~20s)"):
                try:
                    result = engine.predict_clinical_safety(
                        smiles, query_type="smiles",
                        k_neighbors=k_nbr, min_tanimoto=min_tan,
                    )
                except ValueError as e:
                    st.error(str(e))

    with tab_name:
        st.markdown(
            "Validate the system on a known drug from the SIDER ∩ ChEMBL catalog. "
            "Try: `atorvastatin`, `rofecoxib`, `terfenadine`, `ibuprofen`, "
            "`imatinib`, `haloperidol`."
        )
        drug_name = st.text_input("Drug name", value="atorvastatin", key="v2_name")
        if st.button("Predict clinical-safety liabilities", key="name_btn",
                       type="primary", use_container_width=True):
            with st.spinner("Looking up + Hybrid prediction (~20s)..."):
                result = engine.predict_clinical_safety(drug_name, query_type="drug_name")

    with tab_profile:
        st.markdown(
            "Paste a binding profile from a safety panel screen "
            "(SafetyScreen44, KinomeScan, etc.). Format: one target per line: "
            "`UniProt,gene_symbol,Ki_nM`"
        )
        default = ("P04035,HMGCR,0.5\n"
                   "Q9Y6L6,SLCO1B1,800\n"
                   "Q9UNQ0,ABCG2,5000\n"
                   "P08684,CYP3A4,2000")
        profile_text = st.text_area("Binding profile", value=default,
                                       height=140, key="v2_profile")
        if st.button("Predict clinical-safety liabilities", key="profile_btn",
                       type="primary", use_container_width=True):
            bp = []
            import re
            for line in profile_text.strip().splitlines():
                parts = [p.strip() for p in re.split(r"[,\t]", line)]
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
                        st.warning(f"Skipping malformed line: {line}")
            if bp:
                with st.spinner("Hybrid prediction (~20s)..."):
                    result = engine.predict_clinical_safety(
                        "<binding profile>", query_type="binding_profile",
                        binding_profile=bp,
                    )

    if result is not None:
        st.markdown("---")
        st.info(result.note)
        if result.is_inferred and result.nearest_drugs:
            render_nearest_drugs(result.nearest_drugs)
        render_binding_profile(result.binding_profile, result.is_inferred)
        st.markdown("---")
        render_predictions(result)


if __name__ == "__main__":
    main()
