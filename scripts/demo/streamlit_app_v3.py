"""SCM Off-Target Safety — Clinical Safety Demo (v3, Sprint 6F-I).

UX-redesigned for biopharma BD conversations. Optimized for:
  - 30-second value proposition comprehension
  - One-click example demos
  - Hierarchical results (executive summary → top-3 risks → details)
  - Plain-language framing throughout
  - Progressive loading feedback during 20-30s LLM call

Launch:
  streamlit run scripts/demo/streamlit_app_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

import pandas as pd
import streamlit as st

from scripts.demo.predict_hybrid import (
    ClinicalSafetyEngine, ClinicalSafetyResult, HybridPrediction,
)


# ---------- engine ----------
@st.cache_resource(show_spinner="Loading model (one-time, ~10s)...")
def get_engine() -> ClinicalSafetyEngine:
    return ClinicalSafetyEngine()


# ---------- example compounds for one-click demos ----------
EXAMPLES = {
    "🎗️ Sunitinib (VEGFR-TKI, onc)": {
        "type": "drug_name", "value": "sunitinib", "ta": "Oncology",
        "preview": "→ Expect: hypertension, hand-foot, cardiotoxicity (VEGFR class)",
    },
    "🎗️ Doxorubicin (anthracycline, onc)": {
        "type": "drug_name", "value": "doxorubicin", "ta": "Oncology",
        "preview": "→ Expect: cardiotoxicity, myelosuppression (TOP2A class)",
    },
    "🛡️ Rituximab (anti-CD20, immuno)": {
        "type": "drug_name", "value": "rituximab", "ta": "Immunology",
        "preview": "→ Expect: PML, HBV reactivation, infusion reactions",
    },
    "🛡️ Adalimumab (anti-TNF, immuno)": {
        "type": "drug_name", "value": "adalimumab", "ta": "Immunology",
        "preview": "→ Expect: TB reactivation, lymphoma, opportunistic infections",
    },
}


# ---------- risk level color coding ----------
def risk_level(rank: int, n_evidence_sources: int, in_scm_top100: bool) -> tuple[str, str]:
    """Return (label, color) based on rank + evidence."""
    if rank <= 3 and n_evidence_sources >= 2:
        return "HIGH RISK", "#dc2626"      # red
    if rank <= 3:
        return "HIGH RISK", "#ea580c"      # dark orange
    if rank <= 10:
        return "MODERATE RISK", "#f59e0b"  # amber
    return "POSSIBLE", "#71717a"            # gray


# ---------- source emoji + plain-language names ----------
SOURCE_DISPLAY = {
    "Class-effect prior": ("⭐", "FDA black-box / class-effect prior"),
    "SIDER": ("📋", "FDA drug labels"),
    "CTD": ("📚", "Curated mechanism database"),
    "OpenTargets (FAERS)": ("📊", "FDA adverse-event reports"),
    "PharmGKB": ("🧬", "Pharmacogenomic clinical evidence"),
    "AOP-Wiki": ("🔬", "Regulatory toxicology pathways"),
}


# ---------- hero section ----------
def render_hero() -> None:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
      <h1 style="font-size: 2.5rem; margin: 0;">
        Predict Drug Safety Risks Before Phase 1
      </h1>
      <p style="font-size: 1.1rem; opacity: 0.7; margin: 0.5rem 0 1rem 0;">
        Catch the canonical safety liability at rank 1–3 in <strong>47% of cases</strong>.
        Mechanism-grounded · Works on novel compounds · Multi-source auditable evidence.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Workflow diagram — text color forced dark for readability on pastel backgrounds
    # in any theme (light/dark)
    workflow_card_base = (
        'text-align:center; padding:0.7rem; border-radius:8px; '
        'color:#1f2937;'  # slate-800 — readable on all pastel backgrounds
    )
    col1, col2, col3, col4 = st.columns([1.2, 1, 1.4, 1.2])
    with col1:
        st.markdown(
            f'<div style="{workflow_card_base} background:#fef3c7;">'
            '<div style="font-size:1.5rem;">🧬</div>'
            '<div style="font-size:0.9rem; margin-top:0.3rem; color:#1f2937;">'
            '<strong style="color:#0f172a;">Your compound</strong><br/>'
            'SMILES or<br/>binding profile</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="{workflow_card_base} background:#dbeafe;">'
            '<div style="font-size:1.5rem;">🎯</div>'
            '<div style="font-size:0.9rem; margin-top:0.3rem; color:#1f2937;">'
            '<strong style="color:#0f172a;">Binding targets</strong><br/>'
            'From your data or<br/>chemistry-inferred</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div style="{workflow_card_base} background:#e0e7ff;">'
            '<div style="font-size:1.5rem;">🧠</div>'
            '<div style="font-size:0.9rem; margin-top:0.3rem; color:#1f2937;">'
            '<strong style="color:#0f172a;">Causal AI reasoning</strong><br/>'
            '5 curated databases<br/>+ LLM mechanism</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div style="{workflow_card_base} background:#fce7f3;">'
            '<div style="font-size:1.5rem;">📋</div>'
            '<div style="font-size:0.9rem; margin-top:0.3rem; color:#1f2937;">'
            '<strong style="color:#0f172a;">Top 20 risks</strong><br/>'
            '+ mechanism<br/>+ evidence trace</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;")


# ---------- example buttons ----------
def render_examples() -> dict | None:
    st.markdown("##### Try one of these examples (one click):")
    cols = st.columns(len(EXAMPLES))
    selected = None
    for col, (label, ex) in zip(cols, EXAMPLES.items()):
        with col:
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                selected = ex
            st.caption(ex["preview"])
    return selected


# ---------- input panel ----------
def render_input_panel(preset: dict | None) -> tuple[str, str, dict] | None:
    st.markdown("---")
    st.markdown("### Enter a compound to analyze")

    # Therapeutic area selector — biases LLM reasoning toward area-specific
    # safety liabilities (e.g., onc → cardiotoxicity/ILD/irAEs; immuno →
    # infections/MACE/demyelination)
    ta_options = [
        "Oncology",
        "Immunology",
        "Cardiovascular & metabolic",
        "CNS & psychiatry",
        "Other / general (no bias)",
    ]
    default_ta = preset.get("ta", "Oncology") if preset else "Oncology"
    ta_default_idx = ta_options.index(default_ta) if default_ta in ta_options else 0
    ta_choice = st.selectbox(
        "Therapeutic area (biases AI reasoning toward area-specific liabilities)",
        options=ta_options, index=ta_default_idx,
        help="Onc → cardiotox/ILD/irAEs. Immuno → infections/MACE/demyelination. "
             "Other → no bias.",
    )
    ta_value = "" if ta_choice.startswith("Other") else ta_choice

    mode_options = {
        "🧬 Novel compound (SMILES)  — most realistic preclinical use case": "smiles",
        "🏷️ Known drug (validate against literature)": "drug_name",
        "🎯 Custom binding profile (panel-screen results)": "binding_profile",
    }
    default_idx = 0
    if preset:
        if preset["type"] == "drug_name":
            default_idx = 1
        elif preset["type"] == "binding_profile":
            default_idx = 2
    mode_choice = st.selectbox(
        "What kind of input?", options=list(mode_options.keys()), index=default_idx,
    )
    mode = mode_options[mode_choice]

    input_value: str = ""
    options: dict = {}
    if mode == "smiles":
        st.caption(
            "Paste a SMILES string. We infer the binding profile via "
            "chemistry similarity to known drugs — useful when you have a "
            "new structure but no panel-screen data yet."
        )
        default = preset["value"] if preset and preset["type"] == "smiles" else \
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
        input_value = st.text_area("SMILES", value=default, height=70)
        c1, c2 = st.columns(2)
        with c1:
            k = st.slider(
                "Number of structural neighbors to use", 1, 20, 5,
                help="More neighbors = broader signal but noisier. 5 is the default.",
            )
        with c2:
            tan = st.slider(
                "Minimum chemistry similarity", 0.10, 0.90, 0.30, 0.05,
                help="Tanimoto threshold. Lower = more inclusive.",
            )
        options = {"k_neighbors": k, "min_tanimoto": tan}

    elif mode == "drug_name":
        st.caption(
            "Validation mode — look up a drug in the catalog to see how the "
            "system reproduces known safety profiles. Useful for confirming "
            "the tool catches canonical class effects you already know about."
        )
        default = preset["value"] if preset and preset["type"] == "drug_name" else \
            "atorvastatin"
        input_value = st.text_input("Drug name", value=default)

    else:  # binding_profile
        st.caption(
            "Highest-fidelity mode — paste your measured binding profile from "
            "a safety panel (SafetyScreen44, KinomeScan, etc.). One target per "
            "line: `UniProt,gene_symbol,Ki_nM`."
        )
        default = ("P04035,HMGCR,0.5\n"
                   "Q9Y6L6,SLCO1B1,800\n"
                   "Q9UNQ0,ABCG2,5000\n"
                   "P08684,CYP3A4,2000")
        if preset and preset["type"] == "binding_profile":
            default = preset["value"]
        input_value = st.text_area("Binding profile", value=default, height=140)

    submit = st.button(
        "🔬 Predict safety risks",
        type="primary", use_container_width=True,
        disabled=not input_value.strip(),
    )
    if submit:
        options["therapeutic_area"] = ta_value
        return mode, input_value, options
    return None


# ---------- progressive loading ----------
def run_with_progress(engine: ClinicalSafetyEngine, mode: str, value: str,
                       options: dict) -> ClinicalSafetyResult:
    ta = options.pop("therapeutic_area", "")
    progress = st.progress(0, text="Preparing...")
    if mode == "smiles":
        progress.progress(15, text="🧬 Computing chemical fingerprint (ECFP4)...")
    progress.progress(30, text="🔍 Finding structurally similar drugs in catalog...")
    if mode == "binding_profile":
        import re
        bp = []
        for line in value.strip().splitlines():
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
                    pass
        progress.progress(40, text="📊 Running causal model on 6-source evidence substrate...")
        progress.progress(60, text=(f"🤖 LLM mechanism reasoning ({ta or 'general'} bias) ~20s..."))
        result = engine.predict_clinical_safety(
            "<binding profile>", query_type="binding_profile",
            binding_profile=bp, therapeutic_area=ta,
        )
    else:
        progress.progress(40, text="📊 Running causal model on 6-source evidence substrate...")
        progress.progress(60, text=(f"🤖 LLM mechanism reasoning ({ta or 'general'} bias) ~20s..."))
        result = engine.predict_clinical_safety(
            value, query_type=mode, therapeutic_area=ta, **options,
        )
    progress.progress(90, text="📋 Compiling multi-source evidence trace...")
    progress.progress(100, text="Done.")
    progress.empty()
    return result


# ---------- per-prediction card ----------
def render_prediction_card(p: HybridPrediction, primary: bool = False) -> None:
    n_sources_distinct = len({s.source for ev in p.edge_evidences for s in ev.sources})
    label, color = risk_level(p.rank, n_sources_distinct, p.scm_top100)

    border = "2px solid" if primary else "1px solid"
    bg = "#ffffff"
    risk_size = "1.4rem" if primary else "1.05rem"

    promoted = ('<span style="background:#fbbf24; color:#78350f; padding:2px 6px; '
                  'border-radius:4px; font-size:0.7rem; margin-left:6px;">'
                  '⚡ AI-detected (outside DB)</span>') if not p.scm_top100 else ''

    targets_chips = "".join(
        f'<span style="display:inline-block; background:#e0e7ff; color:#3730a3; '
        f'padding:2px 8px; border-radius:12px; margin:2px; font-size:0.85rem;">'
        f'{t.gene_symbol} <strong>{t.contribution_pct:.0%}</strong></span>'
        for t in p.top_targets[:5] if t.contribution_pct > 0.02
    )

    # Evidence checkmarks
    evidence_html = ""
    sources_seen = set()
    for ev in p.edge_evidences[:3]:
        for src in ev.sources:
            if src.source in sources_seen:
                continue
            sources_seen.add(src.source)
            emoji, friendly = SOURCE_DISPLAY.get(src.source, ("•", src.source))
            evidence_html += (
                f'<div style="margin:3px 0; color:#374151;">'
                f'<span style="color:#16a34a;">✓</span> {emoji} '
                f'<strong style="color:#1f2937;">{friendly}</strong></div>'
            )

    # All text colors explicitly set so cards are readable in both light + dark themes
    st.markdown(f"""
    <div style="border:{border} #d4d4d4; border-radius:10px; padding:1rem;
                margin:0.7rem 0; background:{bg}; color:#1f2937;">
      <div style="display:flex; align-items:flex-start; gap:1rem;">
        <div style="flex:0 0 60px; text-align:center;">
          <div style="font-size:1.5rem; color:#525252;"><strong>#{p.rank}</strong></div>
          <div style="background:{color}; color:#ffffff; padding:2px 8px;
                      border-radius:4px; font-size:0.7rem; font-weight:bold;
                      margin-top:4px;">{label}</div>
        </div>
        <div style="flex:1;">
          <div style="font-size:{risk_size}; font-weight:600; margin-bottom:0.3rem;
                      color:#0f172a;">
            {p.side_effect_name}{promoted}
          </div>
          <div style="color:#525252; font-size:0.95rem; margin-bottom:0.6rem;">
            {p.mechanism_rationale}
          </div>
          <div style="margin-bottom:0.5rem;">
            <span style="font-size:0.8rem; color:#525252; margin-right:6px;">
              Driven by:</span>{targets_chips}
          </div>
          <div style="font-size:0.85rem; color:#404040;">
            <strong style="color:#1f2937;">Evidence supporting this prediction:</strong>
            {evidence_html}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if primary:
        with st.expander("📂 Show full evidence detail + target attribution"):
            if p.top_targets:
                attr_df = pd.DataFrame([
                    {"Target gene": t.gene_symbol,
                     "Contribution %": f"{t.contribution_pct:.0%}",
                     "Affinity (nM)": f"{t.standard_value_nm:.1f}"}
                    for t in p.top_targets[:5] if t.contribution_pct > 0.02
                ])
                if not attr_df.empty:
                    st.markdown("**Target attribution breakdown:**")
                    st.dataframe(attr_df, use_container_width=True, hide_index=True)
            st.markdown("**Per-source evidence detail:**")
            for ev in p.edge_evidences[:3]:
                st.markdown(f"**{ev.gene_symbol} → {ev.side_effect_name}:**")
                for src in ev.sources:
                    emoji, friendly = SOURCE_DISPLAY.get(src.source, ("•", src.source))
                    st.markdown(f"- {emoji} **{friendly}**: {src.detail}")


# ---------- results ----------
def render_results(result: ClinicalSafetyResult) -> None:
    if not result.predictions:
        st.error(result.note)
        return

    st.markdown("---")
    n_high = sum(1 for p in result.predictions[:5]
                  if len({s.source for ev in p.edge_evidences for s in ev.sources}) >= 2)
    drug_blind_str = ("Drug-blind (works on novel compounds)"
                        if result.is_inferred or result.query_type == "smiles"
                        else "Reference compound (looked up in catalog)")

    # Executive summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("High-risk predictions", n_high,
                    help="Top-5 predictions with ≥2 evidence sources")
    with c2:
        st.metric("Binding targets analyzed", result.n_targets_used)
    with c3:
        st.metric("Confidence", result.confidence.upper())
    with c4:
        st.metric("Mode", drug_blind_str)

    st.info(f"📋 {result.note}")

    # Top-3 risks (large, prominent)
    st.markdown("### 🚨 Top 3 safety risks (highest priority)")
    st.caption(
        "These are the most-likely safety liabilities based on your compound's "
        "binding profile + multi-source causal evidence. Each prediction is "
        "explained by mechanism reasoning + auditable evidence trace."
    )
    for p in result.predictions[:3]:
        render_prediction_card(p, primary=True)

    # Predictions 4-20 (collapsed)
    if len(result.predictions) > 3:
        with st.expander(f"📋 Additional predictions (ranks 4–{len(result.predictions)})", expanded=False):
            for p in result.predictions[3:]:
                render_prediction_card(p, primary=False)

    # Binding profile / nearest neighbors (deep-dive)
    st.markdown("---")
    with st.expander("🧬 View binding profile + chemistry neighbors"):
        st.markdown("##### Binding profile used for prediction")
        if result.is_inferred:
            st.caption("Inferred via Tanimoto chemical similarity to training catalog drugs.")
        else:
            st.caption("Measured affinities from ChEMBL bioactivity database.")
        if result.binding_profile:
            bp_df = pd.DataFrame([
                {"Gene": t.get("gene_symbol", t.get("uniprot", "?")),
                 "UniProt": t.get("uniprot", ""),
                 "Affinity (nM)": f"{t.get('standard_value_nm', 0):.2f}",
                 "Type": t.get("standard_type", "?")}
                for t in result.binding_profile[:30]
            ])
            st.dataframe(bp_df, use_container_width=True, hide_index=True)

        if result.nearest_drugs:
            st.markdown("##### Chemistry-space neighbors (your compound is anchored on these)")
            nn_df = pd.DataFrame([
                {"Drug": d["drug_name"], "Tanimoto": f"{d['tanimoto']:.3f}",
                 "# Targets": d.get("n_targets", "")}
                for d in result.nearest_drugs
            ])
            st.dataframe(nn_df, use_container_width=True, hide_index=True)


# ---------- sidebar / methodology ----------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 💊 SCM Off-Target Safety")
        st.markdown(
            "Mechanism-grounded clinical-safety prediction. "
            "Drug-blind operation. Multi-source auditable evidence."
        )

        st.markdown("---")
        st.markdown("### 🔬 Why this matters")
        st.markdown(
            "- **70%** of preclinical programs fail; tox is the #1 reason\n"
            "- Phase 2/3 safety failures: **\\$500M-\\$5B per drug**\n"
            "- Post-market withdrawals (Vioxx, Cerivastatin): **\\$1-10B liability**\n"
            "- Existing tools: black-box ML or LLM with memorization gaps on novel drugs"
        )

        st.markdown("---")
        st.markdown("### ✓ Validation (pre-registered)")
        st.markdown(
            "**n=51 clinical-safety benchmark:**\n"
            "- Top-3 hit rate: **47%**\n"
            "  - (vs LLM with full memorization: 31%)\n"
            "  - (vs black-box ML chemistry: 0%)\n"
            "- Top-10 hit rate: **57%**\n"
            "- McNemar p=0.006 vs LLM-drug-blind\n"
            "- Wins on novel preclinical compounds where LLMs degrade"
        )

        st.markdown("---")
        st.markdown("### 📚 Causal evidence sources")
        st.markdown(
            "- 📋 **FDA drug labels** (SIDER)\n"
            "- 📚 **Curated mechanism** (CTD, 1,622 edges)\n"
            "- 📊 **FDA adverse-event reports** (OpenTargets FAERS, 3,837 edges)\n"
            "- 🧬 **Pharmacogenomic** (PharmGKB, 248 clinical annotations)\n"
            "- 🔬 **Regulatory toxicology** (AOP-Wiki, 14 formalized pathways)"
        )

        st.markdown("---")
        st.markdown("### 🎯 Use cases")
        st.markdown(
            "- Preclinical safety triage (de-risk before tox studies)\n"
            "- SAR optimization (which off-targets to engineer out)\n"
            "- In-licensing safety due diligence\n"
            "- Regulatory submission preparation (cite evidence sources)"
        )

        st.markdown("---")
        with st.expander("🧠 How it actually works"):
            st.markdown(
                "**Step 1**: Your compound's binding profile (measured or "
                "chemistry-inferred) tells us which proteins it engages.\n\n"
                "**Step 2**: A structural causal model — built from 5 curated "
                "databases — produces 100 candidate safety risks ranked by "
                "evidence strength. Each candidate has per-target attribution.\n\n"
                "**Step 3**: An LLM (Claude Sonnet 4.6) reads the binding "
                "profile + 100 candidates with attribution + mechanism evidence, "
                "and re-ranks them by clinical likelihood using pharmacology "
                "reasoning. The LLM operates **drug-blind** — it doesn't "
                "know which drug it's analyzing, only the binding profile.\n\n"
                "**Step 4**: We trace each top-20 prediction back to the "
                "specific evidence sources supporting it for auditability."
            )


# ---------- main ----------
def main() -> None:
    st.set_page_config(
        page_title="Drug Safety Prediction · Clinical-grade",
        page_icon="💊", layout="wide",
    )
    render_sidebar()
    render_hero()
    preset = render_examples()
    submitted = render_input_panel(preset)

    # When example clicked, advance to running
    if preset is not None and submitted is None:
        st.session_state["pending_run"] = preset
        st.rerun()

    if "pending_run" in st.session_state:
        preset_run = st.session_state.pop("pending_run")
        engine = get_engine()
        opts = {"therapeutic_area": preset_run.get("ta", "")}
        if preset_run["type"] == "smiles":
            opts["k_neighbors"] = 5
            opts["min_tanimoto"] = 0.30
        with st.spinner("Running pipeline..."):
            result = run_with_progress(
                engine, preset_run["type"], preset_run["value"], opts,
            )
        render_results(result)

    elif submitted is not None:
        mode, value, options = submitted
        engine = get_engine()
        result = run_with_progress(engine, mode, value, options)
        render_results(result)

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#737373; font-size:0.85rem; '
        'padding:1rem 0;">'
        'SCM Off-Target Safety · Validated on n=51 pre-registered clinical-safety benchmark · '
        'Sprint 5 results: 47% top-3 hit rate, McNemar p=0.006 vs LLM-drug-blind'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
