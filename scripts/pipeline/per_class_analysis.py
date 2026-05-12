"""Sprint L.3: per-AE-class hit-rate analysis with bootstrap CIs.

Classifies each benchmark case's causal AE(s) into one of 10 clinical
classes and reports Hybrid hit@10 per class with bootstrap 95% CI.

Identifies systematic weak classes for future targeted curation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


# AE class lookup by UMLS code (10 clinical classes).
AE_CLASS_RULES: dict[str, list[str]] = {
    "Cardiac": [
        "C0018799", "C0018790", "C0018801", "C0018802", "C0027059",
        "C0151744", "C0027051", "C0151878", "C0040479", "C0003811",
        "C0085612", "C0085610", "C0085620", "C0085675", "C0152096",
        "C0007194", "C0231807", "C0264714", "C0264714", "C0007193",
        "C0878544", "C0344440", "C0344434", "C0265279", "C2939193",
        "C0242698", "C0039231", "C0428977", "C0004238", "C0035410",
        "C0034067", "C0036983", "C0036980", "C0036690",
    ],
    "Hepatic": [
        "C0085605", "C0019158", "C0019163", "C0235378", "C0151766",
        "C0023895", "C0235996", "C0160390", "C0267792", "C0241910",
        "C0080226", "C0151903", "C0013182", "C0008372", "C0022346",
        "C0311468", "C0151905", "C0151904", "C0438717", "C0086565",
    ],
    "Renal": [
        "C0022660", "C0035304", "C0033687", "C0017658", "C0027697",
        "C0040076", "C0035091", "C0035410", "C0151746", "C0022346",
        "C0035304", "C1565489", "C0017677",
    ],
    "Hematologic": [
        "C0027947", "C0040034", "C0002871", "C0746883", "C0026986",
        "C0023467", "C0023449", "C0023530", "C0030312", "C0023524",
        "C0034155", "C0002874", "C0002878", "C0019080", "C0017181",
        "C0041364", "C0001824", "C0024299", "C0024305", "C0079545",
        "C0014335", "C0497156", "C0024312", "C0853986", "C0006826",
    ],
    "Neuro": [
        "C0234016", "C0079737", "C0011304", "C0026769", "C0018378",
        "C0026896", "C0036572", "C0038220", "C0023524", "C0021053",
        "C0234518", "C0014038", "C0029134", "C0042571", "C0040264",
        "C0011053", "C0085655", "C0011633", "C0027849", "C0040822",
        "C0030554", "C0003537", "C0013378", "C0338656", "C0011570",
        "C0011581", "C0497327", "C0438696", "C0085631", "C2363742",
        "C0036875", "C0149931", "C2830004", "C0031117", "C0270922",
        "C0151313",
    ],
    "Immune": [
        "C0029118", "C0086438", "C0004364", "C0010823", "C0041296",
        "C0004026", "C0032305", "C2317799", "C2363741", "C0024299",
        "C0024305", "C0021053", "C0009450", "C0042769", "C0026946",
        "C0019360", "C0006848", "C0700345", "C1609512", "C0009763",
        "C0042164", "C0022568", "C0019163", "C5203670", "C0021400",
        "C0032285", "C0032310", "C0006277", "C0037199", "C0040329",
        "C0041912", "C0700184", "C0027441", "C0035243", "C0036690",
        "C1535939", "C0023524", "C0006840", "C0020517", "C0002792",
        "C0036974", "C0002994", "C0148071", "C0007642", "C0010692",
        "C0019095", "C0042029", "C0343886", "C0239295", "C0343863",
        "C0919659", "C1257843", "C0025289", "C0025291", "C0025290",
        "C0036983",
    ],
    "Endocrine": [
        "C0040128", "C0001623", "C0020538", "C0020649", "C0011854",
        "C0011860", "C0020456", "C0020514", "C0020676", "C0020635",
        "C0596022", "C0011880", "C0001620", "C0853286", "C0020550",
    ],
    "GI": [
        "C0011991", "C0009806", "C0027497", "C0042963", "C0030305",
        "C0157654", "C0017181", "C0024862", "C0030193", "C0021368",
        "C0011168", "C0009319", "C0009324", "C0019158", "C0019163",
        "C0026590", "C0043352",
    ],
    "Skin": [
        "C0015230", "C0011603", "C0014742", "C0014518", "C0038325",
        "C2700346", "C1740659", "C0033581", "C0151654", "C0041834",
        "C0234913", "C0549410", "C0033774", "C0016436", "C0001144",
        "C0002170", "C0027339", "C0042963", "C0014335",
    ],
    "Vascular": [
        "C0042487", "C0151942", "C0034065", "C0149871", "C0151602",
        "C0152114", "C0014236", "C0007222", "C0853692", "C0040053",
        "C0040046", "C0014591", "C0038454",
    ],
}


def classify_ae(umls: str) -> str:
    """Returns the clinical class for a UMLS code, or 'Other' if unclassified."""
    for cls, codes in AE_CLASS_RULES.items():
        if umls in codes:
            return cls
    return "Other"


def classify_case(causal_umls_list: list[str]) -> str:
    """Returns the dominant class across a case's causal AEs.

    If multiple classes, returns the FIRST (in canonical order).
    """
    if not causal_umls_list:
        return "Other"
    counts = defaultdict(int)
    for u in causal_umls_list:
        cls = classify_ae(u)
        if cls != "Other":
            counts[cls] += 1
    if not counts:
        return "Other"
    # Return class with most representations; ties broken by canonical order
    canonical = list(AE_CLASS_RULES.keys())
    return max(counts.items(), key=lambda x: (x[1], -canonical.index(x[0])))[0]


def _bootstrap_ci_proportion(hits: int, n: int, n_boot: int = 1000,
                                seed: int = 42) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    successes = np.array([1] * hits + [0] * (n - hits))
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(successes[idx].mean())
    boot_means.sort()
    return (boot_means[int(0.025 * n_boot)], boot_means[int(0.975 * n_boot)])


def per_class_analysis(records: list[dict], label: str = "MAIN"):
    print("\n" + "=" * 78)
    print(f"Per-AE-class analysis: {label} (n={len(records)})")
    print("=" * 78)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("skipped"):
            continue
        causal = r.get("causal_side_effects_umls") or []
        cls = classify_case(causal)
        by_class[cls].append(r)

    print(f"\n{'Class':<15s} {'n':>6s} {'hit@10':>8s} {'95% CI':>20s} "
          f"{'top-3':>8s} {'hit@20':>8s}")
    print("-" * 78)

    summary = {}
    canonical = list(AE_CLASS_RULES.keys()) + ["Other"]
    for cls in canonical:
        sub = by_class.get(cls, [])
        n_cls = len(sub)
        if n_cls == 0:
            continue
        h10 = sum(1 for r in sub
                   if r.get("hybrid_rank") is not None and r["hybrid_rank"] <= 10)
        h3 = sum(1 for r in sub
                  if r.get("hybrid_rank") is not None and r["hybrid_rank"] <= 3)
        h20 = sum(1 for r in sub
                   if r.get("hybrid_rank") is not None and r["hybrid_rank"] <= 20)
        ci = _bootstrap_ci_proportion(h10, n_cls)
        print(f"{cls:<15s} {n_cls:>6d} {h10/n_cls:>7.0%}  "
              f"[{ci[0]:>5.0%}-{ci[1]:>5.0%}]   "
              f"{h3/n_cls:>7.0%}  {h20/n_cls:>7.0%}")
        summary[cls] = {
            "n": n_cls,
            "hit_at_3": h3, "hit_at_10": h10, "hit_at_20": h20,
            "hit_at_10_rate": h10 / n_cls,
            "hit_at_10_95ci": list(ci),
        }
    return summary


def main():
    """Analyze Sprint K Sonnet results per-class."""
    summary_all = {}
    for fn, label in (
        ("sprint_k_safety_sonnet.json", "MAIN"),
        ("sprint_k_ood_safety_sonnet.json", "OOD"),
    ):
        path = RESULTS / fn
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        summary_all[label] = per_class_analysis(d.get("per_drug", []), label)

    # Save summary
    out_path = RESULTS / "sprint_l_per_class.json"
    with open(out_path, "w") as f:
        json.dump(summary_all, f, indent=2)
    print(f"\n[save] {out_path}")

    # Pre-reg evaluation
    print("\n" + "=" * 78)
    print("Sprint L.3 pre-registered hypotheses:")
    print("=" * 78)
    main_classes = summary_all.get("MAIN", {})
    n_strong = sum(1 for c, v in main_classes.items()
                    if v["hit_at_10_rate"] >= 0.70 and v["n"] >= 5)
    n_below_50 = sum(1 for c, v in main_classes.items()
                      if v["hit_at_10_rate"] < 0.50 and v["n"] >= 5)
    n_below_40 = sum(1 for c, v in main_classes.items()
                      if v["hit_at_10_rate"] < 0.40 and v["n"] >= 5)
    print(f"H7: classes with hit@10 ≥ 70% and n≥5: {n_strong}/10 (target ≥6)")
    print(f"H8: classes with hit@10 < 50% and n≥5: {n_below_50}/10 (target =0)")
    print(f"H8b: classes with hit@10 < 40% and n≥5: {n_below_40}/10 (target =0)")
    if n_strong >= 8 and n_below_40 == 0:
        decision = "STRONG WIN"
    elif n_strong >= 6 and n_below_50 == 0:
        decision = "MODERATE WIN"
    elif n_strong >= 4:
        decision = "NULL"
    else:
        decision = "LOSS"
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()
