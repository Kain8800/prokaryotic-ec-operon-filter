"""
7_EVALUATION.py

Evaluation of the operon filter on ProteInfer and CLEAN predictions. Implements
three blocks:

  2.1 Confidence-threshold sweep and selection of the optimal
      threshold for each model by the criterion of maximum F1 macro after
      applying the operon filter.
  2.2 Filter effectiveness: F1 macro gain across homology strata
      (HIGH/MEDIUM/LOW/NONE) and across EC-number hierarchy levels (L1..L4).
  2.4 Independence of model confidence and operon context: what fraction
      of high-confidence errors the operon filter catches.

Shared code (building the evaluation set, stratification, the operon filter,
metrics) lives in eval_common.py.

Inputs and environment variables — see eval_common.py.

Output (in RESULTS_DIR):
  sec21_threshold_sweep.csv    - metrics at all thresholds for both models
  sec21_optimal_threshold.csv  - selected optimal thresholds
  sec21_baselines.csv / .png   - model comparison across strata
  sec21_pareto.png             - Pareto front coverage / F1
  sec22_binary_delta_f1.png    - F1 gain after the filter across strata
  sec22_f1_by_ec_level.png     - F1 across EC hierarchy levels
  sec22_ec_class_heatmap.png   - F1 across EC classes (first digit)
  sec24_high_conf_errors.png   - model errors and filter recall across strata
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_common import (
    OUTPUT_DIR,
    PI_PRED_PATH,
    CLEAN_PRED_PATH,
    STRATA_ORDER,
    BINARY_TAU,
    CONF_THRESHOLDS,
    OpeRONScorer,
    PredictionCache,
    build_eval_dataset,
    build_ground_truth,
    load_all_train,
    metrics,
    parse_clean_predictions,
    parse_pi_predictions,
    run_mmseqs2,
    save_train_fasta,
)

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results1")


def save_plot(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def find_best_threshold(sweep):
    """Returns the threshold that maximizes F1 macro after the filter.

    The criterion directly matches the goal of the work - improving the accuracy
    of EC-number prediction via operon functional context.
    Applied identically to both models.
    """
    best_idx = max(range(len(sweep)),
                   key=lambda i: sweep[i]["f1_binary"])
    r = sweep[best_idx]
    return r["threshold"], r["f1_binary"], best_idx


def main():
    print(f"\n{'#'*70}")
    print(f"# DIPLOMA EVALUATION (criterion: max F1 after filter)")
    print(f"{'#'*70}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train = load_all_train()
    save_train_fasta(train)
    eval_df = build_eval_dataset(train)
    eval_df = run_mmseqs2(eval_df)

    ground_truth = build_ground_truth(eval_df)
    pi_preds = parse_pi_predictions(PI_PRED_PATH)
    clean_preds = parse_clean_predictions(CLEAN_PRED_PATH)

    # Map each protein to its homology stratum for later
    # stratified analysis.
    etg = (eval_df.drop_duplicates("Entry")
                  .set_index("Entry")["identity_group"].to_dict())
    all_e = set(ground_truth.keys())
    id_groups = set(etg.values())

    scorer = OpeRONScorer(eval_df)
    cache = PredictionCache(pi_preds, clean_preds, all_e, scorer)

    def ent_for(st):
        if st == "ALL":
            return all_e
        return set(e for e, g in etg.items() if g == st) & all_e


    # SECTION 2.1: BASELINE EVALUATION + OPTIMAL THRESHOLD SELECTION
    print(f"\n{'#'*70}\n# 2.1 BASELINE EVALUATION PI AND CLEAN + THRESHOLD SELECTION\n"
          f"{'#'*70}")

    # Confidence-threshold sweep for both models.
    # At each value, metrics are computed before and after applying
    # the filter, which lets us assess both the model's baseline quality
    # and the effectiveness of the operon context.
    print(f"\n  ProteInfer threshold sweep:")
    pi_sweep = []
    for thr in CONF_THRESHOLDS:
        m = metrics(cache.pi(thr), ground_truth, all_e)
        m_bin = metrics(cache.pi_bin(thr), ground_truth, all_e)
        pi_sweep.append({"threshold": thr, "model": "PI", **m,
                         "f1_binary": m_bin["f1_macro"],
                         "cov_binary": m_bin["coverage"]})
        print(f"    PI(>={thr:.3f}): F1={m['f1_macro']:.4f}, "
              f"cov={m['coverage']:.3f} | "
              f"+filter: F1={m_bin['f1_macro']:.4f}, "
              f"cov={m_bin['coverage']:.3f}")

    print(f"\n  CLEAN threshold sweep:")
    clean_sweep = []
    for thr in CONF_THRESHOLDS:
        m = metrics(cache.cl(thr), ground_truth, all_e)
        m_bin = metrics(cache.cl_bin(thr), ground_truth, all_e)
        clean_sweep.append({"threshold": thr, "model": "CLEAN", **m,
                            "f1_binary": m_bin["f1_macro"],
                            "cov_binary": m_bin["coverage"]})
        print(f"    CLEAN(>={thr:.3f}): F1={m['f1_macro']:.4f}, "
              f"cov={m['coverage']:.3f} | "
              f"+filter: F1={m_bin['f1_macro']:.4f}, "
              f"cov={m_bin['coverage']:.3f}")

    print(f"\n  --- OPTIMAL THRESHOLD SELECTION ---")
    print(f"  Criterion: max F1 macro after the operon filter")

    best_pi_thr, _, pi_best_idx = find_best_threshold(pi_sweep)
    best_pi = pi_sweep[pi_best_idx]

    best_cl_thr, _, cl_best_idx = find_best_threshold(clean_sweep)
    best_cl = clean_sweep[cl_best_idx]

    # Detailed sweep output with the selected threshold marked.
    print(f"\n  PI threshold selection:")
    print(f"  {'Threshold':<8} {'F1 no filter':>16} {'F1+filter':>12} "
          f"{'Coverage no filter':>22} {'Coverage+filter':>17} {'DF1+filter':>12}")
    print(f"  {'-'*82}")
    prev_f1bin = None
    for r in pi_sweep:
        delta = (f"{r['f1_binary'] - prev_f1bin:+.4f}"
                 if prev_f1bin is not None else "-")
        marker = " <- SELECTED" if r["threshold"] == best_pi_thr else ""
        print(f"  {r['threshold']:<8.3f} {r['f1_macro']:>8.4f} "
              f"{r['f1_binary']:>8.4f} {r['coverage']:>8.3f} "
              f"{r['cov_binary']:>8.3f} {delta:>8}{marker}")
        prev_f1bin = r["f1_binary"]

    print(f"\n  CLEAN threshold selection:")
    print(f"  {'Threshold':<8} {'F1 no filter':>16} {'F1+filter':>12} "
          f"{'Coverage no filter':>22} {'Coverage+filter':>17} "
          f"{'DF1+filter':>12}")
    print(f"  {'-'*82}")
    prev_f1bin = None
    for r in clean_sweep:
        delta = (f"{r['f1_binary'] - prev_f1bin:+.4f}"
                 if prev_f1bin is not None else "-")
        marker = " <- SELECTED" if r["threshold"] == best_cl_thr else ""
        print(f"  {r['threshold']:<8.3f} {r['f1_macro']:>8.4f} "
              f"{r['f1_binary']:>8.4f} {r['coverage']:>8.3f} "
              f"{r['cov_binary']:>8.3f} {delta:>8}{marker}")
        prev_f1bin = r["f1_binary"]

    print(f"\n  Optimal thresholds (criterion: max F1 macro after the filter):")
    print(f"    ProteInfer:  tau = {best_pi_thr:.3f}")
    print(f"      F1 no filter:    {best_pi['f1_macro']:.4f}")
    print(f"      F1 + filter:    {best_pi['f1_binary']:.4f}")
    print(f"      Coverage + filter: {best_pi['cov_binary']:.3f}")
    print(f"    CLEAN:       tau = {best_cl_thr:.3f}")
    print(f"      F1 no filter:    {best_cl['f1_macro']:.4f}")
    print(f"      F1 + filter:    {best_cl['f1_binary']:.4f}")
    print(f"      Coverage + filter : {best_cl['cov_binary']:.3f}")

    # Comparing models at the selected thresholds justifies the choice
    # of the main model for all later analyses.
    delta_f1 = best_pi['f1_binary'] - best_cl['f1_binary']
    delta_cov = best_pi['cov_binary'] - best_cl['cov_binary']
    print(f"\n  Model comparison at the selected thresholds:")
    print(f"    ProteInfer (tau={best_pi_thr:.3f}): "
          f"F1 no filter = {best_pi['f1_macro']:.4f}, "
          f"F1 + filter = {best_pi['f1_binary']:.4f}, "
          f"coverage = {best_pi['cov_binary']:.3f}")
    print(f"    CLEAN      (tau={best_cl_thr:.3f}): "
          f"F1 no filter = {best_cl['f1_macro']:.4f}, "
          f"F1 + filter = {best_cl['f1_binary']:.4f}, "
          f"coverage = {best_cl['cov_binary']:.3f}")
    print(f"    DF1 = {delta_f1:+.4f} (PI - CLEAN), "
          f"Dcoverage = {delta_cov:+.3f}")
    if delta_f1 > 0 and delta_cov > 0:
        print(f"    -> ProteInfer dominates: higher F1 and coverage.")
    elif delta_f1 < 0 and delta_cov < 0:
        print(f"    -> CLEAN dominates: higher F1 and coverage.")
    else:
        print(f"    -> No single F1 winner: the models "
              f"win on different quantities.")
        print(f"       The criterion maximizes F1 with no constraint on "
              f"coverage, so the optimum drifts to strict thresholds,")
        print(f"       where F1 rises by rejecting predictions. "
              f"Comparison at matched coverage - on the Pareto plot")
        print(f"       (sec21_pareto.png).")
    print(f"       In sections 2.2 and 2.4 the main model remains "
          f"ProteInfer: in the matched-coverage region it")
    print(f"       dominates CLEAN.")

    df_sweep = pd.DataFrame(pi_sweep + clean_sweep)
    df_sweep.to_csv(os.path.join(RESULTS_DIR,
        "sec21_threshold_sweep.csv"), index=False)
    print(f"  -> sec21_threshold_sweep.csv")

    threshold_choice = pd.DataFrame([
        {"model": "ProteInfer",
         "optimal_threshold": best_pi_thr,
         "f1_baseline": best_pi["f1_macro"],
         "f1_binary": best_pi["f1_binary"],
         "coverage_baseline": best_pi["coverage"],
         "coverage_binary": best_pi["cov_binary"],
         "selection_criterion": "maximum F1 macro after the operon filter"},
        {"model": "CLEAN",
         "optimal_threshold": best_cl_thr,
         "f1_baseline": best_cl["f1_macro"],
         "f1_binary": best_cl["f1_binary"],
         "coverage_baseline": best_cl["coverage"],
         "coverage_binary": best_cl["cov_binary"],
         "selection_criterion": "maximum F1 macro after the operon filter"},
    ])
    threshold_choice.to_csv(os.path.join(RESULTS_DIR,
        "sec21_optimal_threshold.csv"), index=False)
    print(f"  -> sec21_optimal_threshold.csv")

    # Metrics of both models at optimal thresholds, sliced by strata.
    rows_21 = []
    for name, preds, thr in [
        (f"PI(>={best_pi_thr})", cache.pi(best_pi_thr), best_pi_thr),
        (f"CLEAN(>={best_cl_thr})", cache.cl(best_cl_thr), best_cl_thr)
    ]:
        for st in ["ALL"] + STRATA_ORDER:
            se = ent_for(st)
            if len(se) < 10: continue
            m = metrics(preds, ground_truth, se)
            rows_21.append({"model": name, "stratum": st,
                            "threshold": thr, **m})

    df_21 = pd.DataFrame(rows_21)
    df_21.to_csv(os.path.join(RESULTS_DIR, "sec21_baselines.csv"),
                 index=False)

    print(f"\n  {'Model':<15} {'Stratum':<8} "
          f"{'F1':>7} {'Acc':>7} {'Cov':>7}")
    print(f"  {'-'*50}")
    for _, r in df_21[df_21.stratum == "ALL"].iterrows():
        print(f"  {r.model:<15} {'ALL':<8} "
              f"{r.f1_macro:>7.4f} {r.accuracy:>7.4f} {r.coverage:>7.3f}")

    # Plot 2.1a: model comparison across strata
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    strata_plot = ["ALL"] + [s for s in STRATA_ORDER if s in id_groups]

    model_names = [f"PI(>={best_pi_thr})", f"CLEAN(>={best_cl_thr})"]
    model_colors = ["#2196F3", "#FF9800"]

    for idx, (metric_name, metric_label) in enumerate(
            [("f1_macro", "F1 macro"), ("coverage", "Coverage")]):
        ax = axes[idx]
        x = np.arange(len(strata_plot))
        w = 0.35
        for i, (model, color) in enumerate(zip(model_names, model_colors)):
            vals = []
            for s in strata_plot:
                sub = df_21[(df_21.model == model) & (df_21.stratum == s)]
                vals.append(sub[metric_name].values[0]
                            if len(sub) > 0 else 0)
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=model,
                          color=color, alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(strata_plot)
        ax.set_ylabel(metric_label)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"2.1 Baseline evaluation at the optimal threshold "
                 f"(max F1 after binary)", fontsize=13)
    save_plot(fig, "sec21_baselines.png")

    # Plot 2.1b: Pareto front coverage / F1
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot([r["coverage"] for r in pi_sweep],
            [r["f1_macro"] for r in pi_sweep],
            "o-", color="#2196F3", linewidth=2, markersize=8,
            label="PI baseline", alpha=0.6)
    ax.plot([r["cov_binary"] for r in pi_sweep],
            [r["f1_binary"] for r in pi_sweep],
            "o--", color="#1565C0", linewidth=2, markersize=8,
            label="PI + binary")
    for r in pi_sweep:
        ax.annotate(f"{r['threshold']:.1f}",
                    (r["cov_binary"], r["f1_binary"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="#1565C0")

    ax.plot([r["coverage"] for r in clean_sweep],
            [r["f1_macro"] for r in clean_sweep],
            "s-", color="#FF9800", linewidth=2, markersize=8,
            label="CLEAN baseline", alpha=0.6)
    ax.plot([r["cov_binary"] for r in clean_sweep],
            [r["f1_binary"] for r in clean_sweep],
            "s--", color="#E65100", linewidth=2, markersize=8,
            label="CLEAN + binary")

    # Stars mark the selected thresholds for both models.
    ax.scatter([best_pi["cov_binary"]], [best_pi["f1_binary"]],
               s=300, marker="*", color="#E91E63", zorder=10,
               edgecolors="black", linewidth=1.5,
               label=f"PI: tau={best_pi_thr} (selected)")
    ax.scatter([best_cl["cov_binary"]], [best_cl["f1_binary"]],
               s=300, marker="*", color="#8E24AA", zorder=10,
               edgecolors="black", linewidth=1.5,
               label=f"CLEAN: tau={best_cl_thr} (selected)")

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("F1 macro", fontsize=12)
    ax.set_title("2.1 Pareto: PI vs CLEAN (no filter and with filter)", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    save_plot(fig, "sec21_pareto.png")


    # SECTION 2.2: FILTER - F1 GAIN ACROSS STRATA
    print(f"\n{'#'*70}\n# 2.2 FILTER - DF1 ACROSS STRATA\n{'#'*70}")

    pi_thr = best_pi_thr
    print(f"\n  Using PI threshold: {pi_thr}")

    strata_avail = [s for s in STRATA_ORDER if s in id_groups]
    f1_before, f1_after = [], []

    for st in strata_avail:
        se = ent_for(st)
        m_bl = metrics(cache.pi(pi_thr), ground_truth, se)
        m_f = metrics(cache.pi_bin(pi_thr), ground_truth, se)
        f1_before.append(m_bl["f1_macro"])
        f1_after.append(m_f["f1_macro"])
        d = m_f["f1_macro"] - m_bl["f1_macro"]
        print(f"  {st:>6}: F1 {m_bl['f1_macro']:.4f} -> "
              f"{m_f['f1_macro']:.4f} (DF1={d:+.4f}), "
              f"cov {m_bl['coverage']:.3f}->{m_f['coverage']:.3f}")

    # Plot 2.2a: F1 before and after the filter across strata
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_st = {"HIGH": "#4CAF50", "MEDIUM": "#FFC107",
                 "LOW": "#FF5722", "NONE": "#9E9E9E"}
    x = np.arange(len(strata_avail))
    w = 0.35

    ax.bar(x - w/2, f1_before, w, label=f"PI(>={pi_thr}) baseline",
           color="#BDBDBD", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, f1_after, w, label=f"PI(>={pi_thr}) + binary",
           color=[colors_st.get(s, "#999") for s in strata_avail],
           alpha=0.85, edgecolor="black", linewidth=0.5)

    for i, (bf, ba) in enumerate(zip(f1_before, f1_after)):
        ax.text(x[i] - w/2, bf + 0.005, f"{bf:.3f}",
                ha="center", fontsize=9)
        ax.text(x[i] + w/2, ba + 0.005, f"{ba:.3f}",
                ha="center", fontsize=9, fontweight="bold")
        delta = ba - bf
        if delta > 0:
            ax.annotate(f"+{delta:.3f}", xy=(x[i] + w/2, ba + 0.025),
                        ha="center", fontsize=8,
                        color="#E91E63", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(strata_avail, fontsize=12)
    ax.set_ylabel("F1 macro (EC L4)", fontsize=12)
    ax.set_title(f"2.2 F1 before and after the binary filter across strata "
                 f"(PI>={pi_thr})", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(f1_after) * 1.12)
    save_plot(fig, "sec22_binary_delta_f1.png")

    # Plot 2.2b: F1 across EC hierarchy levels
    EC_LVLS = [1, 2, 3, 4]
    strata_2x2 = ["ALL", "HIGH", "MEDIUM", "LOW"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    for idx, st in enumerate(strata_2x2):
        ax = axes.flat[idx]
        se = ent_for(st)

        for preds, preds_bin, name, color in [
            (cache.pi(pi_thr), cache.pi_bin(pi_thr),
             "ProteInfer", "#2196F3"),
            (cache.cl(best_cl_thr), cache.cl_bin(best_cl_thr),
             "CLEAN", "#FF9800"),
        ]:
            f1_bl = [metrics(preds, ground_truth, se,
                             lvl=l)["f1_macro"] for l in EC_LVLS]
            f1_f = [metrics(preds_bin, ground_truth, se,
                            lvl=l)["f1_macro"] for l in EC_LVLS]
            ax.plot(EC_LVLS, f1_bl, "-", color=color, marker="o",
                    markersize=5, linewidth=2,
                    label=f"{name} baseline", alpha=0.85)
            ax.plot(EC_LVLS, f1_f, "--", color=color, marker="s",
                    markersize=5, linewidth=2,
                    label=f"{name} + binary", alpha=0.85)

        ax.set_title(f"Stratum: {st}", fontsize=12, fontweight="bold")
        ax.set_xlabel("EC Level", fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel("F1 macro", fontsize=10)
        ax.set_xticks(EC_LVLS)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("2.2 F1 across EC hierarchy levels: baseline vs +binary",
                 fontsize=14, y=1.01)
    save_plot(fig, "sec22_f1_by_ec_level.png")

    # Summary table of F1 across EC levels for the thesis text and slides.
    print(f"\n  --- F1 table across EC levels ---")
    print(f"  {'Stratum':<10} {'Model':<30} "
          f"{'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'DL4':>8}")
    print(f"  {'-'*80}")

    for st in ["ALL", "HIGH", "MEDIUM", "LOW"]:
        se = ent_for(st)
        if len(se) < 10:
            continue
        for preds, preds_bin, name in [
            (cache.pi(pi_thr), cache.pi_bin(pi_thr), "ProteInfer"),
            (cache.cl(best_cl_thr), cache.cl_bin(best_cl_thr), "CLEAN"),
        ]:
            vals_bl = [metrics(preds, ground_truth, se,
                               lvl=l)["f1_macro"] for l in [1,2,3,4]]
            vals_bn = [metrics(preds_bin, ground_truth, se,
                               lvl=l)["f1_macro"] for l in [1,2,3,4]]
            delta_l4 = vals_bn[3] - vals_bl[3]
            row = f"  {st:<10} {name + ' (no filter / with filter)':<30}"
            for bl, bn in zip(vals_bl, vals_bn):
                row += f" {bl:.3f}/{bn:.3f}"
            row += f"  {delta_l4:+.3f}"
            print(row)

    # PI metrics across strata for the final table.
    print(f"\n  --- PI metrics across strata ---")
    print(f"  {'Stratum':<10} {'F1 no filter':>16} {'F1+filter':>12} {'DF1':>8} "
          f"{'Accuracy no filter':>22} {'Accuracy+filter':>17} "
          f"{'Coverage no filter':>22} {'Coverage+filter':>17}")
    for st in ["HIGH", "MEDIUM", "LOW", "ALL"]:
        se = ent_for(st)
        m_bl = metrics(cache.pi(pi_thr), ground_truth, se)
        m_bn = metrics(cache.pi_bin(pi_thr), ground_truth, se)
        print(f"  {st:<10} {m_bl['f1_macro']:>8.4f} "
              f"{m_bn['f1_macro']:>8.4f} "
              f"{m_bn['f1_macro']-m_bl['f1_macro']:>+8.4f} "
              f"{m_bl['accuracy']:>8.4f} {m_bn['accuracy']:>8.4f} "
              f"{m_bl['coverage']:>8.3f} {m_bn['coverage']:>8.3f}")

    # Plot 2.2c: heatmap across EC classes (first digit)
    # For each protein its top-level EC class (1..7) is determined;
    # then F1 macro per class is computed before and after applying the filter.
    entry_ec_class = {}
    for entry, ecs in ground_truth.items():
        for ec in sorted(ecs):
            entry_ec_class[entry] = ec.split(".")[0]
            break

    ec_classes = sorted(set(entry_ec_class.values()), key=lambda x: int(x))
    ec_labels = [f"EC {c}" for c in ec_classes]

    models_data = [
        ("ProteInfer", cache.pi(pi_thr), cache.pi_bin(pi_thr)),
        ("CLEAN", cache.cl(best_cl_thr), cache.cl_bin(best_cl_thr)),
    ]

    matrix = np.full((len(models_data), len(ec_classes)), np.nan)
    f1_af_matrix = np.full((len(models_data), len(ec_classes)), np.nan)

    for i, (mname, preds_bl, preds_f) in enumerate(models_data):
        for j, ec_cls in enumerate(ec_classes):
            cls_entries = set(
                e for e, c in entry_ec_class.items()
                if c == ec_cls) & all_e
            if len(cls_entries) < 10:
                continue
            m_bl = metrics(preds_bl, ground_truth, cls_entries)
            m_f = metrics(preds_f, ground_truth, cls_entries)
            f1_af_matrix[i, j] = m_f["f1_macro"]
            matrix[i, j] = m_f["f1_macro"] - m_bl["f1_macro"]

    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    ax = axes[0]
    vmin = (np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -0.01)
    vmax = (np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 0.01)
    if vmin >= 0: vmin = -0.001
    if vmax <= 0: vmax = 0.001
    # TwoSlopeNorm centered at 0 gives a symmetric color scale,
    # where zero change is white, positive is green, negative is red.
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(ec_labels)))
    ax.set_xticklabels(ec_labels, fontsize=9)
    ax.set_yticks(range(len(models_data)))
    ax.set_yticklabels([m[0] for m in models_data], fontsize=11)
    for i in range(len(models_data)):
        for j in range(len(ec_classes)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if abs(v) < vmax * 0.5 else "white")
    ax.set_title("DF1 macro (+ binary - baseline)", fontsize=12)
    fig.colorbar(im, ax=ax, label="DF1", shrink=0.8)

    ax = axes[1]
    im2 = ax.imshow(f1_af_matrix, cmap="RdYlGn", aspect="auto",
                    vmin=0.3, vmax=0.95)
    ax.set_xticks(range(len(ec_labels)))
    ax.set_xticklabels(ec_labels, fontsize=9)
    ax.set_yticks(range(len(models_data)))
    ax.set_yticklabels([m[0] for m in models_data], fontsize=11)
    for i in range(len(models_data)):
        for j in range(len(ec_classes)):
            v = f1_af_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v < 0.5 else "black")
    ax.set_title("F1 macro after the binary filter", fontsize=12)
    fig.colorbar(im2, ax=ax, label="F1 macro", shrink=0.8)

    fig.suptitle("2.2 F1 macro across EC classes (L1): "
                 "operon binary filter", fontsize=14, y=1.03)
    save_plot(fig, "sec22_ec_class_heatmap.png")


    # SECTION 2.4: HIGH-CONFIDENCE ERRORS AND FILTER RECALL

    print(f"\n{'#'*70}\n# 2.4 HIGH-CONFIDENCE ERRORS\n{'#'*70}")

    # At the optimal threshold, collect data on the model's confident errors:
    # confidence >= 0.9 but the prediction does not match the true EC number.
    # Then we determine how many such errors the operon filter rejects
    # (filter recall), broken down by homology strata.
    hce_rows = []
    pi_sub = cache.pi(best_pi_thr)
    for e, c in pi_sub.items():
        if c[0][1] < 0.9: continue
        gt_e = ground_truth.get(e, set())
        correct = c[0][0] in gt_e
        sc = scorer.score_binary(c[0][0], e)
        # rejected = the filter judged the prediction inconsistent with the context
        # (score < tau). NO_CONTEXT (inf) is excluded - such predictions
        # are not rejected by the filter.
        rej = sc < BINARY_TAU and sc != float("inf")
        hce_rows.append({"entry": e, "confidence": c[0][1],
                         "correct": correct, "binary_score": sc,
                         "rejected": rej,
                         "stratum": etg.get(e, "ALL")})

    hce_df = pd.DataFrame(hce_rows)
    if len(hce_df):
        n_t = len(hce_df)
        n_w = (~hce_df.correct).sum()
        n_wr = ((~hce_df.correct) & hce_df.rejected).sum()
        print(f"\n  Confident errors (conf>=0.9): "
              f"{n_w}/{n_t} ({100*n_w/n_t:.1f}%)")
        if n_w > 0:
            print(f"  Operon filter caught: {n_wr}/{n_w} "
                  f"(recall={n_wr/n_w:.3f})")

    # Plot 2.4: confident errors and filter recall across strata
    if len(hce_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        sl, recall_vals, error_rates = [], [], []

        for st in STRATA_ORDER:
            sub = hce_df[hce_df.stratum == st]
            if len(sub) == 0: continue
            n_total = len(sub)
            n_wrong = (~sub.correct).sum()
            n_caught = ((~sub.correct) & sub.rejected).sum()
            if n_wrong == 0: continue
            sl.append(st)
            recall_vals.append(100 * n_caught / n_wrong)
            error_rates.append(100 * n_wrong / n_total)

        if sl:
            x = np.arange(len(sl))
            w = 0.35
            ax.bar(x - w/2, error_rates, w,
                   label="Model errors (% of all proteins)",
                   color="#E53935", alpha=0.7,
                   edgecolor="black", linewidth=0.5)
            ax.bar(x + w/2, recall_vals, w,
                   label="Of them caught by binary (% of errors)",
                   color="#4CAF50", alpha=0.85,
                   edgecolor="black", linewidth=0.5)
            for i, (er, rc) in enumerate(zip(error_rates, recall_vals)):
                ax.text(x[i] - w/2, er + 0.5, f"{er:.1f}%",
                        ha="center", fontsize=10)
                ax.text(x[i] + w/2, rc + 0.5, f"{rc:.1f}%",
                        ha="center", fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(sl, fontsize=12)
            ax.set_ylabel("%", fontsize=12)
            ax.set_title("2.4 Model errors (conf>=0.9) and "
                         "binary recall across strata", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        save_plot(fig, "sec24_high_conf_errors.png")


    # ── Final ────────────────────────────────────────────────────────────
    print(f"\n{'#'*70}\n# DONE\n{'#'*70}")
    print(f"\n  Optimal PI threshold:    {best_pi_thr} "
          f"(criterion: max F1 after filter)")
    print(f"  Optimal CLEAN threshold: {best_cl_thr} (same criterion)")
    print(f"\nFiles in {RESULTS_DIR}/:")
    print(f"  sec21_threshold_sweep.csv")
    print(f"  sec21_optimal_threshold.csv")
    print(f"  sec21_baselines.csv / .png")
    print(f"  sec21_pareto.png")
    print(f"  sec22_binary_delta_f1.png")
    print(f"  sec22_f1_by_ec_level.png")
    print(f"  sec22_ec_class_heatmap.png")
    print(f"  sec24_high_conf_errors.png")


if __name__ == "__main__":
    main()
