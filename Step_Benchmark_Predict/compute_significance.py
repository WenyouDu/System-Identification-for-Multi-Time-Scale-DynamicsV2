# -*- coding: utf-8 -*-
"""
Significance tests + improvement percentages for the ablation comparisons
================================
- Paired t-test (ttest_rel) + Wilcoxon signed-rank test: ELSTM (=spectral_fb1, the main model) vs each ablation config
- Improvement %: relative to the simplest baseline MSE+LSTM(b=0); lower is better for errors, higher for R2
- Data: results/multi_seed_results_nolinear.json (main model, 5 seeds)
       results/ablation_lstm_results_nolinear.json (3 ablation configs, 5 seeds)
- Output: results/significance_results.json + console table

Usage: python compute_significance.py
"""
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

METRICS = [
    ("test_r2", "R2", "higher"),
    ("test_rmse_avg", "RMSE", "lower"),
    ("test_mae_avg", "MAE", "lower"),
    ("test_nrmse_avg", "NRMSE", "lower"),
]

CONFIGS = {
    "mse_fb0": "MSE+LSTM (b=0)",
    "mse_fb1": "MSE+LSTM (b=1)",
    "spectral_fb0": "Spectral+LSTM (b=0)",
    "spectral_fb1": "Spectral+LSTM (b=1) = Ours",
}


def load_5seed_metrics():
    """Returns {tag: {metric_key: np.array(5,)}}"""
    main = json.load(open(os.path.join("results", "multi_seed_results_nolinear.json"), encoding="utf-8"))
    abl = json.load(open(os.path.join("results", "ablation_lstm_results_nolinear.json"), encoding="utf-8"))
    out = {}
    # ours = spectral_fb1 (ELSTM)
    out["spectral_fb1"] = {k: np.array([r[k] for r in main["results"]], dtype=float)
                           for k, _, _ in METRICS}
    for tag in ["mse_fb0", "mse_fb1", "spectral_fb0"]:
        out[tag] = {k: np.array([r[k] for r in abl["results"][tag]], dtype=float)
                    for k, _, _ in METRICS}
    return out


def main():
    from scipy import stats
    data = load_5seed_metrics()
    ours_tag = "spectral_fb1"
    base_tag = "mse_fb0"

    print("================ significance tests (ELSTM vs ablations, n=5 seeds) ================")
    print("Paired t-test + Wilcoxon signed-rank test, p<0.05 marked significant (*)")
    print("")

    sig_results = {}
    for tag in ["mse_fb0", "mse_fb1", "spectral_fb0"]:
        print(f"--- {CONFIGS[tag]} vs {CONFIGS[ours_tag]} ---")
        row = {}
        for mkey, mname, direction in METRICS:
            a = data[tag][mkey]
            b = data[ours_tag][mkey]
            # paired t-test
            t_stat, p_t = stats.ttest_rel(a, b)
            # Wilcoxon signed-rank (n=5; nan if all diffs are 0)
            if np.all(a == b):
                p_w = float("nan")
            else:
                try:
                    _, p_w = stats.wilcoxon(a, b)
                except Exception:
                    p_w = float("nan")

            ma, mb = a.mean(), b.mean()
            # improvement % vs the mse_fb0 baseline
            base = data[base_tag][mkey].mean()
            if direction == "higher":  # R2: higher is better
                improve = (mb - base) / abs(base) * 100.0 if abs(base) > 1e-12 else float("nan")
            else:  # errors: lower is better
                improve = (base - mb) / base * 100.0 if abs(base) > 1e-12 else float("nan")
            star = "*" if p_t < 0.05 else ""
            print(f"  {mname:6s}: base={ma:.4f} ours={mb:.4f} | t-test p={p_t:.4f}{star} "
                  f"wilcoxon p={p_w if np.isnan(p_w) else round(p_w,4)} | "
                  f"improve% (vs {CONFIGS[base_tag][:12]}): {improve:+.1f}%")
            row[mkey] = {
                "ours_mean": float(mb), "base_mean": float(ma),
                "ttest_p": float(p_t), "wilcoxon_p": float(p_w),
                "improve_pct_vs_mse_fb0": float(improve),
                "significant_05": bool(p_t < 0.05),
            }
        sig_results[tag] = row
        print()

    # summary: mean +/- std per config
    print("================ per-config mean +/- std (test) ================")
    for tag in ["mse_fb0", "mse_fb1", "spectral_fb0", "spectral_fb1"]:
        parts = [f"{CONFIGS[tag]}: "]
        for mkey, mname, _ in METRICS:
            v = data[tag][mkey]
            parts.append(f"{mname}={v.mean():.4f}+/-{v.std(ddof=1):.4f}")
        print("  " + " | ".join(parts))

    out_path = os.path.join("results", "significance_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sig_results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
