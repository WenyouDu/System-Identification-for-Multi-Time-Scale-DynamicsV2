# -*- coding: utf-8 -*-
"""
Alpha x multi-seed validation (nonlinear system version)
==========================================
Repeats the alpha sensitivity check on the nonlinear system data
(Step_*_Data_Nolinear.xls, saturation + dead zone); results and models are saved separately.

Results: results/alpha_seeds_results_nolinear.json
Models: results/saved_models/alpha_*_seed_*_nolinear.pt
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # keep scaler/weights in the project dir

from train import train_LSTM_models
from sklearn.metrics import r2_score

DATA_ROOT = os.path.join(BASE_DIR, "Step_Dataset")

# ---------------- alpha sweep ----------------
ALPHA_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]   # candidate alphas
SEEDS = [9, 66, 108, 88, 52]         # five random seeds
FORGET_BIAS = 1.0                 # forget gate bias (default 1)
# --------------------------------------------------

# fixed hyperparams
HIDDEN = 256
NUM_LAYERS = 2
LR = 0.005
DROPOUT = 0.2
EPOCHS = 800
PATIENCE = 300
TRAIN_LEN = 8000


def load_data():
    train_data = pd.read_excel(os.path.join(DATA_ROOT, 'Step_Train_Data_Nolinear.xls')).values
    test_data = pd.read_excel(os.path.join(DATA_ROOT, 'Step_Test_Data_Nolinear.xls')).values
    validata_data = pd.read_excel(os.path.join(DATA_ROOT, 'Step_Validata_Data_Nolinear.xls')).values
    return {
        "train_input": train_data[:TRAIN_LEN, 1:7],
        "train_output": train_data[:TRAIN_LEN, 7:10],
        "test_input": test_data[:TRAIN_LEN, 1:7],
        "test_output": test_data[:TRAIN_LEN, 7:10],
        "validata_input": validata_data[:TRAIN_LEN, 1:7],
        "validata_output": validata_data[:TRAIN_LEN, 7:10],
    }


def make_parameter(alpha, seed):
    return {
        "LSTM": {
            "input_size": 6,
            "output_size": 3,
            "hidden_size": HIDDEN,
            "num_layers": NUM_LAYERS,
            "learning_rate": LR,
            "dropout": DROPOUT,
            "epoch_frequency": EPOCHS,
            "lambda_reg": 1e-5,
            "max_norm": 3,
            "alpha": alpha,
            "early_stop_patience": PATIENCE,
            "seed": seed,
            "forget_bias": FORGET_BIAS,
        }
    }


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    # guard against divergence (NaN/Inf)
    if (np.isnan(y_true).any() or np.isnan(y_pred).any()
            or np.isinf(y_true).any() or np.isinf(y_pred).any()):
        n_c = y_true.shape[1]
        return (float("nan"), np.full(n_c, float("nan")),
                np.full(n_c, float("nan")), np.full(n_c, float("nan")),
                np.full(n_c, float("nan")), np.full(n_c, float("nan")))
    diff = y_true - y_pred
    rmse_per = np.sqrt(np.mean(diff ** 2, axis=0))
    mae_per = np.mean(np.abs(diff), axis=0)
    # NRMSE = RMSE / (y_true range), per channel
    span = y_true.max(axis=0) - y_true.min(axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    nrmse_per = rmse_per / span
    # freq error: complex-spectrum L1 (same as the SpectralLoss term), averaged over freq per channel
    pred_fft = np.fft.rfft(y_pred, axis=0)
    true_fft = np.fft.rfft(y_true, axis=0)
    freq_err = np.mean(np.abs(pred_fft.real - true_fft.real)
                       + np.abs(pred_fft.imag - true_fft.imag), axis=0)
    fs = 1000
    n = y_true.shape[0]
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_true = np.abs(np.fft.rfft(y_true, axis=0))
    fft_pred = np.abs(np.fft.rfft(y_pred, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(fft_pred / np.clip(fft_true, 1e-12, None), 1e-2, 1e2)
        spec_err_db = 20 * np.log10(ratio)
    mask = (freqs >= 1) & (freqs <= 100)
    avg_abs_db = np.nanmean(np.abs(spec_err_db[mask]), axis=0)
    r2 = r2_score(y_true, y_pred)
    return r2, rmse_per, mae_per, nrmse_per, freq_err, avg_abs_db


def save_best_model(alpha, seed):
    src = os.path.join("weights", "best_EnhancedLSTM.pt")
    if not os.path.exists(src):
        print(f"  [warn] {src} not found, skipping model save")
        return None
    import shutil
    save_dir = os.path.join("results", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    name = f"alpha_{alpha:.2f}_seed_{seed}_nolinear.pt"
    dst = os.path.join(save_dir, name)
    torch.save(torch.load(src, map_location="cpu"), dst)
    # scaler copy (same for all seeds of a dataset, keep one)
    sc_dir = os.path.join(save_dir, "scalers_nolinear")
    os.makedirs(sc_dir, exist_ok=True)
    for f in ["input_scaler.pkl", "output_scaler.pkl"]:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(sc_dir, f))
    return dst


def fmt_metric(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "diverged"
    return f"{v:.4f}"


def sanitize(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def run_one(data, alpha, seed):
    param = make_parameter(alpha, seed)
    trainer = train_LSTM_models(data, param)
    train_result = trainer.main()
    test_pred, test_true = trainer.test(data['test_input'], data['test_output'], use_best_model=True)
    tr_r2, tr_rmse, tr_mae, tr_nrmse, tr_freq, tr_avg_abs = calc_metrics(
        train_result['train_labels'], train_result['train_predictions'])
    te_r2, te_rmse, te_mae, te_nrmse, te_freq, te_avg_abs = calc_metrics(test_true, test_pred)
    model_path = save_best_model(alpha, seed)
    return {
        "train_r2": float(tr_r2),
        "train_rmse_avg": float(tr_rmse.mean()),
        "train_mae_avg": float(tr_mae.mean()),
        "train_nrmse_avg": float(tr_nrmse.mean()),
        "train_freq_err_avg": float(tr_freq.mean()),
        "train_avg_abs_db_avg": float(tr_avg_abs.mean()),
        "test_r2": float(te_r2),
        "test_rmse_avg": float(te_rmse.mean()),
        "test_rmse": [float(x) for x in te_rmse],
        "test_mae_avg": float(te_mae.mean()),
        "test_mae": [float(x) for x in te_mae],
        "test_nrmse_avg": float(te_nrmse.mean()),
        "test_nrmse": [float(x) for x in te_nrmse],
        "test_freq_err_avg": float(te_freq.mean()),
        "test_freq_err": [float(x) for x in te_freq],
        "test_avg_abs_db_avg": float(te_avg_abs.mean()),
        "test_avg_abs_db": [float(x) for x in te_avg_abs],
        "model_path": model_path,
    }


def mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return m, s, len(arr)


def intervals_overlap(m1, s1, m2, s2):
    """Whether two mean+/-std intervals overlap"""
    return (m1 + s1) >= (m2 - s2) and (m2 + s2) >= (m1 - s1)


def main():
    smoke = "--smoke" in sys.argv
    global EPOCHS
    alpha_sweep = ALPHA_SWEEP
    seeds = SEEDS
    if smoke:
        EPOCHS = 3
        alpha_sweep = ALPHA_SWEEP[:2]
        seeds = [0]

    print(f"===== Alpha x multi-seed validation (nonlinear system data) =====")
    print(f"candidate alphas: {alpha_sweep}, seeds per alpha: {seeds}, epochs: {EPOCHS}")
    data = load_data()

    results = []
    for a in alpha_sweep:
        for s in seeds:
            t0 = time.time()
            r = run_one(data, alpha=a, seed=s)
            r = {"alpha": a, "seed": s, **r}
            results.append(r)
            print(f"α={a:<4} seed={s} | train R2={fmt_metric(r['train_r2'])}"
                  f" | test R2={fmt_metric(r['test_r2'])} RMSE={fmt_metric(r['test_rmse_avg'])}"
                  f" | {time.time()-t0:.0f}s")

    # ---------- mean±std per alpha ----------
    print("\n================ per-alpha mean +/- std (test metrics) ================")
    print(f"{'alpha':<8}{'n':<4}{'test R2':<20}{'RMSE':<17}{'MAE':<17}{'NRMSE':<17}{'FreqErr':<17}{'AvgAbs':<17}")
    summary = []
    for a in alpha_sweep:
        rows = [r for r in results if r["alpha"] == a]
        r2s = [r["test_r2"] for r in rows]
        rms = [r["test_rmse_avg"] for r in rows]
        maes = [r["test_mae_avg"] for r in rows]
        nrms = [r["test_nrmse_avg"] for r in rows]
        freqs = [r["test_freq_err_avg"] for r in rows]
        abss = [r["test_avg_abs_db_avg"] for r in rows]
        m_r2, s_r2, n = mean_std(r2s)
        m_rm, s_rm, _ = mean_std(rms)
        m_mae, s_mae, _ = mean_std(maes)
        m_nrm, s_nrm, _ = mean_std(nrms)
        m_fq, s_fq, _ = mean_std(freqs)
        m_ab, s_ab, _ = mean_std(abss)
        summary.append({"alpha": a, "n": n, "test_r2_mean": m_r2, "test_r2_std": s_r2,
                        "test_rmse_mean": m_rm, "test_rmse_std": s_rm,
                        "test_mae_mean": m_mae, "test_mae_std": s_mae,
                        "test_nrmse_mean": m_nrm, "test_nrmse_std": s_nrm,
                        "test_freq_err_mean": m_fq, "test_freq_err_std": s_fq,
                        "test_avg_abs_db_mean": m_ab, "test_avg_abs_db_std": s_ab})
        if np.isnan(m_r2):
            print(f"{a:<8.2f}{n:<4}{'diverged':<20}{'diverged':<17}{'diverged':<17}{'diverged':<17}{'diverged':<17}{'diverged':<17}")
        else:
            print(f"{a:<8.2f}{n:<4}{m_r2:.4f} ± {s_r2:.4f} {m_rm:.5f} ± {s_rm:.5f} "
                  f"{m_mae:.5f} ± {s_mae:.5f} {m_nrm:.5f} ± {s_nrm:.5f} "
                  f"{m_fq:.5f} ± {s_fq:.5f} {m_ab:.4f} ± {s_ab:.4f}")

    # ---------- interval overlap check ----------
    print("\n================ pairwise interval-overlap check (test R2) ================")
    valid = [s for s in summary if not np.isnan(s["test_r2_mean"])]
    if len(valid) >= 2:
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                x, y = valid[i], valid[j]
                ov = intervals_overlap(x["test_r2_mean"], x["test_r2_std"],
                                       y["test_r2_mean"], y["test_r2_std"])
                print(f"α={x['alpha']:.2f} vs α={y['alpha']:.2f}: "
                      f"overlap={ov} ({x['test_r2_mean']:.3f}+-{x['test_r2_std']:.3f} "
                      f"vs {y['test_r2_mean']:.3f}+-{y['test_r2_std']:.3f})")

    # ---------- save ----------
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "alpha_seeds_results_nolinear.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitize({
            "config": {"alpha_sweep": alpha_sweep, "seeds": seeds,
                       "epochs": EPOCHS, "forget_bias": FORGET_BIAS, "data": "Nolinear"},
            "results": results,
            "summary": summary,
        }), f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
