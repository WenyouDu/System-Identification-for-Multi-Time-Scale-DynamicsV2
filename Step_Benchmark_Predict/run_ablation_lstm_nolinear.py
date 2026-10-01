# -*- coding: utf-8 -*-
"""
Multi-seed ablation scan for other LSTM configs
====================================
Three configs x multiple seeds:
  1) MSE + LSTM (forget_bias=0)       -> plain time-domain MSE
  2) MSE + LSTM (forget_bias=1)       -> plain time-domain MSE
  3) spectral loss + LSTM (forget_bias=0) -> SpectralLoss(alpha=0.95) (same loss as the main method, different b)
Main method = spectral loss + LSTM (forget_bias=1, alpha=0.85), finalized in scan_alpha_seeds_nolinear.py.

Same protocol as the main multi-seed scan:
  - train metrics from the final model (train_predictions/labels returned by trainer.main)
  - test metrics from the best model (trainer.test(use_best_model=True))
  - metrics: R2 / RMSE / MAE / NRMSE / FreqErr / AvgAbs(dB), with mean+/-std
  - saves best + final model copies per seed (results/saved_models/ablation_*_nolinear.pt)

Results: results/ablation_lstm_results_nolinear.json
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

# ---------------- tunable (same as the main method) ----------------
SEEDS = [9, 66, 108, 88, 52]     # same seed set as the main table
ALPHA = 0.85                     # SpectralLoss weight (final value for the nonlinear system)
HIDDEN = 256
NUM_LAYERS = 2
LR = 0.005
DROPOUT = 0.2
EPOCHS = 800
PATIENCE = 300
TRAIN_LEN = 8000
# ablation configs: loss in {mse, spectral}, forget_bias in {0.0, 1.0}
CONFIGS = [
    {"tag": "mse_fb0",     "name": "MSE+LSTM (b=0)",          "loss": "mse",      "forget_bias": 0.0},
    {"tag": "mse_fb1",     "name": "MSE+LSTM (b=1)",          "loss": "mse",      "forget_bias": 1.0},
    {"tag": "spectral_fb0", "name": "spectral loss + LSTM (b=0)",     "loss": "spectral", "forget_bias": 0.0},
]
# ---------------------------------------------------------------------------


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


def make_parameter(seed, cfg):
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
            "alpha": ALPHA,
            "early_stop_patience": PATIENCE,
            "seed": seed,
            "forget_bias": cfg["forget_bias"],   # forget gate bias b
        }
    }


def calc_metrics(y_true, y_pred):
    """Same metric set as the main scan: R2 / RMSE / MAE / NRMSE / FreqErr / AvgAbs(dB)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if (np.isnan(y_true).any() or np.isnan(y_pred).any()
            or np.isinf(y_true).any() or np.isinf(y_pred).any()):
        n_c = y_true.shape[1]
        return (float("nan"), np.full(n_c, float("nan")),
                np.full(n_c, float("nan")), np.full(n_c, float("nan")),
                np.full(n_c, float("nan")), np.full(n_c, float("nan")))
    diff = y_true - y_pred
    rmse_per = np.sqrt(np.mean(diff ** 2, axis=0))
    mae_per = np.mean(np.abs(diff), axis=0)
    span = y_true.max(axis=0) - y_true.min(axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    nrmse_per = rmse_per / span
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


def save_models(cfg_tag, seed):
    """Save the best / final models of this run as config+seed-tagged copies to avoid overwrites.
    Returns (best_path, final_path)."""
    import shutil
    save_dir = os.path.join("results", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    # best model (early-stop optimum, used for test metrics)
    src = os.path.join("weights", "best_EnhancedLSTM.pt")
    if os.path.exists(src):
        name = f"ablation_{cfg_tag}_seed_{seed}_nolinear.pt"
        dst = os.path.join(save_dir, name)
        torch.save(torch.load(src, map_location="cpu"), dst)
        best_dst = dst
    else:
        best_dst = None
        print(f"  [warn] {src} not found, skipping best save")
    # final model (last training step, used for train metrics)
    fsrc = os.path.join("weights", "last_EnhancedLSTM.pt")
    if os.path.exists(fsrc):
        fname = f"ablation_{cfg_tag}_seed_{seed}_final_nolinear.pt"
        fdst = os.path.join(save_dir, fname)
        torch.save(torch.load(fsrc, map_location="cpu"), fdst)
        final_dst = fdst
    else:
        final_dst = None
        print(f"  [warn] {fsrc} not found, skipping final save")
    # scaler copy (same train data -> same scaler across configs)
    sc_dir = os.path.join(save_dir, "scalers_nolinear")
    os.makedirs(sc_dir, exist_ok=True)
    for f in ["input_scaler.pkl", "output_scaler.pkl"]:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(sc_dir, f))
    return best_dst, final_dst


def summarize(results):
    """Print mean +/- std following the main scan summary structure"""
    def mstd(vals):
        v = np.array(vals, dtype=float)
        return [float(v.mean()),
                float(v.std(ddof=1) if len(v) > 1 else 0.0)]
    return {
        "n_seeds": len(results),
        "train_r2_mean_std": mstd([r["train_r2"] for r in results]),
        "test_r2_mean_std": mstd([r["test_r2"] for r in results]),
        "test_rmse_avg_mean_std": mstd([r["test_rmse_avg"] for r in results]),
        "test_mae_avg_mean_std": mstd([r["test_mae_avg"] for r in results]),
        "test_nrmse_avg_mean_std": mstd([r["test_nrmse_avg"] for r in results]),
        "test_freq_err_avg_mean_std": mstd([r["test_freq_err_avg"] for r in results]),
        "test_avg_abs_db_avg_mean_std": mstd([r["test_avg_abs_db_avg"] for r in results]),
    }


def main():
    smoke = "--smoke" in sys.argv
    seeds = [0] if smoke else SEEDS
    if smoke:
        global EPOCHS
        EPOCHS = 3

    print(f"===== LSTM ablation multi-seed scan: seeds={seeds}, alpha={ALPHA}, "
          f"hidden={HIDDEN}, epochs={EPOCHS} =====")
    for c in CONFIGS:
        print(f"  - {c['name']} (loss={c['loss']}, forget_bias={c['forget_bias']})")
    data = load_data()
    all_results = {}
    all_summary = {}

    for cfg in CONFIGS:
        tag = cfg["tag"]
        print(f"\n############ config: {cfg['name']} ############")
        results = []
        for seed in seeds:
            t0 = time.time()
            param = make_parameter(seed, cfg)
            print(f"\n---------- {tag} | Seed {seed} ----------")
            trainer = train_LSTM_models(data, param)
            if cfg["loss"] == "mse":
                trainer.criterion = torch.nn.MSELoss()   # switch to plain time-domain MSE
                print("  [loss] criterion switched to MSELoss()")
            train_result = trainer.main()
            test_pred, test_true = trainer.test(data['test_input'], data['test_output'],
                                                use_best_model=True)
            tr_pred, tr_true = (train_result['train_predictions'],
                                train_result['train_labels'])
            te_pred, te_true = test_pred, test_true
            model_path, final_model_path = save_models(tag, seed)

            tr_r2, tr_rmse, tr_mae, tr_nrmse, tr_freq, tr_avg_abs = calc_metrics(tr_true, tr_pred)
            te_r2, te_rmse, te_mae, te_nrmse, te_freq, te_avg_abs = calc_metrics(te_true, te_pred)
            results.append({
                "seed": seed,
                "loss": cfg["loss"],
                "forget_bias": cfg["forget_bias"],
                "train_r2": float(tr_r2),
                "train_rmse": [float(x) for x in tr_rmse],
                "train_rmse_avg": float(tr_rmse.mean()),
                "train_mae_avg": float(tr_mae.mean()),
                "train_nrmse_avg": float(tr_nrmse.mean()),
                "train_freq_err_avg": float(tr_freq.mean()),
                "train_avg_abs_db_avg": float(tr_avg_abs.mean()),
                "test_r2": float(te_r2),
                "test_rmse": [float(x) for x in te_rmse],
                "test_rmse_avg": float(te_rmse.mean()),
                "test_mae_avg": float(te_mae.mean()),
                "test_mae": [float(x) for x in te_mae],
                "test_nrmse_avg": float(te_nrmse.mean()),
                "test_nrmse": [float(x) for x in te_nrmse],
                "test_freq_err_avg": float(te_freq.mean()),
                "test_freq_err": [float(x) for x in te_freq],
                "test_avg_abs_db_avg": float(te_avg_abs.mean()),
                "test_avg_abs_db": [float(x) for x in te_avg_abs],
                "model_path": model_path,
                "final_model_path": final_model_path,
                "time_s": time.time() - t0,
            })
            print(f"[{tag}|Seed {seed}] train R2={tr_r2:.4f} RMSE={tr_rmse.mean():.5f} | "
                  f"test R2={te_r2:.4f} RMSE={te_rmse.mean():.5f} "
                  f"MAE={te_mae.mean():.5f} NRMSE={te_nrmse.mean():.5f} "
                  f"FreqErr={te_freq.mean():.5f} AvgAbs={te_avg_abs.mean():.4f}dB "
                  f"| {time.time()-t0:.0f}s")

        summary = summarize(results)
        all_results[tag] = results
        all_summary[tag] = summary
        print(f"\n---- {cfg['name']} summary (mean +/- std, n={summary['n_seeds']}) ----")
        print(f"  Train R2     : {summary['train_r2_mean_std'][0]:.4f} ± {summary['train_r2_mean_std'][1]:.4f}")
        print(f"  Test  R2     : {summary['test_r2_mean_std'][0]:.4f} ± {summary['test_r2_mean_std'][1]:.4f}")
        print(f"  Test RMSE(avg): {summary['test_rmse_avg_mean_std'][0]:.5f} +/- {summary['test_rmse_avg_mean_std'][1]:.5f}")
        print(f"  Test MAE(avg) : {summary['test_mae_avg_mean_std'][0]:.5f} +/- {summary['test_mae_avg_mean_std'][1]:.5f}")
        print(f"  Test NRMSE   : {summary['test_nrmse_avg_mean_std'][0]:.5f} ± {summary['test_nrmse_avg_mean_std'][1]:.5f}")
        print(f"  Test FreqErr : {summary['test_freq_err_avg_mean_std'][0]:.5f} ± {summary['test_freq_err_avg_mean_std'][1]:.5f}")
        print(f"  Test AvgAbs  : {summary['test_avg_abs_db_avg_mean_std'][0]:.4f} ± {summary['test_avg_abs_db_avg_mean_std'][1]:.4f} (dB)")

    if not smoke:   # smoke only checks the pipeline, never touches the results json
        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", "ablation_lstm_results_nolinear.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "alpha": ALPHA, "hidden": HIDDEN, "epochs": EPOCHS,
                    "seeds": seeds, "configs": CONFIGS,
                },
                "results": all_results, "summary": all_summary,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nsaved: {out_path}")
    else:
        print("\n[smoke] pipeline OK, no result json written.")


if __name__ == "__main__":
    main()
