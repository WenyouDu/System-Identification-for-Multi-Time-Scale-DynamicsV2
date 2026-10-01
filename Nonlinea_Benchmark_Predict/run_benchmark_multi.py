"""
Multi-seed runs on the public nonlinear benchmarks (CT / SB / CED / WH).
Training and testing both come from train_LSTM_models in
Nonlinea_Benchmark_Predict/train.py: .main() trains, .test() tests.
This script just loops over seeds and aggregates six metrics
(R2 / RMSE / MAE / NRMSE / FreqErr / AvgAbs in the normalized domain,
plus a raw-scale scaled RMSE, i.e. x rmse_scaling, per benchmark convention).
"""
# run config
DATASETS    = ["SB"]   # datasets to run: CT / SB / CED / WH
SEEDS       = [9, 66, 108, 88, 52]                # random seeds (same as the main tables)
MODELS      = ["EnhancedLSTM"]                            # options: EnhancedLSTM / LSTM / GRU / RNN
LOSS        = "spectral"                               # "spectral"=time-freq loss (main) / "mse"
ALPHA       = 0.95                                # alpha for the spectral loss (used when LOSS=spectral)
SAVE_MODEL  = True                                # save the per-seed best/last models
# ============================================================================

import sys, os, time, json, statistics
import numpy as np
import torch
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)

for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_HERE)

from Nonlinea_Benchmark_Predict.Option import ALL_DATASETS, MODEL_PARAMETERS
from Nonlinea_Benchmark_Predict.train import train_LSTM_models

METRIC_KEYS = ["r2", "rmse", "mae", "nrmse", "freq_err", "avg_abs_db"]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def calc_metrics(y_true, y_pred, fs=1000.0):
    """Six normalized-domain metrics, same protocol as the main scan script."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    diff = y_true - y_pred
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    mae = np.mean(np.abs(diff), axis=0)
    span = y_true.max(axis=0) - y_true.min(axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    nrmse = rmse / span
    pred_fft = np.fft.rfft(y_pred, axis=0)
    true_fft = np.fft.rfft(y_true, axis=0)
    freq_err = np.mean(np.abs(pred_fft.real - true_fft.real)
                       + np.abs(pred_fft.imag - true_fft.imag), axis=0)
    n = y_true.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_true = np.abs(np.fft.rfft(y_true, axis=0))
    fft_pred = np.abs(np.fft.rfft(y_pred, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(fft_pred / np.clip(fft_true, 1e-12, None), 1e-2, 1e2)
        spec_err_db = 20 * np.log10(ratio)
    mask = (freqs >= 1) & (freqs <= 100)
    if mask.sum() == 0:
        mask = np.ones_like(freqs, dtype=bool)
    avg_abs_db = np.nanmean(np.abs(spec_err_db[mask]), axis=0)
    r2 = r2_score(y_true, y_pred)
    return {
        "r2": r2,
        "rmse": float(np.mean(rmse)),
        "mae": float(np.mean(mae)),
        "nrmse": float(np.mean(nrmse)),
        "freq_err": float(np.mean(freq_err)),
        "avg_abs_db": float(np.mean(avg_abs_db)),
    }


def _get_fs(data):
    obj = data["train_data"]
    if isinstance(obj, list):
        obj = obj[0] if len(obj) else None
    st = getattr(obj, "sampling_time", None)
    if st is None:
        return 1000.0
    st = float(st)
    return 1.0 / st if st > 0 else 1000.0


def run_one(ds_name, seed, model_key, fs):
    """One (dataset, seed): reuse train_LSTM_models for training + testing.
    Returns a list of per-test-set metric dicts."""
    set_seed(seed)
    data = ALL_DATASETS[ds_name]

    # params: Option.py hyper-params + dataset opts + model/loss/save tag
    params = MODEL_PARAMETERS["LSTM"].copy()
    params.update(data["opts"])
    params["model"] = model_key
    params["loss"] = LOSS
    params["alpha"] = ALPHA
    params["save_tag"] = f"{model_key}_{ds_name}_seed{seed}"
    parameter = {"LSTM": params}

    data_dict = {
        "train_input": data["train_data"], "train_output": data["train_data"],
        "val_input": data["val_data"], "val_output": data["val_data"],
        "test_input": data["test_data"], "test_output": data["test_data"],
    }

    start = time.time()
    trainer = train_LSTM_models(data_dict, parameter, current_dataset_name=ds_name)
    trainer.main()                       # train + save best/last models (with save_tag)
    test_results = trainer.test()        # per test set: best_pred/final_pred/true_labels (raw scale)
    elapsed = time.time() - start

    # normalized-domain metrics: re-normalize raw preds/labels with the training output_scaler
    oscaler = joblib.load("output_scaler.pkl")
    rmse_scaling = data["opts"]["rmse_scaling"]
    per_set = []
    for tr in test_results:
        pred_raw = np.asarray(tr["best_pred"])
        true_raw = np.asarray(tr["true_labels"])
        if pred_raw.ndim == 1:
            pred_raw = pred_raw.reshape(-1, 1)
            true_raw = true_raw.reshape(-1, 1)
        pred_norm = oscaler.transform(pred_raw)
        true_norm = oscaler.transform(true_raw)
        m = calc_metrics(true_norm, pred_norm, fs=fs)
        m["scaled_rmse"] = tr["scaled_rmse_best"]
        per_set.append(m)
    return per_set, elapsed


def main():
    results = {}
    for ds in DATASETS:
        fs = _get_fs(ALL_DATASETS[ds])
        results[ds] = {}
        for model_name in MODELS:
            seed_summary = {k: [] for k in METRIC_KEYS + ["scaled_rmse"]}
            per_test_sets = {}   # test_set_idx -> {metric: [values across seeds]}
            print(f"\n========== dataset {ds} / model {model_name} / loss={LOSS} α={ALPHA} / fs={fs:.3f} ==========")
            for seed in SEEDS:
                per_set, elapsed = run_one(ds, seed, model_name, fs)
                for k in seed_summary:
                    seed_summary[k].append(float(np.mean([s[k] for s in per_set])))
                for ts_idx, s_ts in enumerate(per_set):
                    if ts_idx not in per_test_sets:
                        per_test_sets[ts_idx] = {k: [] for k in METRIC_KEYS + ["scaled_rmse"]}
                    for k in per_test_sets[ts_idx]:
                        per_test_sets[ts_idx][k].append(float(s_ts[k]))
                print(f"    [seed {seed}] scaledRMSE={seed_summary['scaled_rmse'][-1]:.4f}  "
                      f"R2={seed_summary['r2'][-1]:.4f}  MAE={seed_summary['mae'][-1]:.4f}  "
                      f"NRMSE={seed_summary['nrmse'][-1]:.4f}  (n_test={len(per_set)}, time={elapsed:.0f}s)")

            results[ds][model_name] = {"n_seeds": len(SEEDS), "fs": fs}
            for k in seed_summary:
                arr = seed_summary[k]
                results[ds][model_name][k + "_mean_std"] = [
                    statistics.mean(arr),
                    statistics.stdev(arr) if len(arr) > 1 else 0.0,
                ]
            # per test-subset stats (cross-seed mean +/- std)
            results[ds][model_name]["per_test_set"] = {}
            for ts_idx in sorted(per_test_sets):
                pts = results[ds][model_name]["per_test_set"][ts_idx] = {}
                for k, arr in per_test_sets[ts_idx].items():
                    pts[k + "_mean_std"] = [
                        statistics.mean(arr),
                        statistics.stdev(arr) if len(arr) > 1 else 0.0,
                    ]
            s = results[ds][model_name]
            print(f"  => {ds} {model_name}: R2={s['r2_mean_std'][0]:.5f}±{s['r2_mean_std'][1]:.5f} | "
                  f"RMSE={s['rmse_mean_std'][0]:.4f}±{s['rmse_mean_std'][1]:.4f} | "
                  f"MAE={s['mae_mean_std'][0]:.4f}±{s['mae_mean_std'][1]:.4f} | "
                  f"NRMSE={s['nrmse_mean_std'][0]:.4f}±{s['nrmse_mean_std'][1]:.4f} | "
                  f"FreqErr={s['freq_err_mean_std'][0]:.4f}±{s['freq_err_mean_std'][1]:.4f} | "
                  f"AvgAbs={s['avg_abs_db_mean_std'][0]:.4f}±{s['avg_abs_db_mean_std'][1]:.4f} | "
                  f"scaledRMSE={s['scaled_rmse_mean_std'][0]:.4f}±{s['scaled_rmse_mean_std'][1]:.4f}")
            # print each test subset
            for ts_idx in sorted(s["per_test_set"]):
                pts = s["per_test_set"][ts_idx]
                print(f"    test_set {ts_idx}: R2={pts['r2_mean_std'][0]:.5f}±{pts['r2_mean_std'][1]:.5f} | "
                      f"RMSE={pts['rmse_mean_std'][0]:.4f}±{pts['rmse_mean_std'][1]:.4f} | "
                      f"MAE={pts['mae_mean_std'][0]:.4f}±{pts['mae_mean_std'][1]:.4f} | "
                      f"NRMSE={pts['nrmse_mean_std'][0]:.4f}±{pts['nrmse_mean_std'][1]:.4f} | "
                      f"FreqErr={pts['freq_err_mean_std'][0]:.4f}±{pts['freq_err_mean_std'][1]:.4f} | "
                      f"AvgAbs={pts['avg_abs_db_mean_std'][0]:.4f}±{pts['avg_abs_db_mean_std'][1]:.4f} | "
                      f"scaledRMSE={pts['scaled_rmse_mean_std'][0]:.4f}±{pts['scaled_rmse_mean_std'][1]:.4f}")

    # summary
    print("\n================ summary (mean +/- std) ================")
    hdr = f"{'Dataset':<6} {'Model':<14}"
    for k in METRIC_KEYS + ["scaled_rmse"]:
        hdr += f" {k:<18}"
    print(hdr)
    for ds in DATASETS:
        for mn in MODELS:
            s = results[ds][mn]
            row = f"{ds:<6} {mn:<14}"
            for k in METRIC_KEYS + ["scaled_rmse"]:
                if k == "r2":
                    row += f" {s[k + '_mean_std'][0]:.5f}±{s[k + '_mean_std'][1]:.5f}"
                else:
                    row += f" {s[k + '_mean_std'][0]:.4f}±{s[k + '_mean_std'][1]:.4f}"
            print(row)
            # per test subset details
            for ts_idx in sorted(s.get("per_test_set", {})):
                pts = s["per_test_set"][ts_idx]
                detail = f"    {ds} test_set {ts_idx}:"
                for k in METRIC_KEYS + ["scaled_rmse"]:
                    if k == "r2":
                        detail += f" {k}={pts[k+'_mean_std'][0]:.5f}±{pts[k+'_mean_std'][1]:.5f}"
                    else:
                        detail += f" {k}={pts[k+'_mean_std'][0]:.4f}±{pts[k+'_mean_std'][1]:.4f}"
                print(detail)

    # save per dataset+method (one dataset+method per run, no overwrites)
    res_dir = os.path.join(_HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    # full training config (per-dataset effective params after merging dataset opts)
    config = {
        "seeds": SEEDS,
        "loss": LOSS,
        "alpha": ALPHA,
        "parameters": {
            ds: {**MODEL_PARAMETERS["LSTM"], **ALL_DATASETS[ds]["opts"]}
            for ds in DATASETS
        },
    }
    for ds in DATASETS:
        for mn in MODELS:
            out_path = os.path.join(res_dir, f"benchmark_multi_{ds}_{mn}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"dataset": ds, "model": mn, "config": config,
                           "results": results[ds][mn]},
                          f, ensure_ascii=False, indent=2)
            print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
    