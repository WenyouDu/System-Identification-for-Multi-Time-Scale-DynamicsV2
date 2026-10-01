# -*- coding: utf-8 -*-
"""
eval_skip_init.py - re-evaluate trained models with the init window skipped.

Loads already-trained models (own + SOTA), skips the first N steps
(CT/SB/WH = 50, CED = 10), and saves metrics to
results/benchmark_skipinit_{ds}_{model}.json.
"""
# run config
DATASETS = ["SB"]                    # datasets: WH / SB / CT / CED
SEEDS = [9, 66, 108, 88, 52]         # random seeds
MODELS = ["EnhancedLSTM"]
# own + SOTA methods: "EnhancedLSTM", "LSTM", "RNN", "GRU", "TCN", "Transformer", "PatchTST"
SKIP_INIT = {"CT": 50, "SB": 50, "WH": 50, "CED": 10}   # skip the first N steps per dataset
import sys, os, time, json
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_HERE)

from Nonlinea_Benchmark_Predict.Option import ALL_DATASETS
from Nonlinea_Benchmark_Predict.train import _MODEL_MAP, _concat_and_reshape

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

from run_benchmark_sota import build_model, evaluate_full

# window / stride per dataset
SEQ_LEN = {"CED": 32, "CT": 128, "SB": 512, "WH": 512}
EVAL_STRIDE = {"CED": 8, "CT": 64, "SB": 200, "WH": 200}

# per-dataset x per-model config (manually maintained, matching the training config)
MODEL_CFG = {
    "CED": {
        "EnhancedLSTM": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "spectral", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3, "alpha": 0.95},
        "LSTM": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "GRU": {"kind": "own", "hidden_size": 64, "num_layers": 1, "dropout": 0.1, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "RNN": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "TCN": {"kind": "window", "num_channels": [32, 32, 32, 32], "kernel_size": 3, "dropout": 0.2,
                "lr": 0.001, "epochs": 1000, "patience": 300},
        "Transformer": {"kind": "thuml", "d_model": 64, "n_heads": 2, "e_layers": 1, "d_layers": 1,
                        "d_ff": 256, "dropout": 0.2, "lr": 0.0005, "epochs": 2000, "patience": 500},
        "PatchTST": {"kind": "patchtst", "d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 128,
                     "patch_len": 16, "patch_stride": 8, "dropout": 0.1, "lr": 0.1, "epochs": 3000, "patience": 600},
    },
    "CT": {
        "EnhancedLSTM": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "spectral", "lr": 0.001, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "LSTM": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "GRU": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "mse", "lr": 0.001, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "RNN": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "mse", "lr": 0.001, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "TCN": {"kind": "window", "num_channels": [128, 128, 128, 128, 128, 128], "kernel_size": 3,
                "dropout": 0.1, "lr": 0.0001, "epochs": 1000, "patience": 300},
        "Transformer": {"kind": "thuml", "d_model": 128, "n_heads": 4, "e_layers": 2, "d_layers": 1,
                        "d_ff": 256, "dropout": 0.2, "lr": 0.0005, "epochs": 2000, "patience": 500},
        "PatchTST": {"kind": "patchtst", "d_model": 128, "n_heads": 4, "e_layers": 2, "d_ff": 256,
                     "patch_len": 16, "patch_stride": 8, "dropout": 0.2, "lr": 0.0005, "epochs": 3000, "patience": 600},
    },
    "SB": {
        "EnhancedLSTM": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "spectral", "lr": 0.001, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "LSTM": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "GRU": {"kind": "own", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "RNN": {"kind": "own", "hidden_size": 64, "num_layers": 1, "dropout": 0.1, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "TCN": {"kind": "window", "num_channels": [64, 64, 64, 64, 64, 64, 64, 64], "kernel_size": 3,
                "dropout": 0.2, "lr": 0.001, "epochs": 1000, "patience": 300},
        "Transformer": {"kind": "thuml", "d_model": 64, "n_heads": 4, "e_layers": 1, "d_layers": 1,
                        "d_ff": 512, "dropout": 0.2, "lr": 0.0001, "epochs": 1000, "patience": 300},
        "PatchTST": {"kind": "patchtst", "d_model": 64, "n_heads": 4, "e_layers": 1, "d_ff": 128,
                     "patch_len": 16, "patch_stride": 8, "dropout": 0.1, "lr": 0.0005, "epochs": 3000, "patience": 600},
    },
    "WH": {
        "EnhancedLSTM": {"kind": "own", "hidden_size": 64, "num_layers": 1, "dropout": 0.1, "loss": "spectral", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "LSTM": {"kind": "own", "hidden_size": 128, "num_layers": 1, "dropout": 0.1, "loss": "mse", "lr": 0.0005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 0.1},
        "GRU": {"kind": "own", "hidden_size": 128, "num_layers": 1, "dropout": 0.1, "loss": "mse", "lr": 0.005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 3},
        "RNN": {"kind": "own", "hidden_size": 128, "num_layers": 1, "dropout": 0.1, "loss": "mse", "lr": 0.0005, "epochs": 3000, "lambda_reg": 1e-05, "max_norm": 0.1},
        "TCN": {"kind": "window", "num_channels": [64, 64, 64, 64, 64, 64, 64, 64], "kernel_size": 3,
                "dropout": 0.2, "lr": 0.001, "epochs": 1000, "patience": 300},
        "Transformer": {"kind": "thuml", "d_model": 64, "n_heads": 4, "e_layers": 1, "d_layers": 1,
                        "d_ff": 512, "dropout": 0.2, "lr": 0.002, "epochs": 1000, "patience": 300},
        "PatchTST": {"kind": "patchtst", "d_model": 128, "n_heads": 4, "e_layers": 1, "d_ff": 128,
                     "patch_len": 16, "patch_stride": 8, "dropout": 0.1, "lr": 0.0005, "epochs": 1000, "patience": 300},
    },
}

OWN_MODELS = {"EnhancedLSTM", "LSTM", "GRU", "RNN"}
SOTA_MODELS = {"TCN", "PatchTST", "Transformer"}
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
    """Six normalized-domain metrics, same protocol as run_benchmark_multi.calc_metrics."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    if (np.isnan(y_true).any() or np.isnan(y_pred).any()
            or np.isinf(y_true).any() or np.isinf(y_pred).any()):
        return {k: float("nan") for k in METRIC_KEYS}
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


def load_own_model(model_key, ds_name, seed, input_size, output_size, cfg):
    """Load an own-method model with its config from MODEL_CFG."""
    path = os.path.join("weights", f"best_{model_key}_{ds_name}_seed{seed}.pt")
    if not os.path.exists(path):
        return None, f"model file not found: {path}"
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"]
    hidden_size = cfg.get("hidden_size", 64)
    num_layers = cfg.get("num_layers", 1)
    dropout = cfg.get("dropout", 0.1 if num_layers > 1 else 0.0)
    model_cls = _MODEL_MAP[model_key]
    try:
        model = model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        ).to(DEVICE)
        model.load_state_dict(sd)
    except Exception as e:
        return None, f"failed to load model ({model_key} h={hidden_size} l={num_layers}): {e}"
    model.eval()
    return model, f"OK (h={hidden_size}, l={num_layers})"


def predict_own(model, u_norm):
    """Own methods take the whole sequence and return normalized-domain predictions [T, c_out]."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(u_norm)).float().unsqueeze(0).to(DEVICE)
        out = model(x)[0].cpu().numpy()  # [T, c_out]
    return out


def run_one(ds_name, model_key, seed, fs, rmse_scaling):
    """Run one dataset x model x seed and compute metrics after skipping the init window."""
    set_seed(seed)
    data = ALL_DATASETS[ds_name]

    train_u = _concat_and_reshape(data["train_data"], 'u')
    train_y = _concat_and_reshape(data["train_data"], 'y')
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    input_scaler.fit(train_u)
    output_scaler.fit(train_y)
    input_size = train_u.shape[1]
    output_size = train_y.shape[1]

    sota_seq_len = None  # actual seq_len for SOTA (from SEQ_LEN)
    if model_key in OWN_MODELS:
        cfg = MODEL_CFG[ds_name][model_key]
        model, msg = load_own_model(model_key, ds_name, seed, input_size, output_size, cfg)
        if model is None:
            return None, msg
    elif model_key in SOTA_MODELS:
        cfg = dict(MODEL_CFG[ds_name][model_key])
        path = os.path.join("results", "saved_models", f"{ds_name}_{model_key}",
                             f"{ds_name}_{model_key}_seed{seed}.pt")
        if not os.path.exists(path):
            return None, f"model file not found: {path}"
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        sd = ckpt["model_state_dict"]
        seq_len = SEQ_LEN.get(ds_name, cfg.get("seq_len", 256))
        sota_seq_len = seq_len
        stride = EVAL_STRIDE.get(ds_name, max(1, seq_len // 4))
        model = build_model(model_key, cfg, seq_len, input_size, output_size).to(DEVICE)
        model.load_state_dict(sd)  # strict: a config mismatch raises here
        model.eval()
    else:
        return None, f"unsupported model: {model_key}"

    test_inputs = data["test_data"] if isinstance(data["test_data"], list) else [data["test_data"]]
    per_test = []

    for ti, test_obj in enumerate(test_inputs):
        u_raw = _concat_and_reshape(test_obj, 'u')
        y_raw = _concat_and_reshape(test_obj, 'y')
        u_norm = input_scaler.transform(u_raw)
        y_norm = output_scaler.transform(y_raw)

        if model_key in OWN_MODELS:
            pred_norm = predict_own(model, u_norm)
        else:
            cfg = MODEL_CFG[ds_name][model_key]
            seq_len = sota_seq_len or SEQ_LEN.get(ds_name, cfg.get("seq_len", 256))
            stride = EVAL_STRIDE.get(ds_name, max(1, seq_len // 4))
            pred_norm = evaluate_full(model, u_raw, input_scaler, seq_len, cfg, stride)

        T = min(len(pred_norm), len(y_norm))
        skip = min(SKIP_INIT[ds_name], T - 1)
        pred_skip = pred_norm[skip:T]
        y_skip = y_norm[skip:T]

        m = calc_metrics(y_skip, pred_skip, fs=fs)

        pred_raw = output_scaler.inverse_transform(pred_norm)
        y_raw_full = output_scaler.inverse_transform(y_norm)
        pred_raw_skip = pred_raw[skip:T]
        y_raw_skip = y_raw_full[skip:T]
        rmse_raw = float(np.sqrt(np.mean((pred_raw_skip - y_raw_skip) ** 2)))
        scaled_rmse = rmse_raw * rmse_scaling

        per_test.append({
            "test_set": ti,
            "n_total": T,
            "skip": skip,
            "n_evaluated": T - skip,
            **m,
            "rmse_raw": rmse_raw,
            "scaled_rmse": scaled_rmse,
        })

    overall = {}
    for k in METRIC_KEYS + ["scaled_rmse", "rmse_raw"]:
        vals = [t[k] for t in per_test if not np.isnan(t[k])]
        overall[k] = float(np.mean(vals)) if vals else float("nan")

    return {
        "seed": seed,
        "model": model_key,
        "dataset": ds_name,
        "skip_init": SKIP_INIT[ds_name],
        "fs": fs,
        "rmse_scaling": rmse_scaling,
        "per_test": per_test,
        **overall,
    }, "OK"


def main():
    os.makedirs("results", exist_ok=True)
    print(f"Skipping the init window per dataset: {SKIP_INIT}")
    print(f"datasets: {DATASETS}")
    print(f"models: {MODELS}")
    print(f"seeds: {SEEDS}")
    print("=" * 70)

    for ds_name in DATASETS:
        data = ALL_DATASETS[ds_name]
        fs = _get_fs(data)
        rmse_scaling = data["opts"]["rmse_scaling"]
        print(f"\n[dataset {ds_name}] fs={fs:.1f} Hz, rmse_scaling={rmse_scaling}")

        for model_key in MODELS:
            print(f"\n  --- model {model_key} ---")
            per_seed = []
            for seed in SEEDS:
                t0 = time.time()
                result, msg = run_one(ds_name, model_key, seed, fs, rmse_scaling)
                elapsed = time.time() - t0
                if result is None:
                    print(f"    seed {seed}: failed - {msg}")
                    continue
                per_seed.append(result)
                r2 = result["r2"]
                rmse = result["rmse"]
                srmse = result["scaled_rmse"]
                print(f"    seed {seed}: R2={r2:.4f}  RMSE={rmse:.4f}  "
                      f"scaledRMSE={srmse:.4f}  ({elapsed:.1f}s)")
                if len(result["per_test"]) > 1:
                    for pt in result["per_test"]:
                        print(f"      test_set {pt['test_set']}: R2={pt['r2']:.4f}  "
                              f"RMSE={pt['rmse']:.4f}  scaledRMSE={pt['scaled_rmse']:.4f}")

            if not per_seed:
                print(f"    {model_key}: no successful seeds, skipped")
                continue

            # summary mean +/- std
            summary = {"dataset": ds_name, "model": model_key, "skip_init": SKIP_INIT[ds_name],
                       "n_seeds": len(per_seed), "seeds": [r["seed"] for r in per_seed]}
            for k in METRIC_KEYS + ["scaled_rmse", "rmse_raw"]:
                vals = [r[k] for r in per_seed if not np.isnan(r[k])]
                summary[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
                summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            # per-seed details
            summary["per_seed"] = per_seed

            # per-test-set aggregation (datasets with multiple test sets, e.g. CED)
            n_ts = len(per_seed[0]["per_test"])
            per_test_set = []
            for ti in range(n_ts):
                agg = {"test_set": ti}
                for k in METRIC_KEYS + ["scaled_rmse", "rmse_raw"]:
                    vals = [r["per_test"][ti][k] for r in per_seed
                            if ti < len(r["per_test"]) and not np.isnan(r["per_test"][ti][k])]
                    agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
                    agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                per_test_set.append(agg)
            summary["per_test_set"] = per_test_set

            out_path = os.path.join("results", f"benchmark_skipinit_{ds_name}_{model_key}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"\n  => {ds_name} {model_key} ({len(per_seed)} seeds): "
                  f"R2={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}  "
                  f"RMSE={summary['rmse_mean']:.4f}±{summary['rmse_std']:.4f}  "
                  f"scaledRMSE={summary['scaled_rmse_mean']:.4f}±{summary['scaled_rmse_std']:.4f}")
            if len(per_test_set) > 1:
                for agg in per_test_set:
                    print(f"    test_set {agg['test_set']}: R2={agg['r2_mean']:.4f}±{agg['r2_std']:.4f}  "
                          f"RMSE={agg['rmse_mean']:.4f}±{agg['rmse_std']:.4f}  "
                          f"scaledRMSE={agg['scaled_rmse_mean']:.4f}±{agg['scaled_rmse_std']:.4f}")
            print(f"  saved: {out_path}")

    print("\n" + "=" * 70)
    print("All done. Results are in results/benchmark_skipinit_*.json")


if __name__ == "__main__":
    main()
