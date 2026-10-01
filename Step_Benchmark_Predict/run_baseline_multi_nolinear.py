# -*- coding: utf-8 -*-
"""
Multi-seed scan of the standard baselines (GRU / RNN / Baseline LSTM) on the nonlinear STEP data
================================================================
All three baselines train on the full sequence (whole u [1,8000,6] -> y [1,8000,3], same as the main method),
with standard MSE (the original baseline training); the metric protocol matches the main method
(six metrics: R2 / RMSE / MAE / NRMSE / FreqErr / AvgAbsError(dB);
train metrics from the final model, test metrics from the best model; min-max normalized domain).

Results: results/baseline_multi_results_nolinear_{MODEL}.json (one file per model)
Model copies: results/saved_models/baseline_{GRU,RNN,LSTM}_seed_{s}_nolinear.pt (+_final_nolinear.pt)
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DATA_ROOT = os.path.join(BASE_DIR, "Step_Dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- tunable config (same scale as the main method, fair comparison) ----------------
MODELS = ["RNN", "GRU", "LSTM"]   # three recurrent baselines
SEEDS = [9, 66, 108, 88, 52]             # same 5 seeds as the main method and SOTA
HIDDEN = 64                             # same hidden size as Enhanced LSTM
NUM_LAYERS = 1                           # same number of layers
LR = 0.005
DROPOUT = 0.1
EPOCHS = 800
PATIENCE = 300
LAMBDA_REG = 1e-5          # weight decay, same as train.py
TRAIN_LEN = 8000
# --------------------------------------------------------------------------------------


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


def calc_metrics(y_true, y_pred):
    """Six metrics in the normalized domain"""
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


def build_model(name):
    """Load the baseline model class (BaselineGRU/RNN/LSTM; all named LSTMmodel)"""
    if name == "GRU":
        from BaselineGRU import LSTMmodel as Cls
    elif name == "RNN":
        from BaselineRNN import LSTMmodel as Cls
    else:  # LSTM
        from BaselineLSTM import LSTMmodel as Cls
    return Cls(input_size=6, hidden_size=HIDDEN, output_size=3,
               num_layers=NUM_LAYERS, dropout=DROPOUT)


def train_one(model, data, seed):
    """Full-sequence training (same as the main method). Returns (best_state, final_state, us, ys, history, train_time_s).
    best is by the lowest val loss (early stop); final is the last training step (for train metrics)."""
    from sklearn.preprocessing import MinMaxScaler
    us = MinMaxScaler(); ys = MinMaxScaler()
    u_tr = us.fit_transform(data["train_input"]); y_tr = ys.fit_transform(data["train_output"])
    u_va = us.transform(data["validata_input"]); y_va = ys.transform(data["validata_output"])

    ut = torch.FloatTensor(u_tr[None]).to(DEVICE)
    yt = torch.FloatTensor(y_tr[None]).to(DEVICE)
    uv = torch.FloatTensor(u_va[None]).to(DEVICE)
    yv = torch.FloatTensor(y_va[None]).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=LAMBDA_REG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=50, min_lr=1e-6)
    crit = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    final_state = None
    history = []
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(ut)
        loss = crit(out, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(uv), yv).item()
        scheduler.step(vloss)
        # early stop: val loss must drop 1% below the best to count as improvement
        if vloss < best_val * 0.99:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"    [early stop] @ epoch {ep + 1}, best val {best_val:.6f}")
                break
        final_state = {k: v.clone() for k, v in model.state_dict().items()}
        history.append({"epoch": ep + 1, "train_loss": float(loss.item()),
                        "val_loss": float(vloss)})
        if (ep + 1) % 100 == 0:
            print(f"    epoch {ep + 1}/{EPOCHS} train {loss.item():.6f} val {vloss:.6f}")
    return best_state, final_state, us, ys, history, time.time() - t0


def predict(state, u_raw, us, model):
    """Load the state and run forward; returns normalized-domain predictions [T,3]"""
    model.load_state_dict(state)
    model.eval()
    u_n = us.transform(u_raw)
    with torch.no_grad():
        ut = torch.FloatTensor(u_n[None]).to(DEVICE)
        pred = model(ut)[0].cpu().numpy()
    return pred


def save_models(name, seed, best_state, final_state, us, ys):
    import joblib
    save_dir = os.path.join("results", "saved_models")
    sc_dir = os.path.join(save_dir, "baseline_scalers_nolinear")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sc_dir, exist_ok=True)
    bpath = os.path.join(save_dir, f"baseline_{name}_seed_{seed}_nolinear.pt")
    fpath = os.path.join(save_dir, f"baseline_{name}_seed_{seed}_final_nolinear.pt")
    torch.save({"model_state_dict": best_state, "name": name, "seed": seed}, bpath)
    torch.save({"model_state_dict": final_state, "name": name, "seed": seed}, fpath)
    joblib.dump(us, os.path.join(sc_dir, f"baseline_{name}_seed_{seed}_nolinear_input_scaler.pkl"))
    joblib.dump(ys, os.path.join(sc_dir, f"baseline_{name}_seed_{seed}_nolinear_output_scaler.pkl"))
    return bpath, fpath


def main():
    smoke = "--smoke" in sys.argv
    seeds = [SEEDS[0]] if smoke else SEEDS
    global EPOCHS
    if smoke:
        EPOCHS = 3

    print(f"===== baseline multi-seed scan (nonlinear STEP data): models={MODELS}, seeds={seeds}, "
          f"hidden={HIDDEN}, layers={NUM_LAYERS}, epochs={EPOCHS} =====")
    data = load_data()
    out = {"config": {"models": MODELS, "seeds": seeds, "hidden": HIDDEN,
                      "num_layers": NUM_LAYERS, "lr": LR, "dropout": DROPOUT,
                      "epochs": EPOCHS, "patience": PATIENCE,
                      "weight_decay": LAMBDA_REG,
                      "loss": "MSE", "full_sequence": True, "data": "Nolinear"},
           "results": {}, "summary": {}}

    for name in MODELS:
        out["results"][name] = []
        print(f"\n>>> model {name} (Baseline{name}.py, MSE, full sequence)")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(name).to(DEVICE)
            best_state, final_state, us, ys, hist, tt = train_one(model, data, seed)
            # train metrics use the final model
            tr_pred = predict(final_state, data["train_input"], us, model)
            y_tr_n = ys.transform(data["train_output"])
            # test metrics use the best model
            te_pred = predict(best_state, data["test_input"], us, model)
            y_te_n = ys.transform(data["test_output"])
            tr_r2, tr_rmse, tr_mae, tr_nrmse, tr_freq, tr_abs = calc_metrics(y_tr_n, tr_pred)
            te_r2, te_rmse, te_mae, te_nrmse, te_freq, te_abs = calc_metrics(y_te_n, te_pred)
            bpath, fpath = save_models(name, seed, best_state, final_state, us, ys)
            out["results"][name].append({
                "seed": seed,
                "train_r2": float(tr_r2),
                "train_rmse_avg": float(tr_rmse.mean()),
                "train_rmse": [float(x) for x in tr_rmse],
                "train_mae_avg": float(tr_mae.mean()),
                "train_nrmse_avg": float(tr_nrmse.mean()),
                "train_freq_err_avg": float(tr_freq.mean()),
                "train_avg_abs_db_avg": float(tr_abs.mean()),
                "test_r2": float(te_r2),
                "test_rmse_avg": float(te_rmse.mean()),
                "test_rmse": [float(x) for x in te_rmse],
                "test_mae_avg": float(te_mae.mean()),
                "test_mae": [float(x) for x in te_mae],
                "test_nrmse_avg": float(te_nrmse.mean()),
                "test_nrmse": [float(x) for x in te_nrmse],
                "test_freq_err_avg": float(te_freq.mean()),
                "test_freq_err": [float(x) for x in te_freq],
                "test_avg_abs_db_avg": float(te_abs.mean()),
                "test_avg_abs_db": [float(x) for x in te_abs],
                "model_path": bpath, "final_model_path": fpath,
                "time_s": float(tt),
            })
            print(f"  [seed {seed}] train R2={tr_r2:.4f} | test R2={te_r2:.4f} "
                  f"RMSE={te_rmse.mean():.5f} MAE={te_mae.mean():.5f} | {tt:.0f}s")

        # summary: mean±std (six test metrics + train R2)
        rows = out["results"][name]
        def _ms(key):
            arr = np.array([r[key] for r in rows], dtype=float)
            m = float(arr.mean())
            s = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            return m, s
        out["summary"][name] = {
            "n_seeds": len(rows),
            "train_r2_mean_std": list(_ms("train_r2")),
            "test_r2_mean_std": list(_ms("test_r2")),
            "test_rmse_avg_mean_std": list(_ms("test_rmse_avg")),
            "test_mae_avg_mean_std": list(_ms("test_mae_avg")),
            "test_nrmse_avg_mean_std": list(_ms("test_nrmse_avg")),
            "test_freq_err_avg_mean_std": list(_ms("test_freq_err_avg")),
            "test_avg_abs_db_avg_mean_std": list(_ms("test_avg_abs_db_avg")),
        }

        # one json file per model
        os.makedirs("results", exist_ok=True)
        mo = {"config": {**out["config"], "models": [name]},
              "results": {name: out["results"][name]},
              "summary": {name: out["summary"][name]}}
        op = os.path.join("results", f"baseline_multi_results_nolinear_{name}.json")
        with open(op, "w", encoding="utf-8") as f:
            json.dump(mo, f, ensure_ascii=False, indent=2)
        print(f"saved: {op}")


if __name__ == "__main__":
    main()
