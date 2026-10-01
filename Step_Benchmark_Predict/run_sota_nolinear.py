# -*- coding: utf-8 -*-
"""
SOTA comparison script (thuml/Time-Series-Library official implementations)
====================================================
Plug the official TCN / Transformer / PatchTST models into the input-driven same-segment mapping,
and compare them fairly with Enhanced LSTM (our main method).

Protocol (same as Enhanced LSTM):
  input: windows of u [L, 6] (system input)
  output: the same windows of y [L, 3] (system output)
  training: windowed teacher forcing (MSE)
  testing: slide over the whole 8000-step u, predict y per window, stitch into one sequence
         (same protocol as the main method: given all inputs, predict all outputs)

Dependency: Time-Series-Library (thuml) at the repo root, shipped with this repo
      einops / reformer-pytorch (already installed)

Results: results/sota_results_nolinear.json (per-seed metrics + cross-seed mean/std, train & test)
      results/sota_history_nolinear.json (per-epoch train/val loss + lr)
"""
import os
import sys
import json
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# ---------------- model config ----------------
# per-model config; kind picks the train/predict path
#   tcn_full  : our bundled official TCN (locuslab/TCN), full-sequence forward
#   thuml     : thuml Transformer-style encoder-decoder models (windowed)
#   patchtst  : thuml PatchTST (windowed; outputs enc_in channels, projected 6->3)
MODEL_CFG = {
    "TCN": {
        "kind": "tcn_full",
        "num_channels": [128,128,128,128], "kernel_size": 3,
        "dropout": 0.1, "lr": 0.0005, "epochs": 800, "patience": 300,
    },
    "Transformer": {
        "kind": "thuml",
        "seq_len": 1024, "stride": 200,
        "d_model": 128, "n_heads": 4, "e_layers": 2, "d_layers": 1, "d_ff": 256,
        "dropout": 0.2, "lr": 0.0005, "epochs": 800, "patience": 300,
    },
    "PatchTST": {
        "kind": "patchtst",
        "seq_len": 256, "stride": 50,
        "d_model": 128, "n_heads": 4, "e_layers": 2, "d_ff": 256,   
        "patch_len": 16, "patch_stride": 8,
        "dropout": 0.2, "lr": 0.0005, "epochs": 800, "patience": 300,  # more samples, longer training
    },
}
# models to run; names must match the keys in MODEL_CFG
MODELS = ["TCN"]                             #  "Transformer", "PatchTST"
SEEDS = [9, 66, 108, 88, 52]                 # same seed set as the main method
BATCH_SIZE = 16
SMOKE = False                        # True = quick check (2 seeds x 3 epochs)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSLIB = os.path.join(BASE_DIR, "..", "Time-Series-Library")
if TSLIB not in sys.path:
    sys.path.insert(0, TSLIB)

os.chdir(BASE_DIR)
DATA_ROOT = os.path.join(BASE_DIR, "Step_Dataset")
TRAIN_LEN = 8000


def make_config(cfg):
    """Build the config thuml models need (same-segment mapping: seq_len=pred_len=L)."""
    L = cfg["seq_len"]
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=L, pred_len=L, label_len=0,
        enc_in=6, c_out=3, dec_in=3,        # dec_in=3: decoder input is the 3 output channels of y
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
        e_layers=cfg.get("e_layers", 2), d_layers=cfg.get("d_layers", 1),
        d_ff=cfg.get("d_ff", 256), dropout=cfg.get("dropout", 0.2),
        activation="gelu",
        embed="fixed", freq="h", factor=5,
        output_attention=False,
        num_class=1, patch_len=cfg.get("patch_len", 16),
        stride=cfg.get("patch_stride", 8),   # PatchTST patch stride
        moving_avg=25, top_k=5, num_kernels=6,
        distil=cfg.get("distil", True),
    )


def load_data():
    train_data = pd.read_excel(os.path.join(DATA_ROOT, 'Step_Train_Data_Nolinear.xls')).values
    test_data = pd.read_excel(os.path.join(DATA_ROOT, 'Step_Test_Data_Nolinear.xls')).values
    validata_data = pd.read_excel(os.path.join(DATA_ROOT, 'Step_Validata_Data_Nolinear.xls')).values
    return {
        "train_u": train_data[:TRAIN_LEN, 1:7].astype(float),
        "train_y": train_data[:TRAIN_LEN, 7:10].astype(float),
        "test_u": test_data[:TRAIN_LEN, 1:7].astype(float),
        "test_y": test_data[:TRAIN_LEN, 7:10].astype(float),
        "val_u": validata_data[:TRAIN_LEN, 1:7].astype(float),
        "val_y": validata_data[:TRAIN_LEN, 7:10].astype(float),
    }


def make_windows(u, y, seq_len, stride):
    """Build windows: input u[t:t+L] -> output y[t:t+L], same-segment mapping."""
    xs, ys = [], []
    n = len(u)
    for t in range(0, n - seq_len + 1, stride):
        xs.append(u[t:t + seq_len])
        ys.append(y[t:t + seq_len])
    return np.stack(xs), np.stack(ys)


def _load_tslib_model(name):
    """Load TSLIB/models/{name}.py directly via importlib with an absolute path."""
    if TSLIB in sys.path:
        sys.path.remove(TSLIB)
    sys.path.insert(0, TSLIB)
    import importlib.util
    mod_path = os.path.join(TSLIB, "models", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"tslib_model_{name}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(name, cfg):
    if cfg["kind"] == "thuml":
        fname = cfg.get("tslib_file", name)   # model name may differ from the tslib file name
        return _load_tslib_model(fname).Model(make_config(cfg))
    if cfg["kind"] == "patchtst":
        return PatchTSTWrapper(_load_tslib_model("PatchTST").Model(make_config(cfg)), cfg)
    raise ValueError(f"unknown kind: {cfg['kind']}")


class PatchTSTWrapper(nn.Module):
    """Wrap thuml PatchTST: input u [B,L,6] -> output y [B,L,3].
    PatchTST natively outputs enc_in channels (6 input vars); a linear layer maps 6->3.
    (same idea as the Transformer encoder(6)->projection(3))"""

    def __init__(self, inner, cfg):
        super().__init__()
        self.inner = inner
        self.proj = nn.Linear(6, 3)  # channel projection: enc_in(6) -> c_out(3)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, L, 6]
        return self.proj(out)  # [B, L, 3]


def train_tcn(data, cfg, smoke=False):
    """TCN full-sequence mode: whole u [1,T,6] -> whole y [1,T,3], same as Enhanced LSTM.
    Uses the official implementation (locuslab/TCN, Bai et al. 2018): weight_norm + Chomp1d + dilated conv + residual."""
    from OfficialTCN import TCN as TCNModel
    us = MinMaxScaler(); ys = MinMaxScaler()
    u_tr = us.fit_transform(data["train_u"]); y_tr = ys.fit_transform(data["train_y"])
    u_va = us.transform(data["val_u"]); y_va = ys.transform(data["val_y"])

    model = TCNModel(input_size=6, output_size=3,
                     num_channels=cfg["num_channels"], kernel_size=cfg["kernel_size"],
                     dropout=cfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.MSELoss()
    n_epochs = 3 if smoke else cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    ut = torch.FloatTensor(u_tr[None]).to(DEVICE)
    yt = torch.FloatTensor(y_tr[None]).to(DEVICE)
    uv = torch.FloatTensor(u_va[None]).to(DEVICE)
    yv = torch.FloatTensor(y_va[None]).to(DEVICE)

    best_val, best_state, bad = float("inf"), None, 0
    history = []          # per-epoch train/val loss + lr
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        out = model(ut)
        loss = crit(out, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(uv)
            vloss = crit(vout, yv).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"  [TCN] early stop @ epoch {ep + 1}, best val {best_val:.6f}")
                break
        history.append({"epoch": ep + 1, "train_loss": float(loss.item()),
                        "val_loss": float(vloss), "lr": float(sched.get_last_lr()[0])})
        if (ep + 1) % 25 == 0 or smoke:
            print(f"  [TCN] epoch {ep + 1}/{n_epochs} train {loss.item():.6f} val {vloss:.6f} lr={sched.get_last_lr()[0]:.2e}")
        sched.step()
    model.load_state_dict(best_state)
    return model, us, ys, time.time() - t0, history


def tcn_predict_full(model, u_full, us):
    """One full-sequence forward; returns predictions in the normalized domain."""
    model.eval()
    u_n = us.transform(u_full)
    with torch.no_grad():
        ut = torch.FloatTensor(u_n[None]).to(DEVICE)
        pred_n = model(ut)[0].cpu().numpy()
    return pred_n


def train_one(data, name, cfg, smoke=False):
    # normalize
    us = MinMaxScaler()
    ys = MinMaxScaler()
    u_tr = us.fit_transform(data["train_u"])
    y_tr = ys.fit_transform(data["train_y"])
    u_va = us.transform(data["val_u"])
    y_va = ys.transform(data["val_y"])

    L = cfg["seq_len"]
    X_tr, Y_tr = make_windows(u_tr, y_tr, L, cfg["stride"])
    X_va, Y_va = make_windows(u_va, y_va, L, L)  # non-overlapping val windows

    model = build_model(name, cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.MSELoss()
    n_epochs = 3 if smoke else cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    n_w = X_tr.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    history = []          # per-epoch train/val loss + lr
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_w)
        tot = 0.0; nb = 0
        for i in range(0, n_w, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = torch.FloatTensor(X_tr[idx]).to(DEVICE)
            yb = torch.FloatTensor(Y_tr[idx]).to(DEVICE)
            b = xb.shape[0]
            xm = torch.zeros(b, L, 4, device=DEVICE)
            xd = torch.zeros(b, L, 3, device=DEVICE)
            xmd = torch.zeros(b, L, 4, device=DEVICE)
            opt.zero_grad()
            out = model(xb, xm, xd, xmd)
            loss = crit(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()
            tot += loss.item() * b; nb += b
        # validate
        model.eval()
        with torch.no_grad():
            nv = X_va.shape[0]
            xv = torch.FloatTensor(X_va).to(DEVICE)
            yv = torch.FloatTensor(Y_va).to(DEVICE)
            xm = torch.zeros(nv, L, 4, device=DEVICE)
            xd = torch.zeros(nv, L, 3, device=DEVICE)
            xmd = torch.zeros(nv, L, 4, device=DEVICE)
            vout = model(xv, xm, xd, xmd)
            vloss = crit(vout, yv).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"  [{name}] early stop @ epoch {ep + 1}, best val {best_val:.6f}")
                break
        history.append({"epoch": ep + 1, "train_loss": float(tot / max(nb, 1)),
                        "val_loss": float(vloss), "lr": float(sched.get_last_lr()[0])})
        if (ep + 1) % 25 == 0 or smoke:
            print(f"  [{name}] epoch {ep + 1}/{n_epochs} train {tot / max(nb,1):.6f} val {vloss:.6f} lr={sched.get_last_lr()[0]:.2e}")
        sched.step()

    model.load_state_dict(best_state)
    train_time = time.time() - t0
    return model, us, ys, train_time, history


# eval/plot protocol: overlapping sliding window (< seq_len); each time point is
# averaged over the ~L/stride windows that cover it, removing stitch seams
EVAL_STRIDE = 200


def evaluate_full(model, u_full, us, L, stride=EVAL_STRIDE):
    """Predict the full 8000 steps with overlapping sliding windows and average; returns [T,3] in the normalized domain.
    With stride < L each time point is covered by ~L/stride windows and averaged (smooths the seams).
    Metrics and plots use this protocol."""
    model.eval()
    u_n = us.transform(u_full)
    T = len(u_n)
    acc = np.zeros((T, 3))
    cnt = np.zeros(T)
    with torch.no_grad():
        for t in range(0, T, stride):
            seg = u_n[t:t + L]
            n_actual = len(seg)
            if n_actual < L:
                pad = np.zeros((L - n_actual, 6)); pad[:] = seg[-1]
                seg = np.vstack([seg, pad])
            xb = torch.FloatTensor(seg[None]).to(DEVICE)
            b = 1
            xm = torch.zeros(b, L, 4, device=DEVICE)
            xd = torch.zeros(b, L, 3, device=DEVICE)
            xmd = torch.zeros(b, L, 4, device=DEVICE)
            out = model(xb, xm, xd, xmd)[0].cpu().numpy()[:n_actual]
            acc[t:t + n_actual] += out
            cnt[t:t + n_actual] += 1
    cnt = np.where(cnt == 0, 1, cnt)
    return acc / cnt[:, None]


def calc_metrics(y_true, y_pred):
    """Full metric set: R2 / RMSE / MAE / NRMSE / FreqErr / AvgAbsError(dB).
    Same metric protocol as the main method."""
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
    # freq response error: complex-spectrum L1 (same as the SpectralLoss freq term)
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


def save_sota_model(name, state_dict, us, ys, seed=0, extra=None):
    """Save the SOTA best state + matching scalers (per seed) for later reloading/metrics/plots."""
    import joblib
    save_dir = os.path.join("results", "saved_models")
    sc_dir = os.path.join(save_dir, "sota_scalers_nolinear")
    os.makedirs(sc_dir, exist_ok=True)
    ckpt = {"model_state_dict": state_dict, "name": name, "seed": seed, "config": extra}
    mp = os.path.join(save_dir, f"sota_{name}_seed{seed}_nolinear.pt")
    torch.save(ckpt, mp)
    joblib.dump(us, os.path.join(sc_dir, f"sota_{name}_seed{seed}_nolinear_input_scaler.pkl"))
    joblib.dump(ys, os.path.join(sc_dir, f"sota_{name}_seed{seed}_nolinear_output_scaler.pkl"))
    return mp


def main():
    smoke = SMOKE
    seeds = SEEDS
    if smoke:
        seeds = SEEDS[:2]

    def _seed_all(s):
        import random as _r
        _r.seed(s); np.random.seed(s); torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"===== SOTA comparison (thuml implementations, Nolinear STEP) =====")
    print(f"models: {MODELS} | seeds={seeds} | epochs={3 if smoke else 200} | device={DEVICE}")
    data = load_data()
    results = {}      # name -> seed -> metrics
    histories = {}    # name -> seed -> per-epoch logs
    for name in MODELS:
        cfg = MODEL_CFG[name]
        results[name] = {}
        histories[name] = {}
        print(f"\n>>> training {name} (seeds={seeds}, cfg={ {k: v for k, v in cfg.items() if k != 'kind'} })...")
        best_seed, best_tr_r2 = None, -1e9
        best_pred_tr = best_pred_te = None
        best_state = best_us = best_ys = None
        for seed in seeds:
            _seed_all(seed)
            t0 = time.time()
            if cfg["kind"] == "tcn_full":
                model, us, ys, t_tr, hist = train_tcn(data, cfg, smoke)
                pred_tr = tcn_predict_full(model, data["train_u"], us)
                pred_te = tcn_predict_full(model, data["test_u"], us)
            else:
                L = cfg["seq_len"]
                model, us, ys, t_tr, hist = train_one(data, name, cfg, smoke)
                pred_tr = evaluate_full(model, data["train_u"], us, L)
                pred_te = evaluate_full(model, data["test_u"], us, L)
            # metrics in the normalized domain
            y_tr_n = ys.transform(data["train_y"])
            y_te_n = ys.transform(data["test_y"])
            tr_r2, tr_rmse, tr_mae, tr_nrmse, tr_freq, tr_abs = calc_metrics(y_tr_n, pred_tr)
            te_r2, te_rmse, te_mae, te_nrmse, te_freq, te_abs = calc_metrics(y_te_n, pred_te)
            model_path = save_sota_model(name, model.state_dict(), us, ys, seed=seed,
                                         extra={k: v for k, v in cfg.items()})
            # save per-seed predictions (normalized domain)
            npy_dir = os.path.join("results", "multi_curve_data")
            os.makedirs(npy_dir, exist_ok=True)
            np.save(os.path.join(npy_dir, f"sota_{name}_seed{seed}_train_preds_nolinear.npy"), pred_tr)
            np.save(os.path.join(npy_dir, f"sota_{name}_seed{seed}_test_preds_nolinear.npy"), pred_te)
            if name == MODELS[0] and seed == seeds[0]:
                np.save(os.path.join(npy_dir, "sota_test_labels_nolinear.npy"), y_te_n)
            results[name][seed] = {
                # --- train metrics (final model) ---
                "train_r2": float(tr_r2),
                "train_rmse_avg": float(tr_rmse.mean()),
                "train_rmse": [float(x) for x in tr_rmse],
                "train_mae_avg": float(tr_mae.mean()),
                "train_mae": [float(x) for x in tr_mae],
                "train_nrmse_avg": float(tr_nrmse.mean()),
                "train_nrmse": [float(x) for x in tr_nrmse],
                "train_freq_err_avg": float(tr_freq.mean()),
                "train_freq_err": [float(x) for x in tr_freq],
                "train_avg_abs_db_avg": float(tr_abs.mean()),
                "train_avg_abs_db": [float(x) for x in tr_abs],
                # --- test metrics (best model) ---
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
                "model_path": model_path,
                "train_time_s": float(t_tr),
            }
            histories[name][seed] = hist
            # pick the best seed by train R2 and save the generic model/predictions
            if not (isinstance(tr_r2, float) and np.isnan(tr_r2)) and tr_r2 > best_tr_r2:
                best_tr_r2, best_seed = tr_r2, seed
                best_pred_tr, best_pred_te = pred_tr, pred_te
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_us, best_ys = us, ys
            print(f"  [{name}] seed={seed} train R2={tr_r2:.4f} | "
                  f"test R2={te_r2:.4f} RMSE={te_rmse.mean():.5f} | {time.time()-t0:.0f}s")
        # also save under a generic name (for Plot_Multi_Models / later loading)
        if best_seed is not None:
            import joblib
            npy_dir = os.path.join("results", "multi_curve_data")
            save_dir = os.path.join("results", "saved_models")
            sc_dir = os.path.join(save_dir, "sota_scalers_nolinear")
            np.save(os.path.join(npy_dir, f"sota_{name}_train_preds_nolinear.npy"), best_pred_tr)
            np.save(os.path.join(npy_dir, f"sota_{name}_test_preds_nolinear.npy"), best_pred_te)
            torch.save({"model_state_dict": best_state, "name": name, "seed": best_seed},
                       os.path.join(save_dir, f"sota_{name}_nolinear.pt"))
            joblib.dump(best_us, os.path.join(sc_dir, f"sota_{name}_nolinear_input_scaler.pkl"))
            joblib.dump(best_ys, os.path.join(sc_dir, f"sota_{name}_nolinear_output_scaler.pkl"))
            print(f"  [{name}] best seed={best_seed} (train R2={best_tr_r2:.4f}) -> generic sota_{name}_nolinear.* / sota_{name}_preds_nolinear.npy")

    # ---------- cross-seed mean +/- std per model (train + test) ----------
    def mean_std(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float("nan"), float("nan"), 0
        m = float(arr.mean())
        s = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return m, s, len(arr)

    print("\n================ per-model mean +/- std (across seeds) ================")
    print("--- test metrics ---")
    print(f"{'model':<14}{'n':<4}{'test R2':<22}{'RMSE':<18}{'MAE':<18}{'NRMSE':<18}{'FreqErr':<18}{'AvgAbs':<16}")
    summary = {}
    for name in MODELS:
        rows = list(results[name].values())
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
        summary[name] = {"test": {"r2_mean": m_r2, "r2_std": s_r2,
                                  "rmse_mean": m_rm, "rmse_std": s_rm,
                                  "mae_mean": m_mae, "mae_std": s_mae,
                                  "nrmse_mean": m_nrm, "nrmse_std": s_nrm,
                                  "freq_err_mean": m_fq, "freq_err_std": s_fq,
                                  "avg_abs_db_mean": m_ab, "avg_abs_db_std": s_ab,
                                  "n": n}}
        if np.isnan(m_r2):
            print(f"{name:<14}{n:<4}{'diverged':<22}")
        else:
            print(f"{name:<14}{n:<4}{m_r2:.4f} ± {s_r2:.4f} {m_rm:.5f} ± {s_rm:.5f} "
                  f"{m_mae:.5f} ± {s_mae:.5f} {m_nrm:.5f} ± {s_nrm:.5f} "
                  f"{m_fq:.5f} ± {s_fq:.5f} {m_ab:.4f} ± {s_ab:.4f}")

    print("--- train metrics ---")
    print(f"{'model':<14}{'n':<4}{'train R2':<22}{'RMSE':<18}{'MAE':<18}{'NRMSE':<18}{'FreqErr':<18}{'AvgAbs':<16}")
    for name in MODELS:
        rows = list(results[name].values())
        r2s = [r["train_r2"] for r in rows]
        rms = [r["train_rmse_avg"] for r in rows]
        maes = [r["train_mae_avg"] for r in rows]
        nrms = [r["train_nrmse_avg"] for r in rows]
        freqs = [r["train_freq_err_avg"] for r in rows]
        abss = [r["train_avg_abs_db_avg"] for r in rows]
        m_r2, s_r2, n = mean_std(r2s)
        m_rm, s_rm, _ = mean_std(rms)
        m_mae, s_mae, _ = mean_std(maes)
        m_nrm, s_nrm, _ = mean_std(nrms)
        m_fq, s_fq, _ = mean_std(freqs)
        m_ab, s_ab, _ = mean_std(abss)
        summary[name]["train"] = {"r2_mean": m_r2, "r2_std": s_r2,
                                  "rmse_mean": m_rm, "rmse_std": s_rm,
                                  "mae_mean": m_mae, "mae_std": s_mae,
                                  "nrmse_mean": m_nrm, "nrmse_std": s_nrm,
                                  "freq_err_mean": m_fq, "freq_err_std": s_fq,
                                  "avg_abs_db_mean": m_ab, "avg_abs_db_std": s_ab,
                                  "n": n}
        if np.isnan(m_r2):
            print(f"{name:<14}{n:<4}{'diverged':<22}")
        else:
            print(f"{name:<14}{n:<4}{m_r2:.4f} ± {s_r2:.4f} {m_rm:.5f} ± {s_rm:.5f} "
                  f"{m_mae:.5f} ± {s_mae:.5f} {m_nrm:.5f} ± {s_nrm:.5f} "
                  f"{m_fq:.5f} ± {s_fq:.5f} {m_ab:.4f} ± {s_ab:.4f}")

    # ---------- save (one json per model) ----------
    os.makedirs("results", exist_ok=True)
    for name in MODELS:
        out = os.path.join("results", f"sota_{name}_nolinear.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"config": {"models": MODEL_CFG,
                                  "batch_size": BATCH_SIZE,
                                  "seeds": seeds,
                                  "smoke": smoke, "data": "Nolinear"},
                       "results": {name: results[name]},
                       "summary": {name: summary[name]}}, f, ensure_ascii=False, indent=2)
        print(f"\nresults saved: {out} ({name}, {len(results[name])} seeds)")
    out_h = os.path.join("results", "sota_history_nolinear.json")
    with open(out_h, "w", encoding="utf-8") as f:
        json.dump({"config": {"models": list(MODEL_CFG.keys())},
                   "histories": histories}, f, ensure_ascii=False, indent=2)
    print(f"training logs saved: {out_h} ({len(histories)} models kept)")


if __name__ == "__main__":
    main()

