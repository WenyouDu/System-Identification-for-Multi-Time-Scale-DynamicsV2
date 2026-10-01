# -*- coding: utf-8 -*-
"""
Runs the SOTA baselines (TCN / Transformer / PatchTST) on the public
nonlinear benchmarks (CT / SB / CED / WH): windowed training, then
full-sequence testing. Skip-init evaluation is done by eval_skip_init.py.
"""
import os
import sys
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ============ run config ============
DATASETS = ["CT"]   # CT / SB / CED / WH
MODELS = ["PatchTST"]   # TCN / Transformer / PatchTST
SEEDS = [9, 66, 108, 88, 52]   # random seeds
BATCH_SIZE = 16

# window / stride per dataset (short test sets need small windows)
SEQ_LEN = {"CED": 32, "CT": 128, "SB": 512, "WH": 512}
TRAIN_STRIDE = {"CED": 8, "CT": 64, "SB": 256, "WH": 256}
EVAL_STRIDE = {"CED": 8, "CT": 64, "SB": 200, "WH": 200}   # eval window stride (< seq_len -> overlap averaging)

# per-dataset x per-model config (CED uses a small window + small capacity)
MODEL_CFG = {
    "CT": {
        "TCN": {
            "kind": "window",
            "num_channels": [128, 128, 128, 128, 128, 128], "kernel_size": 3,
            "dropout": 0.1, "lr": 0.0001, "epochs": 1000, "patience": 300,
        },
        "Transformer": {
            "kind": "thuml",
            "d_model": 128, "n_heads": 4, "e_layers": 2, "d_layers": 1, "d_ff": 256,
            "dropout": 0.2, "lr": 0.0005, "epochs": 2000, "patience": 500,
        },
        "PatchTST": {
            "kind": "patchtst",
            "d_model": 128, "n_heads": 4, "e_layers": 2, "d_ff": 256,
            "patch_len": 16, "patch_stride": 8,
            "dropout": 0.2, "lr": 0.0005, "epochs": 3000, "patience": 600,
        },
    },
    "SB": {
        "TCN": {
            "kind": "window",
            "num_channels": [64, 64, 64, 64, 64, 64, 64, 64], "kernel_size": 3,
            "dropout": 0.2, "lr": 0.001, "epochs": 1000, "patience": 300,
        },
        "Transformer": {
            "kind": "thuml",
            "d_model": 64, "n_heads": 4, "e_layers": 1, "d_layers": 1, "d_ff": 512,
            "dropout": 0.2, "lr": 0.0001, "epochs": 1000, "patience": 300,
        },
        "PatchTST": {
            "kind": "patchtst",
            "d_model": 64, "n_heads": 4, "e_layers": 1, "d_ff": 128,
            "patch_len": 16, "patch_stride": 8,
            "dropout": 0.1, "lr": 0.0005, "epochs": 3000, "patience": 600,
        },
    },
    "WH": {
        "TCN": {
            "kind": "window",
            "num_channels": [64, 64, 64, 64, 64, 64, 64, 64], "kernel_size": 3,
            "dropout": 0.2, "lr": 0.001, "epochs": 1000, "patience": 300,
        },
        "Transformer": {
            "kind": "thuml",
            "d_model": 64, "n_heads": 4, "e_layers": 1, "d_layers": 1, "d_ff": 512,
            "dropout": 0.2, "lr": 0.002, "epochs": 1000, "patience": 300,
        },
        "PatchTST": {
            "kind": "patchtst",
            "d_model": 128, "n_heads": 4, "e_layers": 1, "d_ff": 128,
            "patch_len": 16, "patch_stride": 8,
            "dropout": 0.1, "lr": 0.0005, "epochs": 1000, "patience": 300,
        },
    },
    "CED": {
        "TCN": {
            "kind": "window",
            "num_channels": [32, 32, 32, 32], "kernel_size": 3,
            "dropout": 0.2, "lr": 0.001, "epochs": 1000, "patience": 300,
        },
        "Transformer": {
            "kind": "thuml",
            "d_model": 64, "n_heads": 2, "e_layers": 1, "d_layers": 1, "d_ff": 256,
            "dropout": 0.2, "lr": 0.0005, "epochs": 2000, "patience": 500,
        },
        "PatchTST": {
            "kind": "patchtst",
            "d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 128,
            "patch_len": 16, "patch_stride": 8,
            "dropout": 0.1, "lr": 0.1, "epochs": 3000, "patience": 600,
        },
    },
}
SAVE_MODEL = True   # save per-seed best models

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_HERE)

TSLIB = os.path.join(_PARENT, "Time-Series-Library")
if TSLIB not in sys.path:
    sys.path.insert(0, TSLIB)

from Nonlinea_Benchmark_Predict.Option import ALL_DATASETS
from Nonlinea_Benchmark_Predict.train import _concat_and_reshape

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    import random as _r
    _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_fs(data):
    obj = data["train_data"]
    if isinstance(obj, list):
        obj = obj[0] if len(obj) else None
    st = getattr(obj, "sampling_time", None)
    if st is None:
        return 1000.0
    st = float(st)
    return 1.0 / st if st > 0 else 1000.0


def load_data_arrays(ds_name):
    """Extract arrays + fs + rmse_scaling for a dataset."""
    data = ALL_DATASETS[ds_name]
    train_u = _concat_and_reshape(data["train_data"], "u")   # [N, n_in]
    train_y = _concat_and_reshape(data["train_data"], "y")   # [N, n_out]
    val_u = _concat_and_reshape(data["val_data"], "u")
    val_y = _concat_and_reshape(data["val_data"], "y")
    test_list = [(o.u, o.y) for o in data["test_data"]]
    fs = _get_fs(data)
    rmse_scaling = data["opts"]["rmse_scaling"]
    if train_u.ndim == 1: train_u = train_u.reshape(-1, 1)
    if train_y.ndim == 1: train_y = train_y.reshape(-1, 1)
    if val_u.ndim == 1: val_u = val_u.reshape(-1, 1)
    if val_y.ndim == 1: val_y = val_y.reshape(-1, 1)
    test_list = [((u.reshape(-1, 1) if u.ndim == 1 else u),
                  (y.reshape(-1, 1) if y.ndim == 1 else y)) for u, y in test_list]
    return train_u, train_y, val_u, val_y, test_list, fs, rmse_scaling


def make_windows(u, y, seq_len, stride):
    xs, ys = [], []
    n = len(u)
    for t in range(0, n - seq_len + 1, stride):
        xs.append(u[t:t + seq_len])
        ys.append(y[t:t + seq_len])
    return np.stack(xs), np.stack(ys)


def make_config(cfg, seq_len, enc_in, c_out):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len, pred_len=seq_len, label_len=0,
        enc_in=enc_in, c_out=c_out, dec_in=c_out,
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
        e_layers=cfg.get("e_layers", 2), d_layers=cfg.get("d_layers", 1),
        d_ff=cfg.get("d_ff", 256), dropout=cfg.get("dropout", 0.2),
        activation="gelu", embed="fixed", freq="h", factor=5,
        output_attention=False, use_norm=True,
        num_class=1, patch_len=cfg.get("patch_len", 16),
        stride=cfg.get("patch_stride", 8),
        moving_avg=25, top_k=5, num_kernels=6,
        seg_len=cfg.get("seg_len", 48), channel_independence=cfg.get("channel_independence", 1),
        distil=cfg.get("distil", True),
    )


def _load_tslib_model(name):
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


def _load_official_tcn():
    """locuslab/TCN official implementation: weight_norm + Chomp1d + dilations + residual."""
    import importlib.util
    tcn_path = os.path.join(_HERE, "OfficialTCN.py")
    spec = importlib.util.spec_from_file_location("official_tcn", tcn_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["official_tcn"] = mod
    spec.loader.exec_module(mod)
    return mod.TCN


class PatchTSTWrapper(nn.Module):
    """PatchTST wrapper: output channels = enc_in; identity when c_out == enc_in."""

    def __init__(self, inner, enc_in, c_out):
        super().__init__()
        self.inner = inner
        self.proj = nn.Linear(enc_in, c_out) if c_out != enc_in else nn.Identity()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B,L,enc_in]
        return self.proj(out)


class TCNWrapper(nn.Module):
    """TCN wrapper: thuml-style 4-arg forward -> TCN single-arg forward."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.inner(x_enc)


def build_model(name, cfg, seq_len, enc_in, c_out):
    if cfg["kind"] in ("tcn_full", "window"):   # official TCN (same structure, different train/eval modes)
        tcn = _load_official_tcn()(input_size=enc_in, output_size=c_out,
                                   num_channels=cfg["num_channels"],
                                   kernel_size=cfg["kernel_size"], dropout=cfg["dropout"])
        return TCNWrapper(tcn) if cfg["kind"] == "window" else tcn
    if cfg["kind"] == "thuml":
        return _load_tslib_model(name).Model(make_config(cfg, seq_len, enc_in, c_out))
    if cfg["kind"] == "patchtst":
        return PatchTSTWrapper(_load_tslib_model("PatchTST").Model(
            make_config(cfg, seq_len, enc_in, c_out),
            patch_len=cfg.get("patch_len", 16), stride=cfg.get("patch_stride", 8)), enc_in, c_out)
    raise ValueError(f"unknown kind: {cfg['kind']}")


def train_one(u_tr, y_tr, u_va, y_va, name, cfg, seq_len, stride, smoke=False):
    us = MinMaxScaler(); ys = MinMaxScaler()
    u_tr_n = us.fit_transform(u_tr); y_tr_n = ys.fit_transform(y_tr)
    u_va_n = us.transform(u_va); y_va_n = ys.transform(y_va)

    enc_in = u_tr.shape[1]; c_out = y_tr.shape[1]

    model = build_model(name, cfg, seq_len, enc_in, c_out).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.MSELoss()
    n_epochs = 3 if smoke else cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    best_val, best_state, bad = float("inf"), None, 0
    best_epoch = 0
    history = []
    t0 = time.time()

    if cfg["kind"] == "tcn_full":
        # ---- TCN full-sequence mode: whole u [1,T,C] in one forward ----
        ut = torch.FloatTensor(u_tr_n[None]).to(DEVICE)
        yt = torch.FloatTensor(y_tr_n[None]).to(DEVICE)
        uv = torch.FloatTensor(u_va_n[None]).to(DEVICE)
        yv = torch.FloatTensor(y_va_n[None]).to(DEVICE)
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
                vloss = crit(model(uv), yv).item()
            if vloss < best_val:
                best_val = vloss
                best_epoch = ep + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    if not smoke:
                        print(f"  [{name}] early stop @ epoch {ep + 1}, best val {best_val:.6f}")
                    break
            history.append({"epoch": ep + 1, "train_loss": float(loss.item()),
                            "val_loss": float(vloss), "lr": float(sched.get_last_lr()[0])})
            if (ep + 1) % 25 == 0 or smoke:
                print(f"  [{name}] epoch {ep + 1}/{n_epochs} train {loss.item():.6f} val {vloss:.6f}")
            sched.step()
        model.load_state_dict(best_state)
        return model, us, ys, time.time() - t0, history, best_epoch

    # ---- windowed mode: thuml / PatchTST ----
    X_tr, Y_tr = make_windows(u_tr_n, y_tr_n, seq_len, stride)
    X_va, Y_va = make_windows(u_va_n, y_va_n, seq_len, seq_len)

    n_w = X_tr.shape[0]
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_w)
        tot = 0.0; nb = 0
        for i in range(0, n_w, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = torch.FloatTensor(X_tr[idx]).to(DEVICE)
            yb = torch.FloatTensor(Y_tr[idx]).to(DEVICE)
            b = xb.shape[0]
            xm = torch.zeros(b, seq_len, 4, device=DEVICE)
            xd = torch.zeros(b, seq_len, c_out, device=DEVICE)
            xmd = torch.zeros(b, seq_len, 4, device=DEVICE)
            out = model(xb, xm, xd, xmd)
            loss = crit(out, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()
            tot += loss.item() * b; nb += b
        # validation
        model.eval()
        with torch.no_grad():
            nv = X_va.shape[0]
            xv = torch.FloatTensor(X_va).to(DEVICE)
            yv = torch.FloatTensor(Y_va).to(DEVICE)
            xm = torch.zeros(nv, seq_len, 4, device=DEVICE)
            xd = torch.zeros(nv, seq_len, c_out, device=DEVICE)
            xmd = torch.zeros(nv, seq_len, 4, device=DEVICE)
            vout = model(xv, xm, xd, xmd)
            vloss = crit(vout, yv).item()
        if vloss < best_val:
            best_val = vloss
            best_epoch = ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if not smoke:
                    print(f"  [{name}] early stop @ epoch {ep + 1}, best val {best_val:.6f}")
                break
        history.append({"epoch": ep + 1, "train_loss": float(tot / max(nb, 1)),
                        "val_loss": float(vloss), "lr": float(sched.get_last_lr()[0])})
        if (ep + 1) % 25 == 0 or smoke:
            print(f"  [{name}] epoch {ep + 1}/{n_epochs} train {tot / max(nb, 1):.6f} val {vloss:.6f}")
        sched.step()

    model.load_state_dict(best_state)
    return model, us, ys, time.time() - t0, history, best_epoch


def evaluate_full(model, u_full, us, seq_len, cfg, stride):
    """Predict the full test u, return normalized preds [T, c_out].
    tcn_full: one forward over the whole segment; windowed: overlap-average."""
    model.eval()
    u_n = us.transform(u_full)
    if cfg["kind"] == "tcn_full":
        with torch.no_grad():
            out = model(torch.FloatTensor(u_n[None]).to(DEVICE))[0].cpu().numpy()
        return out  # [T, c_out]
    # ---- windowed: infer the output channels ----
    with torch.no_grad():
        xb = torch.FloatTensor(u_n[:seq_len][None]).to(DEVICE)
        b = 1
        xm = torch.zeros(b, seq_len, 4, device=DEVICE)
        xd = torch.zeros(b, seq_len, 1, device=DEVICE)
        xmd = torch.zeros(b, seq_len, 4, device=DEVICE)
        out0 = model(xb, xm, xd, xmd)
    c_out = out0.shape[-1]
    T = len(u_n)
    acc = np.zeros((T, c_out)); cnt = np.zeros(T)
    with torch.no_grad():
        for t in range(0, T, stride):
            seg = u_n[t:t + seq_len]
            n_actual = len(seg)
            if n_actual < seq_len:
                pad = np.zeros((seq_len - n_actual, u_n.shape[1])); pad[:] = seg[-1]
                seg = np.vstack([seg, pad])
            xb = torch.FloatTensor(seg[None]).to(DEVICE)
            b = 1
            xm = torch.zeros(b, seq_len, 4, device=DEVICE)
            xd = torch.zeros(b, seq_len, c_out, device=DEVICE)
            xmd = torch.zeros(b, seq_len, 4, device=DEVICE)
            out = model(xb, xm, xd, xmd)[0].cpu().numpy()
            acc[t:t + n_actual] += out[:n_actual]
            cnt[t:t + n_actual] += 1
    cnt = np.where(cnt == 0, 1, cnt)
    return acc / cnt[:, None]


def calc_metrics(y_true, y_pred, fs=1000.0):
    """Six normalized-domain metrics, same protocol as run_benchmark_multi.calc_metrics."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    if (np.isnan(y_true).any() or np.isnan(y_pred).any()
            or np.isinf(y_true).any() or np.isinf(y_pred).any()):
        return {k: float("nan") for k in ["r2", "rmse", "mae", "nrmse", "freq_err", "avg_abs_db"]}
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
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"r2": r2, "rmse": float(np.mean(rmse)), "mae": float(np.mean(mae)),
            "nrmse": float(np.mean(nrmse)), "freq_err": float(np.mean(freq_err)),
            "avg_abs_db": float(np.mean(avg_abs_db))}


def save_sota_model(ds, name, model, seed, cfg):
    import joblib
    save_dir = os.path.join("results", "saved_models", f"{ds}_{name}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "name": name,
                "seed": seed, "config": cfg},
               os.path.join(save_dir, f"{ds}_{name}_seed{seed}.pt"))


def main():
    smoke = "--smoke" in sys.argv
    seeds = SEEDS[:2] if smoke else SEEDS
    ds = DATASETS[0]
    seq_len = SEQ_LEN[ds]; tr_stride = TRAIN_STRIDE[ds]; ev_stride = EVAL_STRIDE[ds]

    print("===== SOTA comparison on public benchmarks (thuml implementations) =====")
    print(f"dataset={ds} seq_len={seq_len} | models={MODELS} | seeds={seeds} | device={DEVICE}")

    train_u, train_y, val_u, val_y, test_list, fs, rmse_scaling = load_data_arrays(ds)
    print(f"train u{train_u.shape} y{train_y.shape} | test sets {len(test_list)} | fs={fs:.4f} | rmse_scaling={rmse_scaling}")

    results = {}
    for name in MODELS:
        cfg = MODEL_CFG[ds][name]
        results[name] = {"seeds": seeds, "fs": fs, "rmse_scaling": rmse_scaling,
                         "n_test": len(test_list), "per_test_set": {}, "overall": {}}
        per_ts = [{} for _ in test_list]   # per test set: metric -> [across seeds]
        seed_rows = []   # per-seed details
        print(f"\n>>> training {name} ({cfg})...")
        for seed in seeds:
            set_seed(seed)
            t0 = time.time()
            model, us, ys, t_tr, hist, best_epoch = train_one(
                train_u, train_y, val_u, val_y, name, cfg, seq_len, tr_stride, smoke)
            if SAVE_MODEL:
                save_sota_model(ds, name, model, seed, cfg)
            # per test set: predict + metrics (no skipping here; see eval_skip_init.py)
            seed_r2s, seed_rmses = [], []
            for ts_idx, (u_te, y_te) in enumerate(test_list):
                pred = evaluate_full(model, u_te, us, seq_len, cfg, ev_stride)
                y_te_n = ys.transform(y_te)
                T = min(len(pred), len(y_te_n))
                pred = pred[:T]
                y_te_n = y_te_n[:T]
                y_te_raw = y_te[:T]
                m = calc_metrics(y_te_n, pred, fs=fs)
                m["n_init"] = 0
                # scaledRMSE (raw x rmse_scaling, full sequence)
                pred_raw = ys.inverse_transform(pred)
                rmse_raw = float(np.sqrt(np.mean((pred_raw - y_te_raw) ** 2)))
                m["scaled_rmse"] = rmse_raw * rmse_scaling
                for k, v in m.items():
                    per_ts[ts_idx].setdefault(k, []).append(v)
                seed_r2s.append(m["r2"]); seed_rmses.append(m["rmse"])
            seed_rows.append({"seed": seed, "best_epoch": best_epoch,
                              "train_time_s": round(time.time() - t0, 1),
                              "r2": seed_r2s, "rmse": seed_rmses})
            print(f"  [{name}] seed={seed} best_epoch={best_epoch} "
                  f"test_R2={['%.4f' % r for r in seed_r2s]} time={time.time() - t0:.0f}s")
        results[name]["per_seed"] = seed_rows

        print(f"\n  == {ds} {name} per-seed details ==")
        if len(test_list) == 1:
            print(f"    {'seed':<6}{'best_epoch':<11}{'test R2':<10}{'test RMSE':<10}{'time(s)':<8}")
            for row in seed_rows:
                print(f"    {row['seed']:<6}{row['best_epoch']:<11}"
                      f"{row['r2'][0]:<10.4f}{row['rmse'][0]:<10.4f}{row['train_time_s']:<8}")
        else:
            hdr = "    " + "".join([f"{'seed':<6}{'best_ep':<9}"] +
                                   [f"ts{i}_R2{'':<8}" for i in range(len(test_list))])
            print(hdr)
            for row in seed_rows:
                line = f"    {row['seed']:<6}{row['best_epoch']:<9}"
                for r in row["r2"]:
                    line += f"{r:<12.4f}"
                print(line)

        # cross-seed mean +/- std (per test set + overall)
        def ms(arr):
            a = np.asarray(arr, dtype=float); a = a[~np.isnan(a)]
            if len(a) == 0: return float("nan"), float("nan")
            return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

        metric_keys = list(per_ts[0].keys())
        for ts_idx in range(len(test_list)):
            results[name]["per_test_set"][ts_idx] = {}
            for k in metric_keys:
                m, s = ms(per_ts[ts_idx][k])
                results[name]["per_test_set"][ts_idx][k + "_mean_std"] = [m, s]
        for k in metric_keys:
            arrs = [per_ts[ts][k] for ts in range(len(test_list))]
            flat = [v for a in arrs for v in a]
            m, s = ms(flat)
            results[name]["overall"][k + "_mean_std"] = [m, s]

        print(f"\n  == {ds} {name} per test set (mean+/-std) ==")
        for ts_idx in range(len(test_list)):
            p = results[name]["per_test_set"][ts_idx]
            print(f"    test_set {ts_idx}: R2={p['r2_mean_std'][0]:.5f}±{p['r2_mean_std'][1]:.5f} "
                  f"RMSE={p['rmse_mean_std'][0]:.4f}±{p['rmse_mean_std'][1]:.4f} "
                  f"MAE={p['mae_mean_std'][0]:.4f}±{p['mae_mean_std'][1]:.4f} "
                  f"NRMSE={p['nrmse_mean_std'][0]:.4f}±{p['nrmse_mean_std'][1]:.4f} "
                  f"FreqErr={p['freq_err_mean_std'][0]:.4f}±{p['freq_err_mean_std'][1]:.4f} "
                  f"AvgAbs={p['avg_abs_db_mean_std'][0]:.4f}±{p['avg_abs_db_mean_std'][1]:.4f} "
                  f"scaledRMSE={p['scaled_rmse_mean_std'][0]:.4f}±{p['scaled_rmse_mean_std'][1]:.4f}")

        # save per dataset+model
        os.makedirs("results", exist_ok=True)
        out = os.path.join("results", f"benchmark_sota_{ds}_{name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"dataset": ds, "model": name,
                       "config": {"model_cfg": cfg, "seq_len": seq_len,
                                  "train_stride": tr_stride, "eval_stride": ev_stride,
                                  "batch_size": BATCH_SIZE, "seeds": seeds,
                                  "fs": fs, "rmse_scaling": rmse_scaling},
                       "results": results[name]}, f, ensure_ascii=False, indent=2)
        print(f"    saved: {out}")

    print("\n================ summary (overall mean+/-std) ================")
    print(f"{'Model':<14} {'R2':<18} {'RMSE':<14} {'MAE':<14} {'NRMSE':<14} {'FreqErr':<14} {'AvgAbs':<12} {'scaledRMSE':<12}")
    for name in MODELS:
        o = results[name]["overall"]
        print(f"{name:<14} {o['r2_mean_std'][0]:.5f}±{o['r2_mean_std'][1]:.5f} "
              f"{o['rmse_mean_std'][0]:.4f}±{o['rmse_mean_std'][1]:.4f} "
              f"{o['mae_mean_std'][0]:.4f}±{o['mae_mean_std'][1]:.4f} "
              f"{o['nrmse_mean_std'][0]:.4f}±{o['nrmse_mean_std'][1]:.4f} "
              f"{o['freq_err_mean_std'][0]:.4f}±{o['freq_err_mean_std'][1]:.4f} "
              f"{o['avg_abs_db_mean_std'][0]:.4f}±{o['avg_abs_db_mean_std'][1]:.4f} "
              f"{o['scaled_rmse_mean_std'][0]:.4f}±{o['scaled_rmse_mean_std'][1]:.4f}")


if __name__ == "__main__":
    main()
