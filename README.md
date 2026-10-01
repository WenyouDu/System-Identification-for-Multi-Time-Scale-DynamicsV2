# System Identification for Multi-Time-Scale Dynamics

Code accompanying the manuscript *"System Identification for Multi-Time-Scale Dynamics"* (submission ID ISATRANS-D-26-02897).

This repository implements an **Enhanced LSTM** for nonlinear system identification in the **simulation-error (free-run)** setting, targeting dynamic systems that exhibit both long-term (slow-transient) and short-term (transient) characteristics. It covers:

1. **STEP experiment** — a staircase-response simulation with saturation and dead-zone nonlinearities (6 inputs, 3 outputs, 8000-step sequences).
2. **Public benchmarks** — Cascaded Tanks (CT), Wiener–Hammerstein (WH), Silverbox (SB), and Coupled Electric Drives (CED) from the [nonlinear benchmark collection](https://www.nonlinearbenchmark.org/).

## Directory layout

```
.
├── hyperparameters.json              # Central network configuration registry
├── Step_Benchmark_Predict/           # STEP staircase-response experiment
│   ├── run_baseline_multi_nolinear.py    # RNN / GRU / Baseline LSTM on STEP (5 seeds)
│   ├── run_sota_nolinear.py              # TCN / Transformer / PatchTST on STEP
│   ├── run_ablation_lstm_nolinear.py     # Ablation study (4 configurations)
│   ├── scan_forget_bias_nolinear.py      # Sensitivity of forget-gate bias b_f
│   ├── scan_alpha_seeds_nolinear.py      # Sensitivity of coupling coefficient alpha
│   ├── compute_significance.py           # Paired t-test across configurations
│   ├── noise_robustness_nolinear.py      # Noise robustness test (SNR sweep)
│   ├── train.py                          # Enhanced LSTM trainer (SpectralLoss, free-run)
│   ├── Enhanced_LSTM.py                  # Enhanced LSTM model definition
│   ├── BaselineRNN.py / BaselineGRU.py / BaselineLSTM.py
│   ├── OfficialTCN.py                    # TCN model definition
│   └── Step_Dataset/                     # STEP data files
│       ├── Step_Train_Data_Nolinear.xls      # training (8000 steps)
│       ├── Step_Test_Data_Nolinear.xls       # test (10001 steps)
│       └── Step_Validata_Data_Nolinear.xls   # validation (8000 steps)
├── Nonlinea_Benchmark_Predict/       # Public benchmark experiments (CT / WH / SB / CED)
│   ├── run_benchmark_multi.py        # Enhanced LSTM (alpha=0.95) on benchmarks
│   ├── run_benchmark_sota.py         # TCN / Transformer / PatchTST on benchmarks
│   ├── eval_skip_init.py             # Skip-first-50 re-evaluation (benchmark protocol)
│   ├── Option.py                     # Dataset loading, split, and per-dataset model hyperparameters
│   ├── train.py                      # Benchmark trainer
│   └── Nonlinear_Benchmark_Dataset/  # CT / WH / SB data files (CED via package)
├── MatlabCode/                        # MATLAB baselines (SS + NARX)
│   ├── Data/                          # Preprocessed .mat data (STEP / CT / SB / WH)
│   ├── SS/                            # State-space (n4sid) training & free-run test scripts
│   └── NARX/                          # NARX (narxnet) training, grid search & free-run test scripts
└── Time-Series-Library/              # Third-party library (thuml), used for TCN/Transformer/PatchTST
```

## Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch (>= 2.0) with CUDA for GPU training. Public benchmark data are downloaded automatically by the `nonlinear_benchmarks` package on first use.

## Data

- **STEP data** are provided as Excel files (`.xls`) in `Step_Benchmark_Predict/Step_Dataset/`. Each record contains 6 input channels and 3 output channels. All inputs/outputs are min-max normalized to `[0, 1]` using **training-set statistics only**; all reported metrics are computed in the normalized domain.
- **Benchmark data** (CT / SB / WH / CED) are loaded via `Option.py` from the `nonlinear_benchmarks` package or local CSV files.
- **MATLAB baseline data** (preprocessed `.mat` files for STEP / CT / SB / WH / CED) are provided under `MatlabCode/Data/`; each MATLAB script resolves its inputs relative to its own directory.

## Experiments and paper tables

| Script | Description | Paper location |
|---|---|---|
| `Step_Benchmark_Predict/run_baseline_multi_nolinear.py` | RNN / GRU / Baseline LSTM, hidden=64, layers=1, 5 seeds | Table: STEP model comparison (RNN/GRU/Baseline LSTM rows) |
| `Step_Benchmark_Predict/run_sota_nolinear.py` | TCN / Transformer / PatchTST, 5 seeds | Table: STEP model comparison (SOTA rows) |
| `Step_Benchmark_Predict/run_ablation_lstm_nolinear.py` | Configs (i)–(iv): MSE vs. Spectral loss × b_f = 0/1, hidden=256, layers=2 | Table: ablation study (four configurations) |
| `Step_Benchmark_Predict/scan_forget_bias_nolinear.py` | b_f ∈ {0, 0.5, 1.0, 1.5} | Table: forget-gate bias sensitivity |
| `Step_Benchmark_Predict/scan_alpha_seeds_nolinear.py` | alpha sweep | Table: coupling coefficient sensitivity |
| `Step_Benchmark_Predict/compute_significance.py` | Paired two-sample t-test (five seeds) between config (iv) and others | Note under ablation table |
| `Step_Benchmark_Predict/noise_robustness_nolinear.py` | Additive Gaussian white noise on input+output, SNR ∈ {40, 30, 20, 10, 5} dB, 5 seeds, retrained from scratch | Table: noise robustness |
| `Nonlinea_Benchmark_Predict/run_benchmark_multi.py` | Enhanced LSTM (alpha=0.95) on CT / WH / SB / CED, 5 seeds | Table: benchmark RMSE comparison |
| `Nonlinea_Benchmark_Predict/run_benchmark_sota.py` | TCN / Transformer / PatchTST on CT / WH / SB / CED | Table: benchmark SOTA rows |
| `Nonlinea_Benchmark_Predict/eval_skip_init.py` | Re-evaluate trained models skipping the first 50 steps (CT/SB/WH) or 10 steps (CED) | Benchmark results (skip-init protocol) |
| `MatlabCode/SS/{STEP,CT,SB,WH,CED}_SS_Train.m` | `n4sid` continuous SS models, order sweep (STEP 1-15; CT/SB/WH/CED 1-10) | Table: SS rows (all datasets) |
| `MatlabCode/SS/{STEP,CT,SB,WH,CED}_SS_Test.m` | Free-run simulation test (`c2d` + model initial state), normalized-domain metrics | Table: SS rows (all datasets) |
| `MatlabCode/NARX/STEP_NARX_Train.m` / `STEP_NARX_Grid_Search.m` | `narxnet(1:10,1:10,10)` + delay×hidden grid search, 5 seeds | Table: STEP NARX row |
| `MatlabCode/NARX/STEP_NARX_Grid_Test.m` | Free-run test of the grid-search models | Table: STEP NARX row |
| `MatlabCode/NARX/{CT,SB,WH,CED}_NARX_Train.m` | `narxnet` per dataset (CT/WH/CED: delays 1:10; SB: delays 1:2), 5 seeds | Table: benchmark NARX rows |
| `MatlabCode/NARX/*_NARX_Test.m` | Closed-loop (free-run) test, 5 seeds | Table: NARX rows (all datasets) |

Each experiment directory produces its own `results/` subfolder (e.g., `Step_Benchmark_Predict/results/*.json`); per-model prediction arrays used by the paper figures are saved under `results/multi_curve_data/`. Run every script from inside its own directory.

## Configuration

Training hyperparameters for the benchmarks are set in `Option.py` (`MODEL_PARAMETERS`) for the recurrent baselines and the Enhanced LSTM, and in the `MODEL_CFG` dictionaries at the top of `run_benchmark_sota.py` and `eval_skip_init.py` for the SOTA models. Each dataset uses its own configuration to match the reported results. `hyperparameters.json` is the authoritative record of every configuration that produced the paper numbers, but it is not read by the code at runtime. Run-time switches (e.g., `MODELS`, `SEEDS`, `ALPHA` in `run_benchmark_multi.py`) are set at the top of each script.

Key settings of the proposed method: two LSTM layers with 256 hidden units, dropout 0.2, Adam (lr 0.005, weight decay 1e-5), 800 epochs with early stopping, forget-gate bias initialization b_f = 1, and a joint time-frequency loss with coupling coefficient alpha = 0.85 (STEP) / 0.95 (benchmarks).

## MATLAB baselines

The state-space (SS) and NARX baselines are implemented in MATLAB:

- **SS** (requires System Identification Toolbox): continuous-time models estimated with `n4sid` (`n4sidOptions`, `Focus='simulation'`, `Form='free'`, `Ts=0`), sweeping the model order (STEP: 1-15; CT/SB/WH: 1-10). Testing discretizes each model with `c2d` to the dataset sampling interval (zero-order hold) and simulates it in free-run mode from the model initial state, exactly following the System Identification app convention. Metrics are computed in the normalized domain with the same definitions as the Python pipeline.
- **NARX** (requires Deep Learning Toolbox): networks trained in open loop with `narxnet` + `trainlm` (Levenberg-Marquardt, 70/15/15 data division); `rng(seed)` fixes both weight initialization and data division so the five seeds `[9, 66, 108, 88, 52]` reproduce the reported results verbatim. STEP uses input/feedback delays 1:10 and 10 hidden units (`STEP_NARX_Grid_Search.m` additionally sweeps delay ∈ {5, 10, 20} and hidden ∈ {10, 20, 30}); CT and WH use delays 1:10 and SB uses delays 1:2 (all with 10 hidden units). Testing closes the loop (free-run simulation).

## Protocol notes

- **Free-run (simulation-error) training**: no teacher forcing; each training sequence is unrolled over the full horizon and the error is back-propagated through the entire sequence.
- **Skip-initialization**: following the state-initialization protocol defined by the `nonlinear_benchmarks` library, the first `state_initialization_window_length` samples of each public-benchmark test sequence are discarded before computing metrics (50 for CT/SB/WH, 10 for CED), matching the library's built-in `n_init` handling in its error metrics. The STEP experiment does **not** skip initial steps.
- **MATLAB baselines**: the state-space (SS, via `n4sid`) and NARX baselines are implemented in MATLAB; their training, grid-search and free-run test scripts are provided under `MatlabCode/` (see "MATLAB baselines" above).
