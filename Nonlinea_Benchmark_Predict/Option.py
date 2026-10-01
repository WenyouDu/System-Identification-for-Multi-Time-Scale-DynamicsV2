"""
Central config + data loader for the public nonlinear benchmarks (CT / SB / CED / WH).
Importing this module actually loads and splits the four datasets
(train / val / test), so expect it to take a moment and print a few
"Loading ..." lines. It also holds the per-dataset model settings
(MODEL_PARAMETERS) that train.py reads when training.
"""

import nonlinear_benchmarks as nlb
# make sure imports resolve
from Nonlinea_Benchmark_Predict.train import train_LSTM_models

ALL_DATASETS = {}

# --- CascadedTanks (CT) ---
print("Loading CT dataset...")
train_ct_raw, test_ct_raw = nlb.Cascaded_Tanks()
cut_ct = 700
opts_ct = {
    'rmse_scaling': 1.0
}
ALL_DATASETS["CT"] = {
    "train_data": train_ct_raw[:cut_ct],
    "val_data": train_ct_raw[cut_ct:],
    "test_data": [test_ct_raw],
    "opts": opts_ct  # merge dataset opts
}
print(f"CT dataset loaded. Train Data Type: {type(ALL_DATASETS['CT']['train_data'])}")
print(f"CT train_data.u shape: {ALL_DATASETS['CT']['train_data'].u.shape if hasattr(ALL_DATASETS['CT']['train_data'], 'u') else 'N/A'}")

# --- Silverbox (SB) ---
print("Loading SB dataset...")
train_val_sb_raw, tests_sb_raw = nlb.Silverbox()
cut_sb = len(train_val_sb_raw) // 2
opts_sb = {
    'rmse_scaling': 1000.0
}
ALL_DATASETS["SB"] = {
    "train_data": train_val_sb_raw[:cut_sb],
    "val_data": train_val_sb_raw[cut_sb:],
    "test_data": list(tests_sb_raw),
    "opts": opts_sb  # merge dataset opts
}
print(f"SB dataset loaded. Train Data Type: {type(ALL_DATASETS['SB']['train_data'])}")
print(f"SB train_data.u shape: {ALL_DATASETS['SB']['train_data'].u.shape if hasattr(ALL_DATASETS['SB']['train_data'], 'u') else 'N/A'}")

# --- CED ---
print("Loading CED dataset...")
train_vals_ced_raw, tests_ced_raw = nlb.CED()
cut_ced = len(train_vals_ced_raw[0]) // 2

trains_ced_list = []
vals_ced_list = []
for dat in train_vals_ced_raw:
    trains_ced_list.append(dat[:cut_ced])
    vals_ced_list.append(dat[cut_ced:])
opts_ced = {
    'rmse_scaling': 1.0
}
ALL_DATASETS["CED"] = {
    "train_data": trains_ced_list,
    "val_data": vals_ced_list,
    "test_data": list(tests_ced_raw),
    "opts": opts_ced  # merge dataset opts
}
print(f"CED dataset loaded. Train Data Type (first item): {type(ALL_DATASETS['CED']['train_data'][0])}")
print(f"CED train_data[0].u shape: {ALL_DATASETS['CED']['train_data'][0].u.shape if hasattr(ALL_DATASETS['CED']['train_data'][0], 'u') else 'N/A'}")

# --- WH ---
print("Loading WH dataset...")
train_val_wh_raw, test_wh_raw = nlb.WienerHammerBenchMark()
cut_wh = len(train_val_wh_raw) // 2
opts_wh = {
    'rmse_scaling': 1000.0
}
ALL_DATASETS["WH"] = {
    "train_data": train_val_wh_raw[:cut_wh],
    "val_data": train_val_wh_raw[cut_wh:],
    "test_data": [test_wh_raw],
    "opts": opts_wh  # merge dataset opts
}
print(f"WH dataset loaded. Train Data Type: {type(ALL_DATASETS['WH']['train_data'])}")
print(f"WH train_data.u shape: {ALL_DATASETS['WH']['train_data'].u.shape if hasattr(ALL_DATASETS['WH']['train_data'], 'u') else 'N/A'}")

MODEL_PARAMETERS = {
    "LSTM": {
        "input_size": 1,
        "output_size": 1,
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "dropout": 0.1,
        "epoch_frequency": 3000,
        "lambda_reg": 1e-5,
        "max_norm": 3,
        "alpha": 0.95,  # alpha is set by ALPHA in run_benchmark_multi.py
        "early_stop_patience": 500,
        "rmse_scaling": None  # kept; the real value comes from the dataset opts
    }
}

if __name__ == '__main__':
    current_dataset_name = "SB"

    if current_dataset_name not in ALL_DATASETS:
        raise ValueError(f"Dataset '{current_dataset_name}' not found in ALL_DATASETS.")

    selected_dataset = ALL_DATASETS[current_dataset_name]

    # merge the dataset opts into the model params
    current_model_params = MODEL_PARAMETERS["LSTM"].copy()
    current_model_params.update(selected_dataset["opts"])  # merge dataset opts into the LSTM params

    data = {
        "train_input": selected_dataset["train_data"],
        "train_output": selected_dataset["train_data"],
        "val_input": selected_dataset["val_data"],
        "val_output": selected_dataset["val_data"],
        "test_input": selected_dataset["test_data"],
        "test_output": selected_dataset["test_data"],
    }

    model_type = "LSTM"
    parameter = {model_type: current_model_params}  # params now include the dataset opts

    print(f"\n--- Initializing {model_type} trainer for {current_dataset_name} ---")
    lstm_trainer = train_LSTM_models(data, parameter, current_dataset_name=current_dataset_name)

    print(f"\n--- Training {model_type} model for {current_dataset_name} ---")
    train_result = lstm_trainer.main()

    print(f"\n--- Testing {model_type} model for {current_dataset_name} (each test set individually) ---")
    individual_test_results_list = lstm_trainer.test()

    print(f"\nExperiment for dataset {current_dataset_name} completed.")
