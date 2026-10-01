"""
Core training script for the benchmark datasets (CT / SB / WH).
Defines the LSTM-family trainer used everywhere in this folder:
train_LSTM_models handles data prep, normalization, the training loop
(with early stopping) and testing on the per-dataset test sets.
Also defines SpectralLoss - the time-domain MSE + frequency-domain L1
loss shared with the main method.
"""
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import torch.optim
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import time
import torch.nn.functional as F
from Nonlinea_Benchmark_Predict.BaselineLSTM import LSTMmodel as _BaseLSTM
from Nonlinea_Benchmark_Predict.BaselineGRU import LSTMmodel as _BaseGRU
from Nonlinea_Benchmark_Predict.BaselineRNN import LSTMmodel as _BaseRNN
from Nonlinea_Benchmark_Predict.Enhanced_LSTM import LSTMmodel as _EnhancedLSTM
_MODEL_MAP = {"LSTM": _BaseLSTM, "GRU": _BaseGRU, "RNN": _BaseRNN, "EnhancedLSTM": _EnhancedLSTM}
LSTMmodel = _BaseGRU  # default model (keeps the original behavior)


class SpectralLoss(nn.Module):
    """
    Time-domain MSE + frequency-domain L1 joint loss.
    Adding the frequency loss improves both training and test results vs plain MSE.
    The frequency term is scaled by 1/sqrt(T) (Parseval energy conservation),
    so alpha stays meaningful across sequence lengths (same as Enhanced LSTM).
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # time loss
        mse_loss = F.mse_loss(pred, target)
        # freq loss: rfft scaled by 1/sqrt(T) (Parseval energy normalization)
        T = pred.size(1)
        pred_fft = torch.fft.rfft(pred, dim=1) / np.sqrt(T)
        target_fft = torch.fft.rfft(target, dim=1) / np.sqrt(T)
        # L1 on real + imaginary parts
        freq_loss = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(pred_fft.imag, target_fft.imag)
        return (1 - self.alpha) * mse_loss + self.alpha * freq_loss


def _concat_and_reshape(data_list, attr_name):
    """
    Concatenate the requested attribute (u or y) of Input_output_data objects and make it 2-D.
    data_list may be a single object or a list of objects.
    """
    if not isinstance(data_list, list):
        data_list = [data_list]  # wrap into a list for uniform handling

    arrays = []

    for item in data_list:
        # skip items without the requested attribute
        if hasattr(item, attr_name):
            arr = getattr(item, attr_name)
            if arr.size == 0:  # skip empty arrays
                continue
            if arr.ndim == 1:
                arrays.append(arr.reshape(-1, 1))
            else:
                arrays.append(arr)

    if len(arrays) == 0:
        return np.array([])  
    elif len(arrays) == 1:
        return arrays[0]
    else:
        concatenated_array = np.concatenate(arrays, axis=0)
        return concatenated_array


def prepare_data(inputs_raw, outputs_raw, input_scaler=None, output_scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if raw inputs are already numpy arrays,
    # use them directly and skip the object-attribute logic
    if isinstance(inputs_raw, np.ndarray):
        inputs_arr = inputs_raw
    else:
        inputs_arr = _concat_and_reshape(inputs_raw, 'u')
    if isinstance(outputs_raw, np.ndarray):
        outputs_arr = outputs_raw
    else:
        outputs_arr = _concat_and_reshape(outputs_raw, 'y')
    # return None if still empty
    if inputs_arr.size == 0 or outputs_arr.size == 0:
        print("Warning: Input or output data is empty after (or during) preparation.")
        return None, None, input_scaler, output_scaler
    # training mode
    if input_scaler is None:
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        inputs_normalized = input_scaler.fit_transform(inputs_arr)
        output_scaler = MinMaxScaler(feature_range=(0, 1))
        outputs_normalized = output_scaler.fit_transform(outputs_arr)
    # test mode
    else:
        inputs_normalized = input_scaler.transform(inputs_arr)
        outputs_normalized = output_scaler.transform(outputs_arr)
    # to tensors and add the batch dim
    inputs_tensor = torch.FloatTensor(inputs_normalized).unsqueeze(0).to(device)
    outputs_tensor = torch.FloatTensor(outputs_normalized).unsqueeze(0).to(device)
    return inputs_tensor, outputs_tensor, input_scaler, output_scaler


class train_LSTM_models:
    def __init__(self, data, parameter, current_dataset_name="Unknown"):
        self.data = data
        self.parameter = parameter
        self.current_dataset_name = current_dataset_name

        # concat and reshape via the helper
        self.train_input_u = _concat_and_reshape(data['train_input'], 'u')
        self.train_output_y = _concat_and_reshape(data['train_output'], 'y')
        self.val_input_u = _concat_and_reshape(data['val_input'], 'u')
        self.val_output_y = _concat_and_reshape(data['val_output'], 'y')

        # test_input/test_output stay as lists or single objects,
        # because test() iterates over them for per-set predictions
        self.test_input_raw = data['test_input']
        self.test_output_raw = data['test_output']  # Keep original objects for individual test predictions

        # loss='spectral' uses the sqrt(T)-normalized SpectralLoss, otherwise MSE
        loss_name = self.parameter['LSTM'].get('loss', 'mse')
        if loss_name == 'spectral':
            self.criterion = SpectralLoss(alpha=self.parameter['LSTM'].get('alpha', 0.4))
        else:
            self.criterion = nn.MSELoss()
        # model class and save tag (configurable, for multi-model / multi-seed runs)
        self.model_key = self.parameter['LSTM'].get('model', 'GRU')
        self.model_cls = _MODEL_MAP.get(self.model_key, _BaseGRU)
        self.save_tag = self.parameter['LSTM'].get('save_tag', '')

        self.scaler_path = {
            'input': os.path.abspath('input_scaler.pkl'),
            'output': os.path.abspath('output_scaler.pkl')
        }
        self.train_history = {'train_loss': [], 'val_loss': [], 'lr': []}  # train history: train/val loss and lr
        self.rmse_scaling = self.parameter['LSTM'].get('rmse_scaling', 1.0)

        print(f"Initialized with rmse_scaling: {self.rmse_scaling} for dataset {self.current_dataset_name}")

    def split_train_val(self):
        """
        No splitting here; use the train/val data from self.data directly,
        and print the concatenated and per-set shapes.
        """
        print(f"\n--- Benchmark: {self.current_dataset_name} ---")  
        # concatenated shapes
        print(f'Concatenated Train Input u_shape: {self.train_input_u.shape}')
        print(f'Concatenated Train Output y_shape: {self.train_output_y.shape}')
        print(f'Concatenated Validation Input u_shape: {self.val_input_u.shape}')
        print(f'Concatenated Validation Output y_shape: {self.val_output_y.shape}')
        # per training set details
        train_data_list = self.data['train_input'] if isinstance(self.data['train_input'], list) else [
            self.data['train_input']]
        print(f"{len(train_data_list)} training sets:")
        for i, dat in enumerate(train_data_list):
            u_shape = dat.u.shape if hasattr(dat, 'u') else 'N/A'
            y_shape = dat.y.shape if hasattr(dat, 'y') else 'N/A'
            print(f"  Training set {i}: Input shape: {u_shape}, Output shape: {y_shape}")
        # per validation set details
        val_data_list = self.data['val_input'] if isinstance(self.data['val_input'], list) else [self.data['val_input']]
        print(f"{len(val_data_list)} validation sets:")
        for i, dat in enumerate(val_data_list):
            u_shape = dat.u.shape if hasattr(dat, 'u') else 'N/A'
            y_shape = dat.y.shape if hasattr(dat, 'y') else 'N/A'
            print(f"  Validation set {i}: Input shape: {u_shape}, Output shape: {y_shape}")
        # per testing set details
        test_data_list = self.data['test_input'] if isinstance(self.data['test_input'], list) else [
            self.data['test_input']]
        print(f"{len(test_data_list)} testing sets:")
        for i, dat in enumerate(test_data_list):
            u_shape = dat.u.shape if hasattr(dat, 'u') else 'N/A'
            y_shape = dat.y.shape if hasattr(dat, 'y') else 'N/A'
            print(f"  Testing set {i}: Input shape: {u_shape}, Output shape: {y_shape}")
        # statistics
        if self.train_input_u.size > 0:
            print("train input mean |u|:", np.abs(self.train_input_u).mean().item())
        if self.val_input_u.size > 0:
            print("val input mean |u|:", np.abs(self.val_input_u).mean().item())
        # test-step amplitude statistics (first test set)
        if test_data_list and hasattr(test_data_list[0], 'u') and test_data_list[0].u.size > 0:
            print("test input mean |u| (first test set):", np.abs(test_data_list[0].u).mean().item())
        else:  # fall back to all test sets concatenated
            all_test_u = _concat_and_reshape(test_data_list, 'u')
            if all_test_u.size > 0:
                print("test input mean |u| (concatenated test sets):", np.abs(all_test_u).mean().item())

    def save_scalers(self, input_scaler, output_scaler):
        """Save the scalers."""
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        joblib.dump(input_scaler, self.scaler_path['input'])
        joblib.dump(output_scaler, self.scaler_path['output'])
        print(f"Scalers saved to: {self.scaler_path}")

    def main(self):

        # print data shapes
        self.split_train_val()

        # normalized training tensors (fit on the concatenated data)
        inputs_tensor, outputs_tensor, input_scaler, output_scaler = prepare_data(self.train_input_u,
                                                                                  self.train_output_y)
        if inputs_tensor is None:
            raise ValueError("Training data is empty, cannot proceed with training.")

        # save the scalers
        self.save_scalers(input_scaler, output_scaler)

        # validation data with the training scalers
        loaded_input_scaler = joblib.load('input_scaler.pkl')
        loaded_output_scaler = joblib.load('output_scaler.pkl')

        val_inputs, val_outputs, _, _ = prepare_data(self.val_input_u, self.val_output_y,
                                                     input_scaler=loaded_input_scaler,
                                                     output_scaler=loaded_output_scaler)
        if val_inputs is None:
            print("Warning: Validation data is empty, validation step will be skipped.")
            return

        print("Training on the concatenated train/val sets")
        print(f"train mean: {inputs_tensor.mean():.2f}, val mean: {val_inputs.mean():.2f}")
        print(f"train var: {inputs_tensor.var():.2f}, val var: {val_inputs.var():.2f}")
        print("val_output shape:", val_outputs.shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # input/output sizes from the concatenated feature counts
        input_size = self.train_input_u.shape[1]
        output_size = self.train_output_y.shape[1]

        model = self.model_cls(
            input_size=input_size,
            hidden_size=self.parameter['LSTM']['hidden_size'],
            num_layers=self.parameter['LSTM']['num_layers'],
            output_size=output_size,
            dropout=self.parameter['LSTM']['dropout']
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.parameter['LSTM']['learning_rate'],
            weight_decay=self.parameter['LSTM']['lambda_reg']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                               min_lr=1e-6, )  # scheduler on val loss

        print('---------------- training (train data) ---------------------')
        best_loss = float('inf')
        best_model_state = None
        # early stop patience comes from Option.py
        patience = self.parameter['LSTM']['early_stop_patience']
        no_improvement_count = 0
        start_time = time.time()
        for epoch in range(self.parameter['LSTM']['epoch_frequency']):
            model.train()
            optimizer.zero_grad()

            outputs = model(inputs_tensor)
            main_loss = self.criterion(outputs, outputs_tensor)
            loss = main_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.parameter['LSTM']['max_norm'])
            optimizer.step()
            current_loss = loss.item()
            self.train_history['train_loss'].append(current_loss)

            model.eval()
            with torch.no_grad():
                val_outputs_prediction = model(val_inputs)
                main_loss = self.criterion(val_outputs_prediction, val_outputs).item()  
                val_loss = main_loss
                self.train_history['val_loss'].append(val_loss)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            self.train_history['lr'].append(current_lr)

            print(f'Epoch {epoch + 1}, Train Loss (MSE): {current_loss:.6f}, Val Loss (MSE): {val_loss:.6f}')

            # early stopping
            if epoch == 0:
                prev_best_loss = val_loss
                best_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                print(f"initial model @ epoch 1, val_loss: {best_loss:.6f}")
            else:
                if val_loss < best_loss * 0.99:
                    improvement = (prev_best_loss - val_loss) / (prev_best_loss + 1e-8) * 100
                    prev_best_loss = best_loss
                    best_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    no_improvement_count = 0
                    print(f"new best @ epoch {best_epoch}, val_loss: {best_loss:.6f} (improvement: {improvement:.2f}%)")
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print(f'early stop at epoch {epoch + 1}, best val loss: {best_loss:.6f} (epoch {best_epoch})')
                        break

        end_time = time.time()
        print(f'training time: {end_time - start_time:.2f}s')

        final_model_state = model.state_dict().copy()  # final epoch state

        # train predictions from the final model
        model.load_state_dict(final_model_state)
        model.eval()
        with torch.no_grad():
            train_predictions = model(inputs_tensor)
            # inverse-transform back to the raw scale (raw-domain metrics)
            train_predictions_raw = output_scaler.inverse_transform(
                train_predictions.squeeze().cpu().numpy().reshape(-1, output_size)).squeeze()
            train_labels_raw = output_scaler.inverse_transform(
                outputs_tensor.squeeze().cpu().numpy().reshape(-1, output_size)).squeeze()

        model_save_dir = "weights"  # model save dir
        os.makedirs(model_save_dir, exist_ok=True)  

        # save the best model
        _suffix = f"_{self.save_tag}" if self.save_tag else ""
        best_model_path = os.path.join(model_save_dir, f"best{_suffix}.pt")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'train_history': self.train_history
        }, best_model_path)

        # save the final model
        final_model_path = os.path.join(model_save_dir, f"last{_suffix}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': final_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
            'train_history': self.train_history
        }, final_model_path)
        return {
            'train_predictions': train_predictions_raw,  # raw-scale data
            'train_labels': train_labels_raw,  # raw-scale data
            'history': self.train_history,
            'best_epoch': best_epoch,
            'final_model': model,
            'best_model_state_dict': best_model_state
        }

    def test(self):
        print('---------------- testing model (test data) ---------------------')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_scaler = joblib.load('input_scaler.pkl')
        output_scaler = joblib.load('output_scaler.pkl')

        input_size_for_model = self.train_input_u.shape[1]
        output_size_for_model = self.train_output_y.shape[1]

        best_model = self.model_cls(
            input_size=input_size_for_model,
            hidden_size=self.parameter['LSTM']['hidden_size'],
            num_layers=self.parameter['LSTM']['num_layers'],
            output_size=output_size_for_model,
            dropout=self.parameter['LSTM']['dropout']
        ).to(device)
        final_model = self.model_cls(
            input_size=input_size_for_model,
            hidden_size=self.parameter['LSTM']['hidden_size'],
            num_layers=self.parameter['LSTM']['num_layers'],
            output_size=output_size_for_model,
            dropout=self.parameter['LSTM']['dropout']
        ).to(device)

        _suffix = f"_{self.save_tag}" if self.save_tag else ""
        best_model_path = os.path.join("weights", f"best{_suffix}.pt")
        best_checkpoint = torch.load(best_model_path)
        best_model.load_state_dict(best_checkpoint['model_state_dict'])
        best_model.eval()

        final_model_path = os.path.join("weights", f"last{_suffix}.pt")
        final_checkpoint = torch.load(final_model_path)
        final_model.load_state_dict(final_checkpoint['model_state_dict'])
        final_model.eval()

        all_individual_test_results = []
        test_inputs_list = self.test_input_raw if isinstance(self.test_input_raw, list) else [self.test_input_raw]
        test_outputs_list = self.test_output_raw if isinstance(self.test_output_raw, list) else [self.test_output_raw]

        print(f"Using RMSE scaling factor: {self.rmse_scaling} for dataset {self.current_dataset_name}")

        for i, (single_test_input, single_test_output) in enumerate(zip(test_inputs_list, test_outputs_list)):
            inputs_tensor, outputs_tensor, _, _ = prepare_data(
                single_test_input,
                single_test_output,
                input_scaler=input_scaler,
                output_scaler=output_scaler
            )
            if inputs_tensor is None:
                print(f"Warning: Test data set {i} is empty, skipping.")
                continue

            with torch.backends.cudnn.flags(enabled=False):
                with torch.no_grad():
                    best_preds_i_normalized = best_model(inputs_tensor).squeeze().cpu().numpy()
                    final_preds_i_normalized = final_model(inputs_tensor).squeeze().cpu().numpy()
            true_labels_i_normalized = outputs_tensor.squeeze().cpu().numpy()

            # --- inverse-transform all predictions and labels (no clipping) ---
            # make 2-D for MinMaxScaler.transform
            if best_preds_i_normalized.ndim == 1:
                best_preds_i_normalized_2d = best_preds_i_normalized.reshape(-1, 1)
            else:
                best_preds_i_normalized_2d = best_preds_i_normalized

            if final_preds_i_normalized.ndim == 1:
                final_preds_i_normalized_2d = final_preds_i_normalized.reshape(-1, 1)
            else:
                final_preds_i_normalized_2d = final_preds_i_normalized

            if true_labels_i_normalized.ndim == 1:
                true_labels_i_normalized_2d = true_labels_i_normalized.reshape(-1, 1)
            else:
                true_labels_i_normalized_2d = true_labels_i_normalized

            best_preds_i_raw = output_scaler.inverse_transform(best_preds_i_normalized_2d).squeeze()
            final_preds_i_raw = output_scaler.inverse_transform(final_preds_i_normalized_2d).squeeze()
            true_labels_i_raw = output_scaler.inverse_transform(true_labels_i_normalized_2d).squeeze()

            # raw-scale RMSE over the full sequence
            rmse_best_raw = np.sqrt(np.mean((best_preds_i_raw - true_labels_i_raw) ** 2))
            rmse_final_raw = np.sqrt(np.mean((final_preds_i_raw - true_labels_i_raw) ** 2))

            # apply rmse scaling
            scaled_rmse_best = rmse_best_raw * self.rmse_scaling
            scaled_rmse_final = rmse_final_raw * self.rmse_scaling

            # --- store results ---
            all_individual_test_results.append({
                'test_set_index': i,
                'best_pred': best_preds_i_raw,
                'final_pred': final_preds_i_raw,
                'true_labels': true_labels_i_raw,
                'rmse_best_raw': rmse_best_raw,
                'rmse_final_raw': rmse_final_raw,
                'scaled_rmse_best': scaled_rmse_best,
                'scaled_rmse_final': scaled_rmse_final
            })

            # --- print results ---
            print(f"Test set {i}: ")
            print(f"  (Full Sequence Raw Scale) Best Model RMSE: {rmse_best_raw:.6f}, Final Model RMSE: {rmse_final_raw:.6f}")
            print(f"  (Full Sequence Scaled) Best Model RMSE: {scaled_rmse_best:.6f}, Final Model RMSE: {scaled_rmse_final:.6f}")
            print(f"  Best Pred Shape {best_preds_i_raw.shape}, Final Pred Shape {final_preds_i_raw.shape}, True Label Shape {true_labels_i_raw.shape}")

        return all_individual_test_results
    