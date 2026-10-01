"""
Core training script for the STEP staircase-response data.
Defines SpectralLoss (time-domain MSE + frequency-domain L1) and
train_LSTM_models, the trainer reused by all the experiment scripts
in this folder: data loading, MinMax normalization, the training
loop with early stopping, and free-run testing on the test set.
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
from Enhanced_LSTM import LSTMmodel


class SpectralLoss(nn.Module):
    """
    Time-domain MSE + frequency-domain L1 joint loss.
    The frequency term is scaled by 1/sqrt(T) (Parseval energy conservation),
    so both terms share the same scale and alpha stays meaningful across sequence lengths.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # ---- time loss: MSE over T ----
        mse_loss = F.mse_loss(pred, target)

        # ---- freq loss: rfft scaled by 1/sqrt(T) (Parseval energy normalization) ----
        T = pred.size(1)
        pred_fft = torch.fft.rfft(pred, dim=1) / np.sqrt(T)
        target_fft = torch.fft.rfft(target, dim=1) / np.sqrt(T)

        # L1 on real + imaginary parts (complex-spectrum L1)
        freq_loss = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(pred_fft.imag, target_fft.imag)

        return (1 - self.alpha) * mse_loss + self.alpha * freq_loss


def prepare_data(inputs, outputs, input_scaler=None, output_scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # reshape 1-D inputs to (n_samples, 1)
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)
    # reshape 1-D outputs to (n_samples, 1)
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    # training mode
    if input_scaler is None:
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        inputs_normalized = input_scaler.fit_transform(inputs)

        output_scaler = MinMaxScaler(feature_range=(0, 1))
        outputs_normalized = output_scaler.fit_transform(outputs)
    # test mode
    else:
        inputs_normalized = input_scaler.transform(inputs)
        outputs_normalized = output_scaler.transform(outputs)
    # to tensors and add the batch dim
    inputs_tensor = torch.FloatTensor(inputs_normalized).unsqueeze(0).to(device)
    outputs_tensor = torch.FloatTensor(outputs_normalized).unsqueeze(0).to(device)
    return inputs_tensor, outputs_tensor, input_scaler, output_scaler


class train_LSTM_models:
    def __init__(self, data, parameter):

        self.data = data
        self.parameter = parameter
        # generic model name from parameter['LSTM']['model_name'],
        # defaults to EnhancedLSTM (controls the weights/best_*.pt filenames;
        # final result names are set by each experiment script)
        self.model_name = self.parameter['LSTM'].get('model_name', 'EnhancedLSTM')

        self.criterion = SpectralLoss(alpha=self.parameter['LSTM']['alpha'])

        self.split_train_val()
        self.scaler_path = {
            'input': os.path.abspath('input_scaler.pkl'),
            'output': os.path.abspath('output_scaler.pkl')
        }
        self.train_history = {'train_loss': [], 'val_loss': [], 'lr': []}   # train history: train/val loss and lr

    def split_train_val(self, test_size=0.5):
        """Set up train/test/val arrays."""
        n_samples = len(self.data['train_input'])
        split_idx = int(n_samples * (1 - test_size))
        
        self.train_input = self.data['train_input']
        if self.train_input.ndim == 1:
            self.train_input = self.train_input.reshape(-1, 1)
        self.train_output = self.data['train_output']
        if self.train_output.ndim == 1:
            self.train_output = self.train_output.reshape(-1, 1)
        self.test_input = self.data['test_input']
        if self.test_input.ndim == 1:
            self.test_input = self.test_input.reshape(-1, 1)
        self.test_output = self.data['test_output']
        if self.test_output.ndim == 1:
            self.test_output = self.test_output.reshape(-1, 1)
        # use the independent validation set
        self.val_input = self.data['validata_input']
        self.val_output = self.data['validata_output']
        print(f'train_input_shape:{self.train_input.shape}')
        print(f'test_input_shape:{self.test_input.shape}')
        print(f'val_input_shape:{self.val_input.shape}')
        print("train input mean |u|:", np.abs(self.train_input).mean().item())
        print("test input mean |u|:", np.abs(self.test_input).mean().item())
        print("val input mean |u|:", np.abs(self.val_input).mean().item())

    def save_scalers(self, input_scaler, output_scaler):
        """Save the scalers."""
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        joblib.dump(input_scaler, self.scaler_path['input'])
        joblib.dump(output_scaler, self.scaler_path['output'])
        print(f"Scalers saved to: {self.scaler_path}")

    def main(self):
        # set random seed from the parameter (no seed if None)
        seed = self.parameter['LSTM'].get('seed', None)
        if seed is not None:
            import random as _random
            _random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"[Seed] fixed random seed: seed={seed}")
        # prepare normalized training tensors
        inputs_tensor, outputs_tensor, input_scaler, output_scaler = prepare_data(self.train_input, self.train_output)

        # save the scalers
        self.save_scalers(input_scaler, output_scaler)

        # validation data with the training scalers
        loaded_input_scaler = joblib.load('input_scaler.pkl')
        loaded_output_scaler = joblib.load('output_scaler.pkl')

        val_inputs, val_outputs, _, _ = prepare_data(self.val_input, self.val_output,
                                                     input_scaler=loaded_input_scaler,
                                                     output_scaler=loaded_output_scaler)
        print("Using the full test set as validation")
        print(f"train mean: {inputs_tensor.mean():.2f}, val mean: {val_inputs.mean():.2f}")
        print(f"train var: {inputs_tensor.var():.2f}, val var: {val_inputs.var():.2f}")
        print("val_output", val_outputs.shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_size = self.train_input.shape[1]
        output_size = self.train_output.shape[1]

        model = LSTMmodel(
            input_size=input_size,
            hidden_size=self.parameter['LSTM']['hidden_size'],
            num_layers=self.parameter['LSTM']['num_layers'],
            output_size=output_size,
            dropout=self.parameter['LSTM']['dropout'],
            forget_bias=self.parameter['LSTM'].get('forget_bias', 1.0)
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.parameter['LSTM']['learning_rate'],
            weight_decay=self.parameter['LSTM']['lambda_reg']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                               min_lr=1e-6,)  # scheduler on val loss

        print('---------------- training (train data) ---------------------')
        best_loss = float('inf')
        best_model_state = None
        patience = self.parameter['LSTM']['early_stop_patience']
        no_improvement_count = 0
        start_time = time.time()
        for epoch in range(self.parameter['LSTM']['epoch_frequency']):
            model.train()
            optimizer.zero_grad()

            outputs = model(inputs_tensor)  # [1,7000,2]
            # main loss (prediction error)
            main_loss = self.criterion(outputs, outputs_tensor)
            loss = main_loss
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.parameter['LSTM']['max_norm'])
            optimizer.step()
            current_loss = loss.item()
            self.train_history['train_loss'].append(current_loss)
            # validation
            model.eval()
            with torch.no_grad():
                val_outputs_prediction = model(val_inputs)
                main_loss = self.criterion(val_outputs_prediction, val_outputs).item()
                val_loss = main_loss
                self.train_history['val_loss'].append(val_loss)
            # step the scheduler
            scheduler.step(val_loss)  
            current_lr = optimizer.param_groups[0]['lr']
            self.train_history['lr'].append(current_lr)
            # print train/val loss
            print(f'Epoch {epoch + 1}, Train Loss (MSE): {current_loss:.6f}, Val Loss (MSE): {val_loss:.6f}')

            # early stopping
            if epoch == 0:  # first epoch is the initial best
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
            train_predictions = train_predictions.squeeze().cpu().numpy()
            train_labels = outputs_tensor.squeeze().cpu().numpy()

        model_save_dir = "weights"  # model save dir
        os.makedirs(model_save_dir, exist_ok=True)  

        # save the best model
        best_model_path = os.path.join(model_save_dir, f"best_{self.model_name}.pt")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'train_history': self.train_history
        }, best_model_path)

        # save the final model
        final_model_path = os.path.join(model_save_dir, f"last_{self.model_name}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': final_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
            'train_history': self.train_history
        }, final_model_path)
        return {
            'train_predictions': train_predictions,
            'train_labels': train_labels,
            'history': self.train_history,
            'best_epoch': best_epoch,
            'Model': model
        }

    def test(self, test_input, test_output, model_path=None, use_best_model=True):
        print('---------------- testing model (test data) ---------------------')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load the data preprocessors
        input_scaler = joblib.load('input_scaler.pkl')
        output_scaler = joblib.load('output_scaler.pkl')

        # prepare test data
        inputs_tensor, outputs_tensor, _, _ = prepare_data(
            test_input,
            test_output,
            input_scaler=input_scaler,
            output_scaler=output_scaler
        )
        true_labels = outputs_tensor.squeeze().cpu().numpy()

        # model loading logic
        if model_path is None:
            model_path = os.path.join("weights",
                                      f"best_{self.model_name}.pt" if use_best_model else f"last_{self.model_name}.pt")

        input_size_for_model = inputs_tensor.shape[2]
        output_size_for_model = outputs_tensor.shape[2]
        # load the saved checkpoint
        checkpoint = torch.load(model_path)
        model = LSTMmodel(
            input_size=input_size_for_model,
            hidden_size=self.parameter['LSTM']['hidden_size'],
            num_layers=self.parameter['LSTM']['num_layers'],
            output_size=output_size_for_model,
            dropout=self.parameter['LSTM']['dropout'],
            forget_bias=self.parameter['LSTM'].get('forget_bias', 1.0)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # predict
        with torch.no_grad():
            predictions = model(inputs_tensor).squeeze().cpu().numpy()

        return predictions, true_labels

