import numpy as np
import torch
import torch.nn as nn
import time

def move_sliding_window(data, window_size, inputs_cols_indices, label_col_index):
    """
    data: numpy array including data
    window_size: size of window
    inputs_cols_indices: col indices to include
    """

    # (# instances created by movement, seq_len (timestamps), # features (input_len))
    inputs = np.zeros((len(data) - window_size, window_size, len(inputs_cols_indices)))
    labels = np.zeros(len(data) - window_size)

    for i in range(window_size, len(data)):
        inputs[i - window_size] = data[i - window_size : i, inputs_cols_indices]
        labels[i - window_size] = data[i, label_col_index]
    inputs = inputs.reshape(-1, window_size, len(inputs_cols_indices))
    labels = labels.reshape(-1, 1)
    print(inputs.shape, labels.shape)

    return inputs, labels



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        )
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )
        return hidden



def sMAPE(outputs, targets):
    sMAPE = (
        100
        / len(targets)
        * np.sum(np.abs(outputs - targets) / (np.abs(outputs + targets)) / 2)
    )
    return sMAPE


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()  # Set model to evaluation mode
    outputs, targets = [], []
    start_time = time.process_time()

    for state in test_x:  # Loop through each state's test data
        # Convert input and target data to tensors
        x = torch.tensor(test_x[state], dtype=torch.float32).to(device)
        y = torch.tensor(test_y[state], dtype=torch.float32)

        # Initialize hidden state
        h = model.init_hidden(x.size(0))

        # Forward pass without gradient computation
        with torch.no_grad():
            pred, _ = model(x, h)

        # Inverse transform to get values in original scale
        y_pred = label_scalers[state].inverse_transform(pred.cpu().numpy()).ravel()
        y_true = label_scalers[state].inverse_transform(y.numpy()).ravel()

        outputs.append(y_pred)
        targets.append(y_true)

    # Concatenate all predictions and targets
    all_preds = np.concatenate(outputs)
    all_truths = np.concatenate(targets)

    # Print evaluation metrics
    print(f"Evaluation Time: {time.process_time() - start_time:.2f} seconds")
    print(f"sMAPE: {round(sMAPE(all_preds, all_truths), 3)}%")

    return outputs, targets, sMAPE(all_preds, all_truths)
