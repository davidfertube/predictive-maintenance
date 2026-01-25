import torch
import torch.nn as nn

class LSTM_RUL_Predictor(nn.Module):
    def __init__(self, input_size=21, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # input_seq: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(input_seq)
        
        # Take only the last time step output
        last_time_step = lstm_out[:, -1, :]
        
        predictions = self.linear(last_time_step)
        return predictions