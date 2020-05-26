import torch
from torch import nn

class RNNModel(nn.Module):

    embedding_size = 49

    def __init__(self, embedding_size, cell_type, num_layers=2, bidirectional=False, dropout_prob=0):
        super(RNNModel, self).__init__()
        hidden_size = embedding_size // 2
        self.cell_type = cell_type
        if cell_type == "rnn":
            self.rnn_1 = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_prob, bidirectional=bidirectional)
            self.rnn_2 = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_prob, bidirectional=bidirectional)
        elif cell_type == "gru":
            self.rnn_1 = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_prob, bidirectional=bidirectional)
            self.rnn_2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_prob, bidirectional=bidirectional)
        elif cell_type == "lstm":
            self.rnn_1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_prob, bidirectional=bidirectional)
            self.rnn_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_prob, bidirectional=bidirectional)

        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)

    def forward(self, x):
        if self.cell_type == "lstm":
            output, (h_n, c_n) = self.rnn_1(x)
            _, (h_n, _) = self.rnn_2(output)
            x = self.fc_1(h_n[-1])
            x = torch.relu(x)
            logits = self.fc_2(x)
            return logits.squeeze()
        else:
            output, h_n = self.rnn_1(x) ## inicijalizacija skrivenog stanja?
            _, h_n = self.rnn_2(output)
            x = self.fc_1(h_n[-1])
            x = torch.relu(x)
            logits = self.fc_2(x)
            return logits.squeeze()