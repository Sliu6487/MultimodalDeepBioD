import torch
import torch.nn as nn


class MLP_DNN(nn.Module):

    def __init__(self, layers=None, input_shape=41, num_hidden_layers=3,
                 num_neurons=126, drop_out_rate=0.5, **kwargs):
        super(MLP_DNN, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.drop_out_rate = drop_out_rate

        if layers is None:
            layers = [input_shape]
            for i in range(num_hidden_layers):
                layers.append(num_neurons)
            layers.append(1)

        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(layers, layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
                x = torch.dropout(x, p=self.drop_out_rate, train=True)
            else:
                x = torch.sigmoid(linear_transform(x))
        return x


class CLF_DNN(nn.Module):

    def __init__(self,
                 fusion_shape,
                 fusion_model,
                 hidden_layers=None,
                 drop_out_rate=0.5, **kwargs):
        super(CLF_DNN, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.fusion_model = fusion_model
        self.layers = [fusion_shape] + hidden_layers + [1]
        self.drop_out_rate = drop_out_rate

        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(self.layers, self.layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, x, y):
        fusion, _ = self.fusion_model(x, y)
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                fusion = torch.relu(linear_transform(fusion))
                fusion = torch.dropout(fusion, p=self.drop_out_rate, train=True)
            else:
                fusion = torch.sigmoid(linear_transform(fusion))
        return fusion

