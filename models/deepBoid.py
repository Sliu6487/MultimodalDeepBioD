import torch
import torch.nn as nn


class DeepBioD(nn.Module):

    def __init__(self,
                 fusion_shape,
                 fusion_model,
                 hidden_layers=None,
                 drop_out_rate=0.5, **kwargs):
        super(DeepBioD, self).__init__()

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
