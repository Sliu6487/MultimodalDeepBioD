import torch
import torch.nn as nn
import warnings
import copy


class Fusion_Model(nn.Module):
    """
    Combine Chemception and MLP
    """

    def __init__(self,
                 trained_chemception,
                 trained_mlp,
                 emb_chemception_section=-1,
                 emb_mlp_layer=-1,
                 fusion='concat',
                 device='cpu'):

        super(Fusion_Model, self).__init__()

        # Fusion
        self.chemception = trained_chemception
        self.mlp_decpt = trained_mlp
        self.emb_chemception_section = emb_chemception_section
        self.emb_mlp_layer = emb_mlp_layer
        self.fusion = fusion
        self.device = device
        self.fusion_dict = {'emb_chemception_section_num': emb_chemception_section,
                            'emb_mlp_layer_num': emb_mlp_layer}

        # no harm factor
        self.nh_factor = nn.Parameter(torch.rand(1).to(self.device),
                                      requires_grad=True)

    def forward(self, x, y):
        chem_emb = self.chemception(x, self.emb_chemception_section)
        decpt_emb = self.mlp_decpt(y, self.emb_mlp_layer)

        decpt_emb_neurons = decpt_emb.shape[1]
        chem_emb_neurons = chem_emb.shape[1]

        self.chem_emb = chem_emb
        self.decpt_emb = decpt_emb

        if self.fusion == 'no_harm':
            combined_emb = (1 - self.nh_factor) * chem_emb + self.nh_factor * decpt_emb
            # print("combined_emb:", combined_emb.shape)

        elif self.fusion == 'abs_no_harm':
            combined_emb = decpt_emb
            # print("combined_emb:", combined_emb.shape)

        elif self.fusion == 'sum':
            combined_emb = chem_emb + decpt_emb

        elif (self.fusion == 'avg'):
            if (chem_emb_neurons == decpt_emb_neurons):
                combined_emb = (chem_emb + decpt_emb) / 2
            else:
                warnings.warn("Mismatching shape, can't everage. Return None.")
                return None

        elif self.fusion == 'tf':
            chem_emb_h = torch.cat((torch.ones(chem_emb.shape[0], 1).to(self.device), chem_emb), dim=1)
            # print('chem_emb_h:', chem_emb_h.shape) # [batch, neuron1+1]
            decpt_emb_h = torch.cat((torch.ones(decpt_emb.shape[0], 1).to(self.device), decpt_emb), dim=1)
            # print('decpt_emb_h:', decpt_emb_h.shape) # [batch, neuron2+1]
            # outer product
            fusion_tensor = torch.bmm(chem_emb_h.unsqueeze(2), decpt_emb_h.unsqueeze(1))
            # print('fusion_tensor:', fusion_tensor.shape) #[batch, neuron1+1, neuron2+1]

            combined_emb = fusion_tensor.view(fusion_tensor.size(0), -1)
            # print('combined_emb:', combined_emb.shape) #[batch, (neuron1+1) * (neuron2+1)]

        else:
            # default as concat
            combined_emb = torch.cat((chem_emb, decpt_emb), 1)

        self.fusion_dict['fusion_method'] = self.fusion
        self.fusion_dict['fusion_neurons'] = combined_emb.shape[1]
        fusion_shape = combined_emb.shape[1]

        return combined_emb, fusion_shape
