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
        self.chemception = copy.deepcopy(trained_chemception)
        self.mlp_decpt = copy.deepcopy(trained_mlp)

        self.emb_chemception_section = emb_chemception_section
        self.emb_mlp_layer = emb_mlp_layer
        self.fusion = fusion
        self.device = device
        self.fusion_dict = {'emb_chemception_section_num': emb_chemception_section,
                            'emb_mlp_layer_num': emb_mlp_layer}

        # no harm factor
        self.nh_factor = nn.Parameter(torch.rand(1).to(self.device),
                                      requires_grad=True)

        self.nh_factor_mtrx = nn.Parameter(torch.rand(1, 126).to(self.device),
                                           requires_grad=True)

        self.shrink_layer1 = nn.Linear(2, 1).to(self.device)
        self.shrink_layer2 = nn.Linear(252, 126).to(self.device)

    def forward(self, x, y):
        chem_emb = self.chemception(x, self.emb_chemception_section)
        decpt_emb = self.mlp_decpt(y, self.emb_mlp_layer)

        decpt_emb_neurons = decpt_emb.shape[1]
        chem_emb_neurons = chem_emb.shape[1]

        self.chem_emb = chem_emb
        self.decpt_emb = decpt_emb

        if self.fusion == 'no_harm':
            if chem_emb_neurons == decpt_emb_neurons:
                combined_emb = (1 - self.nh_factor) * chem_emb + self.nh_factor * decpt_emb
                # print("combined_emb:", combined_emb.shape)
            else:
                return None, None

        elif self.fusion == 'no_harm_matrix':
            if chem_emb_neurons == decpt_emb_neurons:
                if decpt_emb_neurons == 1:
                    combined_emb = (1 - self.nh_factor_mtrx) * chem_emb + self.nh_factor_mtrx * decpt_emb
                else:
                    combined_emb = (1 - self.nh_factor) * chem_emb + self.nh_factor * decpt_emb

                # print("combined_emb:", combined_emb.shape)
            else:
                return None, None

        elif self.fusion == 'no_model1':
            combined_emb = decpt_emb
            # print("combined_emb:", combined_emb.shape)

        elif self.fusion == 'no_model2':
            combined_emb = chem_emb

        elif self.fusion == 'shrink':
            # print("chem_emb:", chem_emb.shape)
            # print("decpt_emb:", decpt_emb.shape)
            combined_emb = torch.cat((chem_emb, decpt_emb), 1)
            # print('combined_emb:', combined_emb.shape)
            if decpt_emb_neurons == 126:
                combined_emb = torch.relu(self.shrink_layer2(combined_emb))
            elif decpt_emb_neurons == 1:
                combined_emb = torch.relu(self.shrink_layer1(combined_emb))
            else:
                return None, None

        elif (self.fusion == 'avg'):
            if (chem_emb_neurons == decpt_emb_neurons):
                combined_emb = (chem_emb + decpt_emb) / 2
            else:
                return None, None

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

        elif self.fusion == 'concat':
            combined_emb = torch.cat((chem_emb, decpt_emb), 1)
        else:
            raise ValueError(f"No such fusion option as {self.fusion}")

        self.fusion_dict['fusion'] = self.fusion
        self.fusion_dict['fusion_neurons'] = combined_emb.shape[1]
        fusion_shape = combined_emb.shape[1]

        return combined_emb, fusion_shape
