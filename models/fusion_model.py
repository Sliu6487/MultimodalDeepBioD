import torch
import torch.nn as nn
import warnings


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
        alpha = torch.rand(1).to(self.device)
        self.alpha = nn.Parameter(alpha, requires_grad=True)

    def get_chemception_embedding(self, x, emb_chemception_section):
        if emb_chemception_section == -2:
            # The results of fc_layers embedding is the same as
            # avg pool last inception layer
            emb_chemception_section = -3

        n_blocks = len(self.chemception.inception_blocks)
        self.fusion_dict['n_inception_blocks'] = n_blocks

        if emb_chemception_section == -1:
            x = self.chemception(x)
            self.fusion_dict['emb_chemception_section'] = 'prediction layer'

        # elif emb_chemception_section == -2:
        #     x = self.chemception.features(x)
        #     x = x.view(x.size(0), -1)
        #     self.fusion_dict['emb_chemception_section'] = 'fully connect layer'

        # blocks
        elif emb_chemception_section <= -3:
            # print(n_blocks)
            emb_inception_block = emb_chemception_section + 2
            # print("emb_inception_block:", emb_inception_block)
            if emb_inception_block < -n_blocks - 2:
                raise ValueError("Not enough Chemception blocks!")
            if emb_inception_block >= -n_blocks - 1:
                x = self.chemception.stem(x)
                # go through inception blocks
                if emb_inception_block >= -n_blocks:
                    self.fusion_dict['emb_chemception_section'] = f'Avg pool inception {emb_inception_block}'
                    for i in range(-n_blocks, emb_inception_block + 1):
                        # print("block:", i)
                        transform = self.chemception.inception_blocks[i]
                        x = transform(x)
                else:
                    self.fusion_dict['emb_chemception_section'] = 'Avg pool stem'
            else:
                self.fusion_dict['emb_chemception_section'] = 'Avg pool input image'
            # avg_pool then stretch to fully connect layer
            kernel_size = (x.shape[-1], x.shape[-2])
            avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=(1, 1))
            x = avg_pool(x)
            x = x.view(x.size(0), -1)

        neurons = x.shape[1]

        return x, neurons

    def get_mlp_embedding(self, x, emb_mlp_layer):

        n_layers = len(self.mlp_decpt.hidden)
        self.fusion_dict['n_mlp_layers'] = n_layers

        if emb_mlp_layer == -1:
            x = self.mlp_decpt(x)
            self.fusion_dict['emb_mlp_layer'] = 'prediction layer'

        elif emb_mlp_layer < -1:

            if emb_mlp_layer < - n_layers - 1:
                raise ValueError('Not enough MLP layers!')
            # use descriptor directly if MLP is not needed
            if emb_mlp_layer >= - n_layers:
                self.fusion_dict['emb_mlp_layer'] = f'hidden layer {emb_mlp_layer + 1}'
                for transform in self.mlp_decpt.hidden[:emb_mlp_layer + 1]:
                    # print("transform layer:", transform)
                    x = transform(x)
            else:
                self.fusion_dict['emb_mlp_layer'] = 'No embedding'

        neurons = x.shape[1]

        return x, neurons

    def forward(self, x, y):
        chem_emb, chem_emb_neurons = self.get_chemception_embedding(x, self.emb_chemception_section)
        decpt_emb, decpt_emb_neurons = self.get_mlp_embedding(y, self.emb_mlp_layer)

        if self.fusion == 'no_harm':
            combined_emb = (1 - self.alpha) * chem_emb + self.alpha * decpt_emb

        elif self.fusion == 'sum':
            combined_emb = chem_emb + decpt_emb

        elif self.fusion == 'avg':
            if (chem_emb_neurons == decpt_emb_neurons):
                combined_emb = (chem_emb + decpt_emb) / 2
            else:
                warnings.warn("Mismatching shape, can't Average. Return None.")
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
