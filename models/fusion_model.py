import torch
import torch.nn as nn


class Fusion_Model(nn.Module):
    """
    Combine Chemception and MLP
    """

    def __init__(self,
                 trained_chemception,
                 trained_mlp,
                 emb_chemception_section=-1,
                 emb_mlp_layer=-1,
                 fusion='concat'):

        super(Fusion_Model, self).__init__()

        # Fusion
        self.chemception = trained_chemception
        self.mlp_decpt = trained_mlp
        self.emb_chemception_section = emb_chemception_section
        self.emb_mlp_layer = emb_mlp_layer
        self.fusion = fusion
        self.fusion_dict = {'emb_chemception_section_num': emb_chemception_section,
                            'emb_mlp_layer_num': emb_mlp_layer}

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

        if (chem_emb_neurons == decpt_emb_neurons) & (self.fusion == 'avg'):
            combined_emb = (chem_emb + decpt_emb) / 2
        else:
            # default is concatenation
            self.fusion = 'concat'
            combined_emb = torch.cat((chem_emb, decpt_emb), 1)

        self.fusion_dict['fusion'] = self.fusion
        fusion_shape = combined_emb.shape[1]

        return combined_emb, fusion_shape
