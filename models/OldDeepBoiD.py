# Deprecated

# from models.chemception_models import Chemception, Chemception_Small
# from models.dnn_models import MLP_DNN


# class DeepBioD(nn.Module):
#     """
#     Combine chemception and MLP
#     """
#
#     def __init__(self, descp_input_shape=41, num_hidden_layers=3,
#                  num_neurons=128, first_dnn_drop_out_rate=0.5,
#                  chemception_name='Chemception_Small',
#                  img_spec: str = "engd",
#                  base_filters: int = 16,
#                  augment: bool = False,
#                  last_layers=None,
#                  last_dnn_drop_out_rate=0.5):
#
#         super(DeepBioD, self).__init__()
#
#         if last_layers is None:
#             last_layers = [2, 128, 64, 32, 1]
#         self.descp_input_shape = descp_input_shape
#         self.num_hidden_layers = num_hidden_layers
#         self.num_neurons = num_neurons
#         self.first_dnn_drop_out_rate = first_dnn_drop_out_rate
#
#         self.chemception_name = chemception_name
#         self.img_spec = img_spec
#         self.base_filters = base_filters
#         self.augment = augment
#
#         self.last_layers = last_layers
#         self.last_dnn_drop_out_rate = last_dnn_drop_out_rate
#
#         self.mlp_decpt = MLP_DNN(input_shape=self.descp_input_shape,
#                                  num_hidden_layers=self.num_hidden_layers,
#                                  num_neurons=self.num_neurons,
#                                  drop_out_rate=self.first_dnn_drop_out_rate)
#         if self.chemception_name == 'Chemception_Small':
#             self.chemception = Chemception_Small(img_spec=self.img_spec,
#                                                  base_filters=self.base_filters,
#                                                  augment=self.augment)
#         elif self.chemception_name == 'Chemception':
#             self.chemception = Chemception(img_spec=self.img_spec,
#                                            base_filters=self.base_filters,
#                                            augment=self.augment)
#
#         self.clf_dnn = MLP_DNN(layers=self.last_layers,
#                                drop_out_rate=self.last_dnn_drop_out_rate)
#
#     def forward(self, x, y):
#         emb_chem = self.chemception(x)
#         emb_decpt = self.mlp_decpt(y)
#         combined = torch.cat((emb_chem, emb_decpt), 1)
#         output = self.clf_dnn(combined)
#         return output

# user case
# model3 = DeepBioD(emb_chemception_section=self.config['emb_chemception_section'],
#                   emb_mlp_layer=self.config['emb_mlp_layer'],
#                   num_neurons=self.config['n_neurons'],
#                   num_hidden_layers=self.config['n_hidden_layers'],
#                   first_dnn_drop_out_rate=self.config['drop_out_rate'],
#                   n_inception_blocks=self.config['n_inception_blocks'],
#                   last_layers=self.config['last_layers'],
#                   last_dnn_drop_out_rate=self.config['last_dnn_drop_out_rate'])
