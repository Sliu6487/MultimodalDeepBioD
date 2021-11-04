import json

import torch

from helpers.data_helpers import get_data
from helpers.print_and_plot import show_training_plots  # , print_hyper_parameters
from train.train_model_multi import train_model_multi

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

with open('config_files/default_config.json') as f:
    config = json.load(f)

X_tr_img, X_val_img, X_tr_tbl, X_val_tbl, y_tr, y_val = get_data()

datasets = {'X_tr_tuple': (X_tr_img, X_tr_tbl),
            'X_val_tuple': (X_val_img, X_val_tbl),
            'y_tr': y_tr,
            'y_val': y_val}

# print(X_tr_img.shape)

train_multimodal = train_model_multi(config=config, device=device,
                                     datasets=datasets)

# change config value here
train_multimodal.config['epochs'] = 2

epochs = train_multimodal.config['epochs']
rotate = train_multimodal.config['rotate']
early_stop = train_multimodal.config['early_stop']

print()
model_number = 1
train_args = [model_number, epochs, rotate, early_stop]
train_multimodal.train(*train_args)

show_training_plots(model_number=model_number, show_accuracy=True, show_lr=True,
                    train_history_dict=train_multimodal.train_history)

print()
model_number = 2
train_args = [model_number, epochs, rotate, early_stop]
train_multimodal.train(*train_args)

# print_hyper_parameters(train_multimodal.config)
show_training_plots(model_number=model_number, show_accuracy=True, show_lr=True,
                    train_history_dict=train_multimodal.train_history)

print()
model_number = 3
print(f"emb_chemception_section = {train_multimodal.config['emb_chemception_section']}, "
      f"emb_mlp_layer = {train_multimodal.config['emb_mlp_layer']}.")

train_args = [model_number, epochs, rotate, early_stop]
train_multimodal.train(*train_args)

# print_hyper_parameters(train_multimodal.config)
# no accuracy and learning rate tracked
show_training_plots(model_number=model_number, show_accuracy=False, show_lr=False,
                    train_history_dict=train_multimodal.train_history)

print()
train_multimodal.config['emb_chemception_section'] = -2
train_multimodal.config['emb_mlp_layer'] = -2
print(f"emb_chemception_section = {train_multimodal.config['emb_chemception_section']}, "
      f"emb_mlp_layer = {train_multimodal.config['emb_mlp_layer']}.")

train_args = [model_number, epochs, rotate, early_stop]
train_multimodal.train(*train_args)

# print_hyper_parameters(train_multimodal.config)
show_training_plots(model_number=model_number, show_accuracy=False, show_lr=False,
                    train_history_dict=train_multimodal.train_history)

print()
train_multimodal.config['emb_chemception_section'] = -3
train_multimodal.config['emb_mlp_layer'] = -3
print(f"emb_chemception_section = {train_multimodal.config['emb_chemception_section']}, "
      f"emb_mlp_layer = {train_multimodal.config['emb_mlp_layer']}.")

train_args = [model_number, epochs, rotate, early_stop]
train_multimodal.train(*train_args)

# print_hyper_parameters(train_multimodal.config)
show_training_plots(model_number=model_number, show_accuracy=False, show_lr=False,
                    train_history_dict=train_multimodal.train_history)
