import json

import torch

from helpers.data_helpers import get_data
from helpers.print_and_plot import show_training_plots
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
#
train_multimodal.config['epochs'] = 2

model_number = 1
train_multimodal.train(model_number=model_number,
                       transform=train_multimodal.config['transform'],
                       epochs=train_multimodal.config['epochs'])

show_training_plots(model_number=model_number, show_accuracy=True, show_lr=True,
                    train_history_dict=train_multimodal.train_history)

model_number = 2
train_multimodal.train(model_number=model_number, transform=False,
                       epochs=train_multimodal.config['epochs'])

show_training_plots(model_number=model_number, show_accuracy=True, show_lr=True,
                    train_history_dict=train_multimodal.train_history)

model_number = 3
train_multimodal.train(model_number=model_number, transform=True,
                       epochs=train_multimodal.config['epochs'])
# no accuracy and learning rate tracked
show_training_plots(model_number=model_number, show_accuracy=False, show_lr=False,
                    train_history_dict=train_multimodal.train_history)


# from sklearn.model_selection import KFold
# from train.metrics import clf_err_rate
# import numpy as np
#
# with open('config_files/default_config0.json') as f:
#     config = json.load(f)
#
# config["kf"] = KFold(n_splits=5, random_state=41, shuffle=True)
# config["customized_metric"] = clf_err_rate
# print(config['kf'].split(np.zeros(y_tr.shape[0])))
#
# # paper baseline
# from train.cv_train import CV_Train
# import math


# batch_size_global = 32
# train_x_data = X_tr_img
# train_size = int(len(train_x_data) * 0.8)
# cv_train = CV_Train('Chemception_Small')
# avg_aer = cv_train._train(train_x_data, y_tr, num_epochs=150,
#                           learning_rate=0.001, optimizer_name='SGD',
#                           batch_size=batch_size_global,
#                           patience=50, verbose=1,
#                           use_lr_scheduler='batch',
#                           lr_scheduler_mode='triangular',
#                           max_lr=0.03, base_lr=0.0001,
#                           step_size_up=math.ceil(train_size / batch_size_global) * 4)