import math

import torch

from helpers.data_helpers import create_data_loader
from helpers.model_helpers import copy_freeze_parameters
from models.chemception_models import Chemception, Chemception_Small
from models.deepBoid import DeepBioD
from models.dnn_models import MLP_DNN
from train.training_functions import train_model, train_multi_model


class train_model_multi:
    def __init__(self, config, device, datasets):
        """
        datasets = {'X_tr_tuple': (X_tr_img, X_tr_tbl),
                    'X_val_tuple': (X_val_img, X_val_tbl),
                    'y_tr': y_tr,
                    'y_val': y_val}
        """

        self.device = device
        self.config = config
        self.datasets = datasets

        self.data_loaders = {}
        self.trained_models = {'model_trained1': None,
                               'model_trained2': None,
                               'model_trained3': None}
        self.train_history = {}

    def get_model(self, model_number):
        model = None
        if model_number == 3:
            if self.trained_models['model_trained1'] is None:
                raise ValueError('Train model1 first!')
            elif self.trained_models['model_trained2'] is None:
                raise ValueError('Train model2 first!')

            model3 = DeepBioD(chemception_name=self.config['chemception_name'],
                              num_neurons=self.config['n_neurons'],
                              num_hidden_layers=self.config['n_hidden_layers'],
                              first_dnn_drop_out_rate=self.config['drop_out_rate'],
                              last_layers=self.config['last_layers'],
                              last_dnn_drop_out_rate=self.config['last_dnn_drop_out_rate'])

            # todo: be able to customize trainable layer
            model = copy_freeze_parameters(self.trained_models['model_trained1'],
                                           self.trained_models['model_trained2'],
                                           model3=model3,
                                           emb_section=self.config['emb_section'],
                                           emb_layer=self.config['emb_layer'])
        elif model_number == 1:
            if self.config['chemception_name'] == 'Chemception_Small':
                model = Chemception_Small()
            elif self.config['chemception_name'] == 'Chemception':
                model = Chemception()

        elif model_number == 2:
            model = MLP_DNN(num_neurons=self.config['n_neurons'],
                            num_hidden_layers=self.config['n_hidden_layers'],
                            drop_out_rate=self.config['drop_out_rate'])

        return model.to(self.device)

    def train(self, model_number, transform, epochs=None):
        if epochs:
            # provide an option to pass epochs from outside
            self.config['epochs'] = epochs

        train_loader, val_loader = create_data_loader(model_number=model_number,
                                                      transform=transform,
                                                      y_train=self.datasets['y_tr'],
                                                      y_val=self.datasets['y_val'],
                                                      X_train_tuple=self.datasets['X_tr_tuple'],
                                                      X_val_tuple=self.datasets['X_val_tuple'],
                                                      batch_size=self.config['batch_size'])
        # check data loader just in case
        self.data_loaders[f'train_loader_mode{model_number}'] = train_loader
        self.data_loaders[f'val_loader_mode{model_number}'] = val_loader

        criterion = torch.nn.BCELoss()
        model = self.get_model(model_number=model_number)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           model.parameters()),
                                    lr=self.config['learning_rate'])

        if self.config['use_lr_scheduler']:
            # todo: better way to get length
            if model_number != 3:
                train_len = len(train_loader.dataset)
                val_len = len(train_loader.dataset)
                # train_loader.dataset.tensors[1].shape[0]
                # val_loader.dataset.tensors[1].shape[0]

            else:
                # tensors[0] is dataset mode 1: (x_train, y_train).
                train_len = len(train_loader.dataset)
                val_len = len(val_loader.dataset)

            print(f'model{model_number}, train_len: {train_len}, val_len: {val_len}')

            if self.config['step_size_up'] is None:
                default_step_size_up = math.ceil(train_len / self.config['batch_size']) * 4
                self.config['step_size_up'] = default_step_size_up
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          base_lr=self.config['base_lr'],
                                                          max_lr=self.config['max_lr'],
                                                          step_size_up=self.config['step_size_up'],
                                                          mode=self.config['lr_scheduler_mode'])
        else:
            scheduler = None

        print(f"Training model {model_number}...")

        arguments = [self.config['epochs'], scheduler, train_loader, val_loader,
                     model, optimizer, criterion, self.device]
        if model_number != 3:
            results = train_model(*arguments)
        else:
            results = train_multi_model(*arguments)

        self.trained_models[f'model_trained{model_number}'] = results[0]
        self.train_history[f'model{model_number}_history'] = results[1]

        return None
