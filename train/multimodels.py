import math
import copy

import torch

from helpers.data_helpers import create_data_loader
from helpers.model_helpers import freeze_layers
from models.chemception_models import Chemception
from models.deepBoid import DeepBioD
from models.dnn_models import MLP_DNN
from models.fusion_model import Fusion_Model
from train.training_functions import train_model, train_multi_model


class MultiModels:
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

        if model_number == 1:
            model = Chemception(n_inception_blocks=self.config['n_inception_blocks'])

        elif model_number == 2:
            model = MLP_DNN(num_neurons=self.config['n_neurons'],
                            num_hidden_layers=self.config['n_hidden_layers'],
                            drop_out_rate=self.config['drop_out_rate'])

        elif model_number == 3:
            if self.trained_models['model_trained1'] is None:
                raise ValueError('Train model1 first!')
            if self.trained_models['model_trained2'] is None:
                raise ValueError('Train model2 first!')

            model_trained1 = copy.deepcopy(self.trained_models['model_trained1'])
            model_trained2 = copy.deepcopy(self.trained_models['model_trained2'])

            fusion_model = Fusion_Model(trained_chemception=model_trained1,
                                        trained_mlp=model_trained2,
                                        emb_chemception_section=self.config['emb_chemception_section'],
                                        emb_mlp_layer=self.config['emb_mlp_layer'],
                                        fusion=self.config['fusion'],
                                        device=self.device)

            # get fusion shape to design the layers of last classifier
            # make input data's device the device of fusion_model
            # fusion_model's weights device is the device of 2 trained models inside
            test_data_img = self.datasets['X_tr_tuple'][0][:2].to(self.device)
            test_data_tbl = self.datasets['X_tr_tuple'][1][:2].to(self.device)

            fusion_model.eval()
            _, fusion_shape = fusion_model(test_data_img, test_data_tbl)
            if fusion_shape is None:
                return None

            model_trained1.eval()
            y_1 = model_trained1(test_data_img, self.config['emb_mlp_layer'])
            print("y_1:", y_1.shape)
            print("fusion_model:", fusion_model.decpt_emb.shape)
            # todo: check why can't pass this asserting
            # assert torch.equal(fusion_model.chem_emb,y_1)

            model_trained2.eval()
            y_2 = model_trained2(test_data_tbl, self.config['emb_mlp_layer'])
            assert torch.equal(fusion_model.decpt_emb, y_2)

            model = DeepBioD(fusion_shape=fusion_shape,
                             fusion_model=fusion_model,
                             hidden_layers=self.config['last_dnn_hidden_layers'],
                             drop_out_rate=self.config['last_dnn_drop_out_rate'])

            # freeze fusion_model to get fixed embeddings
            # todo: be able to customize trainable layer
            # for param in model.fusion_model.parameters():
            #     param.requires_grad = False
            freeze_layers(fusion_model, self.config['freeze_mlp_layers_to'])

        return model.to(self.device)

    def train(self, model_number,use_diff_epochs=None):
        """
        :param use_diff_epochs: use this parameter to pass epochs when calling function
        """

        if use_diff_epochs:
            # provide an option to pass epochs from outside
            self.config['epochs'] = use_diff_epochs

        train_loader, val_loader = create_data_loader(model_number=model_number,
                                                      transform=self.config['transform'],
                                                      y_train=self.datasets['y_tr'],
                                                      y_val=self.datasets['y_val'],
                                                      X_train_tuple=self.datasets['X_tr_tuple'],
                                                      X_val_tuple=self.datasets['X_val_tuple'],
                                                      batch_size=self.config['batch_size'])
        # # in case of checking data loader outside
        # self.data_loaders[f'train_loader_mode{model_number}'] = train_loader
        # self.data_loaders[f'val_loader_mode{model_number}'] = val_loader

        criterion = torch.nn.BCELoss()
        model = self.get_model(model_number=model_number)

        if model is None:
            print("No model3 because can't averge embeddings of 2 mode. ")
            return "Can't train."

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                     lr=self.config['learning_rate'])

        if self.config['use_lr_scheduler']:
            # todo: better way to get length
            if model_number != 3:
                train_len = len(train_loader.dataset)
                val_len = len(val_loader.dataset)
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
                     model, optimizer, criterion, self.device,
                     self.config['rotate'], self.config['early_stop']]
        if model_number != 3:
            results = train_model(*arguments)
        else:
            results = train_multi_model(*arguments)

        self.trained_models[f'model_trained{model_number}'] = results[0]
        self.train_history[f'model{model_number}_history'] = results[1]

        return None
