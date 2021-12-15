import random

import numpy as np

from train.multimodels import MultiModels


def random_search_hyper_parameters(model_number: int,
                                   multimodal: MultiModels) -> None:
    """
    Random search with some restrains.
    """

    # early_stop means take the best model based on validation,
    # doesn't mean stop earlier
    multimodal.config['early_stop'] = True
    multimodal.config['use_lr_scheduler'] = False
    multimodal.config['transform'] = False
    multimodal.config['rotate'] = True

    r = np.random.uniform(low=0.6, high=1, size=(1,)).item()
    lr = 10 ** (-5 * r)
    # range from 0.001 to 0.00001
    while lr > 0.001:
        r = -5 * np.random.rand()
        lr = 10 ** r

    multimodal.config['learning_rate'] = lr
    multimodal.config['epochs'] = random.choice([1, 2])
    multimodal.config['batch_size'] = random.choice([16, 32, 64, 128])

    print("learning rate:", multimodal.config['learning_rate'])
    print("epochs:", multimodal.config['epochs'])
    print("batch_size:", multimodal.config['batch_size'])

    if model_number == 1:

        # multimodal.config['n_inception_blocks'] = random.choice([1,2,3,4, 5])
        multimodal.config['n_inception_blocks'] = 5
        print("n_inception_blocks:", multimodal.config['n_inception_blocks'])

    elif model_number == 2:
        # multimodal.config['n_neurons'] = random.choice([16, 32, 64, 126])
        multimodal.config['n_neurons'] = 126
        multimodal.config['n_hidden_layers'] = random.choice([1, 2, 3, 4])
        multimodal.config['drop_out_rate'] = np.random.uniform(0.2, 0.8)

        print("n_neurons:", multimodal.config['n_neurons'])
        print("n_hidden_layers:", multimodal.config['n_hidden_layers'])
        print("drop_out_rate:", multimodal.config['drop_out_rate'])

    elif model_number == 3:

        multimodal.config['last_dnn_drop_out_rate'] = np.random.rand()

        emb1, emb2 = random.choice([[-1, -1], [-2, -2], [-2, -4]])
        multimodal.config['emb_chemception_section'] = emb1
        multimodal.config['emb_mlp_layer'] = emb2
        multimodal.config['fusion'] = random.choice(['shrink', 'no_harm'])
        multimodal.config['last_dnn_hidden_layers'] = []

        multimodal.config['freeze_mlp_layers_to'] = random.choice([-1, -5 - emb2])

        print("last_dnn_hidden_layers:", multimodal.config['last_dnn_hidden_layers'])
        print("last_dnn_drop_out_rate:", multimodal.config['last_dnn_drop_out_rate'])
        print("emb_chemception_section:", multimodal.config['emb_chemception_section'])
        print("emb_mlp_layer:", multimodal.config['emb_mlp_layer'])
        print("fusion_method:", multimodal.config['fusion'])
        print("freeze_mlp_layers_to:", multimodal.config['freeze_mlp_layers_to'])

    return None


def set_hyper_parameters(model_number: int,
                         multimodal: MultiModels,
                         default_config: dict = None,
                         config_by_model: dict = None) -> None:

    # early_stop means take the best model based on validation,
    # doesn't mean stops earlier
    multimodal.config['early_stop'] = True
    multimodal.config['use_lr_scheduler'] = False
    multimodal.config['transform'] = False
    multimodal.config['rotate'] = True

    if config_by_model:
        multimodal.config.update(config_by_model[f'model{model_number}'])
    elif default_config:
        multimodal.config.update(default_config)
    else:
        raise ValueError('Please pass a config dict to set hyper-parameters.')

    print("learning rate:", multimodal.config['learning_rate'])
    print("epochs:", multimodal.config['epochs'])
    print("batch_size:", multimodal.config['batch_size'])

    if model_number == 1:

        print("n_inception_blocks:", multimodal.config['n_inception_blocks'])

    elif model_number == 2:

        print("n_neurons:", multimodal.config['n_neurons'])
        print("n_hidden_layers:", multimodal.config['n_hidden_layers'])
        print("drop_out_rate:", multimodal.config['drop_out_rate'])

    elif model_number == 3:

        print("last_dnn_hidden_layers:", multimodal.config['last_dnn_hidden_layers'])
        print("last_dnn_drop_out_rate:", multimodal.config['last_dnn_drop_out_rate'])
        print("emb_chemception_section:", multimodal.config['emb_chemception_section'])
        print("emb_mlp_layer:", multimodal.config['emb_mlp_layer'])
        print("fusion:", multimodal.config['fusion'])
        print("freeze_mlp_layers_to:", multimodal.config['freeze_mlp_layers_to'])

    return None
