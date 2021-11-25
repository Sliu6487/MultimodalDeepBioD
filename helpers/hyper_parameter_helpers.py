import numpy as np
import random


def random_search_hyper_parameters(model_number, multimodal):
    """
    Random search
    """
    # early_stop means take the best model based on validation,
    # doesn't mean stops earlier
    multimodal.config['early_stop'] = True
    multimodal.config['use_lr_scheduler'] = False
    multimodal.config['transform'] = False
    multimodal.config['rotate'] = True

    r = np.random.uniform(low=0.6, high=1, size=(1,)).item()
    lr = 10**(-5*r) # range from 0.001 to 0.00001
    while lr > 0.01:
        r = -5*np.random.rand()
        lr = 10**r

    multimodal.config['learning_rate'] = lr
    multimodal.config['epochs'] = random.choice([100,300,500,1000])
    multimodal.config['batch_size'] = random.choice([16, 32, 64, 128])

    print("learning rate:", multimodal.config['learning_rate'])
    print("epochs:", multimodal.config['epochs'])
    print("batch_size:", multimodal.config['batch_size'])

    hps_tuned = {
        "learning_rate": multimodal.config['learning_rate'],
        "epochs": multimodal.config['epochs'],
        "batch_size": multimodal.config['batch_size']
    }

    if model_number == 1:

        multimodal.config['n_inception_blocks'] = random.choice([1,2,3,4, 5])
        print("n_inception_blocks:", multimodal.config['n_inception_blocks'])
        hps_tuned['n_inception_blocks'] = multimodal.config['n_inception_blocks']

    elif model_number == 2:
        multimodal.config['n_neurons'] = random.choice([16, 32, 64, 126])
        multimodal.config['n_hidden_layers'] = random.choice([1, 2, 3, 4])
        multimodal.config['drop_out_rate'] = np.random.rand()

        print("n_neurons:", multimodal.config['n_neurons'])
        print("n_hidden_layers:", multimodal.config['n_hidden_layers'])
        print("drop_out_rate:", multimodal.config['drop_out_rate'])

        hps_tuned['n_neurons'] = multimodal.config['n_neurons']
        hps_tuned['n_hidden_layers'] = multimodal.config['n_hidden_layers']
        hps_tuned['drop_out_rate'] = multimodal.config['drop_out_rate']

    elif model_number == 3:

        multimodal.config['last_dnn_drop_out_rate'] = np.random.rand()

        n_mlp_layers = multimodal.config['n_hidden_layers'] + 2

        def get_fusion_parameters(n_mlp_layers):
            n_chem_blocks = multimodal.config['n_inception_blocks']
            multimodal.config['emb_chemception_section'] = random.choice(range(-n_chem_blocks, 0))
            multimodal.config['emb_mlp_layer'] = random.choice(range(-n_mlp_layers, 0))
            multimodal.config['fusion'] = random.choice(['avg', 'tf', 'concat'])

            return None

        get_fusion_parameters(n_mlp_layers)
        model3 = multimodal.get_model(model_number=3)

        while model3 is None:
            get_fusion_parameters()
            model3 = multimodal.get_model(model_number=3)

        fusion_size = model3.fusion_model.fusion_dict['fusion_neurons']
        if fusion_size == 2:
            multimodal.config['last_dnn_hidden_layers'] = random.choice([[0], [16], [32]])
        elif fusion_size <= 300:
            multimodal.config['last_dnn_hidden_layers'] = random.choice([[16], [32], [64, 32]])
        elif fusion_size <= 3000:
            multimodal.config['last_dnn_hidden_layers'] = random.choice([[128, 64], [512, 128, 64], [64]])
        elif fusion_size > 3000:
            multimodal.config['last_dnn_hidden_layers'] = random.choice([[128, 64], [512, 128, 64], [1024, 512, 128, 64]])


        multimodal.config['freeze_mlp_layers_to'] = random.choice(range(-n_mlp_layers, 0))

        print("last_dnn_hidden_layers:", multimodal.config['last_dnn_hidden_layers'])
        print("last_dnn_drop_out_rate:", multimodal.config['last_dnn_drop_out_rate'])
        print("emb_chemception_section:", multimodal.config['emb_chemception_section'])
        print("emb_mlp_layer:", multimodal.config['emb_mlp_layer'])
        print("fusion:", multimodal.config['fusion'])
        print("freeze_mlp_layers_to:", multimodal.config['freeze_mlp_layers_to'])

        hps_tuned['last_dnn_hidden_layers'] = multimodal.config['last_dnn_hidden_layers']
        hps_tuned['last_dnn_drop_out_rate'] = multimodal.config['last_dnn_drop_out_rate']
        hps_tuned['emb_chemception_section'] = multimodal.config['emb_chemception_section']
        hps_tuned['emb_mlp_layer'] = multimodal.config['emb_mlp_layer']
        hps_tuned['fusion'] = multimodal.config['fusion']
        hps_tuned['freeze_mlp_layers_to'] = multimodal.config['freeze_mlp_layers_to']


    return hps_tuned


def set_hyper_parameters(model_number, multimodal, config):
    # early_stop means take the best model based on validation,
    # doesn't mean stops earlier
    multimodal.config['early_stop'] = True
    multimodal.config['use_lr_scheduler'] = False
    multimodal.config['transform'] = False
    multimodal.config['rotate'] = True

    multimodal.config.update(config[f'model{model_number}'])

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


