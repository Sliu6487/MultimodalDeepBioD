import pickle
import random
import time
from random import randrange

import numpy as np
from IPython.display import display
from ipywidgets import IntProgress


class RandomSearch:
    def __init__(self):
        self.search_hist = {}
        self.rand_n = randrange(0, 10000)

    def fit(self, search_time, multimodal_obj, model_number, datasets, device):
        cv_train = CV_Train_Val(multimodal=multimodal_obj,
                                device=device)

        print(f'Seach {search_time} times...')
        f = IntProgress(min=0, max=search_time)
        display(f)

        for i in range(1, search_time + 1):
            print("------------------------------------------------------")
            print("------------------------------------------------------")
            print("Search:", i)

            cv_train.train(model_number=model_number,
                           X_tr_img=datasets['X_tr_tuple'][0],
                           X_tr_tbl=datasets['X_tr_tuple'][1],
                           y_tr=datasets['y_tr'],
                           X_val_img=datasets['X_val_tuple'][0],
                           X_val_tbl=datasets['X_val_tuple'][1],
                           y_val=datasets['y_val'])

            test_metirc = cv_train.hist_5cv[f'model{model_number}']['test_metirc']
            trained_models_5cv = cv_train.hist_5cv[f'model{model_number}']['trained_models_5cv']
            mean_val_metric = cv_train.hist_5cv[f'model{model_number}']['mean_val_metric']
            max_val_metric = cv_train.hist_5cv[f'model{model_number}']['max_val_metric']
            min_val_metric = cv_train.hist_5cv[f'model{model_number}']['min_val_metric']

            hist_i = {}
            hist_i['hps'] = cv_train.multimodal.config
            hist_i['test_err'] = test_metirc
            hist_i['mean_val_err'] = mean_val_metric
            hist_i['max_val_err'] = max_val_metric
            hist_i['min_val_err'] = min_val_metric
            hist_i['trained_models_5cv'] = trained_models_5cv

            self.search_hist[f'model{model_number}_search{i}'] = hist_i

            # update progress bar
            f.value += 1
            time.sleep(0.1)

            if i % 10 == 0:
                pickle_folder = "intermediate_files"
                with open(f'{pickle_folder}/search_hist_model{model_number}_{self.rand_n}.pkl', 'wb') as f:
                    pickle.dump(self.search_hist, f)
                print(f"✅Saved {i} search results.")

        pickle_folder = "results_files"
        with open(f'{pickle_folder}/search_hist_model{model_number}_{self.rand_n}.pkl', 'wb') as f:
            pickle.dump(self.search_hist, f)

        return None


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
    lr = 10 ** (-5 * r)  # range from 0.001 to 0.00001
    while lr > 0.01:
        r = -5 * np.random.rand()
        lr = 10 ** r

    multimodal.config['learning_rate'] = lr
    multimodal.config['epochs'] = random.choice([100, 300, 500, 1000])
    multimodal.config['batch_size'] = random.choice([16, 32, 64, 128])

    print("learning rate:", multimodal.config['learning_rate'])
    print("epochs:", multimodal.config['epochs'])
    print("batch_size:", multimodal.config['batch_size'])

    if model_number == 1:

        multimodal.config['n_inception_blocks'] = random.choice([1, 2, 3, 4, 5])
        print("n_inception_blocks:", multimodal.config['n_inception_blocks'])

    elif model_number == 2:
        multimodal.config['n_neurons'] = random.choice([16, 32, 64, 126])
        multimodal.config['n_hidden_layers'] = random.choice([1, 2, 3, 4])
        multimodal.config['drop_out_rate'] = np.random.rand()

        print("n_neurons:", multimodal.config['n_neurons'])
        print("n_hidden_layers:", multimodal.config['n_hidden_layers'])
        print("drop_out_rate:", multimodal.config['drop_out_rate'])

    elif model_number == 3:

        multimodal.config['last_dnn_drop_out_rate'] = np.random.rand()

        def get_fusion_parameters():
            n_mlp_layers = multimodal.config['n_hidden_layers'] + 2
            n_chem_blocks = multimodal.config['n_inception_blocks']
            multimodal.config['emb_chemception_section'] = random.choice(range(-n_chem_blocks, 0))
            multimodal.config['emb_mlp_layer'] = random.choice(range(-n_mlp_layers, 0))
            multimodal.config['fusion'] = random.choice(['avg', 'tf', 'concat'])

            return None

        get_fusion_parameters()
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
            multimodal.config['last_dnn_hidden_layers'] = random.choice(
                [[128, 64], [512, 128, 64], [1024, 512, 128, 64]])

        multimodal.config['freeze_mlp_layers_to'] = random.choice(range(-n_mlp_layers, 0))

        print("last_dnn_hidden_layers:", multimodal.config['last_dnn_hidden_layers'])
        print("last_dnn_drop_out_rate:", multimodal.config['last_dnn_drop_out_rate'])
        print("emb_chemception_section:", multimodal.config['emb_chemception_section'])
        print("emb_mlp_layer:", multimodal.config['emb_mlp_layer'])
        print("fusion_method:", multimodal.config['fusion'])
        print("freeze_mlp_layers_to:", multimodal.config['freeze_mlp_layers_to'])

    return None
