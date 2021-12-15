import pickle
import random
import time
from random import randrange

import numpy as np
from IPython.display import display
from ipywidgets import IntProgress

from train.cross_validation import CVTune


class RandomSearch:
    def __init__(self):
        self.search_hist = {}
        self.rand_n = randrange(0, 10000)

    def fit(self, search_time, cv_tune: CVTune, model_number):

        print(f'Search {search_time} times...')
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

            test_metirc = cv_train.hist_dict[f'model{model_number}']['test_metirc']
            trained_models_5cv = cv_train.hist_dict[f'model{model_number}']['trained_models_5cv']
            mean_val_metric = cv_train.hist_dict[f'model{model_number}']['mean_val_metric']
            max_val_metric = cv_train.hist_dict[f'model{model_number}']['max_val_metric']
            min_val_metric = cv_train.hist_dict[f'model{model_number}']['min_val_metric']

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
