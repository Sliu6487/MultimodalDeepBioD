import numpy as np
import torch
from sklearn.model_selection import KFold

from helpers.data_helpers import upsample
from helpers.result_helpers import show_training_plots
from train.metrics import clf_err_rate
from train.random_search import random_search_hyper_parameters


class CV_Train_Val:
    def __init__(self, multimodal, device):

        self.multimodal = multimodal
        self.device = device

        self.hist_5cv = {}

    def train(self, model_number,
              X_tr_img, X_tr_tbl, y_tr,
              X_val_img, X_val_tbl, y_val,
              hp_tuning=True,
              test_epochs=None):

        print("🟢 Model number trained:", model_number)
        self.hist_5cv['model_number'] = model_number

        if hp_tuning:
            print("➡️ Hyperparameters:")
            hps_tuned = random_search_hyper_parameters(model_number, self.multimodal)

            self.hist_5cv['hps_tuned'] = hps_tuned

        if test_epochs:
            self.multimodal.config['epochs'] = test_epochs

        self.hist_5cv[f'model{model_number}'] = {}

        best_val_metric = []
        trained_model_5cv = []

        kf = KFold(n_splits=5, random_state=41, shuffle=True)
        cv_f = 1

        for t_index, v_index in kf.split(np.zeros(y_tr.shape[0]), y_tr):
            print("♻️ Fold:", cv_f)
            X_t_img = X_tr_img[t_index]
            X_t_tbl = X_tr_tbl[t_index]
            y_t = y_tr[t_index]

            # upsample
            X_t_img, X_t_tbl, y_t = upsample(X_t_img, X_t_tbl, y_t)

            X_v_img = X_tr_img[v_index]
            X_v_tbl = X_tr_tbl[v_index]
            y_v = y_tr[v_index]

            cv_datasets = {'X_tr_tuple': (X_t_img, X_t_tbl),
                           'X_val_tuple': (X_v_img, X_v_tbl),
                           'y_tr': y_t,
                           'y_val': y_v}

            self.multimodal.datasets = cv_datasets

            train_result = self.multimodal.train(model_number=model_number,
                                                 epochs=self.multimodal.config['epochs'],
                                                 rotate=self.multimodal.config['rotate'],
                                                 early_stop=self.multimodal.config['early_stop'])

            if train_result == "Can't train.":
                raise ValueError("Can't train. Hyper-paramter problem.")

            show_training_plots(model_number=model_number, show_accuracy=False, show_lr=False,
                                train_history_dict=self.multimodal.train_history)

            hist = self.multimodal.train_history[f'model{model_number}_history']
            self.hist_5cv[f'model{model_number}'][f'hist_cv{cv_f}'] = hist
            min_val_hist = min(hist['hist_metrics_val']).item()
            best_val_metric.append(min_val_hist)

            trained_model_5cv.append(self.multimodal.trained_models[f'model_trained{model_number}'])

            cv_f += 1

        mean_val = sum(best_val_metric) / len(best_val_metric)
        max_val = max(best_val_metric)
        min_val = min(best_val_metric)
        self.hist_5cv[f'model{model_number}']['mean_val_metric'] = mean_val
        self.hist_5cv[f'model{model_number}']['max_val_metric'] = max_val
        self.hist_5cv[f'model{model_number}']['min_val_metric'] = min_val
        print("best_val_metric:", [round(metric, 4) for metric in best_val_metric])
        print(
            f'☑️ mean_val_metric: {round(mean_val, 4)}, max_val_metric: {round(max_val, 4)}, min_val_metric: {round(min_val, 4)}')

        predictions = []
        for i in range(5):
            if model_number == 3:
                outputs = trained_model_5cv[i](X_val_img.to(self.device), X_val_tbl.to(self.device))
            elif model_number == 1:
                outputs = trained_model_5cv[i](X_val_img.to(self.device))
            elif model_number == 2:
                outputs = trained_model_5cv[i](X_val_tbl.to(self.device))

            y_pred = torch.round(outputs.data)
            predictions.append(y_pred)
        emsemble_pred = sum(predictions) / len(predictions)
        y_pred = torch.empty(emsemble_pred.shape, dtype=torch.int32, device=self.device)
        pos_indx = (emsemble_pred >= 0.6).nonzero(as_tuple=True)[0]
        neg_indx = (emsemble_pred < 0.6).nonzero(as_tuple=True)[0]
        y_pred[pos_indx] = 1
        y_pred[neg_indx] = 0

        test_metric = clf_err_rate(y_val, y_pred).numpy().item()

        print(f'✅ test metric: {round(test_metric, 4)}')

        self.hist_5cv[f'model{model_number}']['test_metirc'] = test_metric
        self.hist_5cv[f'model{model_number}']['trained_models_5cv'] = trained_model_5cv

        return None
