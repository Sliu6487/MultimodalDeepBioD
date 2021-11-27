import numpy as np
from sklearn.model_selection import KFold

from helpers.data_helpers import upsample
from helpers.ensemble_test import ensemble_test
from helpers.hyper_parameter_helpers import set_hyper_parameters, random_search_hyper_parameters
from helpers.result_helpers import show_training_plots
from train.multimodels import MultiModels


class CvTrainTest:
    def __init__(self, multimodal: MultiModels, datasets: dict, folds: int = 5):

        """
        5-fold cross validate training set then ensemble predictions on a test set.
        :param multimodal: multimodal object that is going to be tuned
        :param datasets: dataset that is used for hyper-parameter tuning and test
                               tune_datasets = {'X_train_tuple': (X_train_img, X_train_tbl),
                                                'X_test_tuple': (X_test_img, X_test_tbl),
                                                'y_train': y_train,
                                                'y_test': y_test}
        """

        self.multimodal = multimodal
        self.device = self.multimodal.device
        self.datasets = datasets
        self.folds = folds

        self.hist_dict = {}
        self.prepare_datasets()

    def prepare_datasets(self):

        # test datasets
        self.hist_dict['test_datasets'] = {
            'X_test_img': self.datasets['X_test_tuple'][0],
            'X_test_tbl': self.datasets['X_test_tuple'][1],
            'y_test': self.datasets['y_test']
        }

        # cv datasets: train-val
        X_tr_img = self.datasets['X_train_tuple'][0]
        X_tr_tbl = self.datasets['X_train_tuple'][1]
        y_tr = self.datasets['y_train']

        kf = KFold(n_splits=self.folds, random_state=41, shuffle=True)
        fold = 1
        for t_index, v_index in kf.split(np.zeros(y_tr.shape[0]), y_tr):
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

            self.hist_dict[f'cv_datasets_fold{fold}'] = cv_datasets
            fold += 1

        return None

    def cross_validate(self, model_number: int,
                       config: dict = None,
                       test_epochs: int = None,
                       show_plots: bool = True):

        print("🟢 Model number trained:", model_number)
        print("➡️ Hyper-parameters:")
        if config is None:
            random_search_hyper_parameters(model_number, self.multimodal)
        else:
            set_hyper_parameters(model_number, self.multimodal, config=config)

        if test_epochs:
            self.multimodal.config['epochs'] = test_epochs

        self.hist_dict[f'model{model_number}'] = {}
        best_val_metric = []
        trained_model_cv = []

        for fold in range(1, self.folds + 1):
            self.multimodal.datasets = self.hist_dict[f'cv_datasets_fold{fold}']

            if model_number == 3:
                self.multimodal.trained_models['model_trained1'] = self.hist_dict['model1']['trained_models_cv'][
                    fold - 1]
                self.multimodal.trained_models['model_trained2'] = self.hist_dict['model2']['trained_models_cv'][
                    fold - 1]

            train_result = self.multimodal.train(model_number=model_number)

            if train_result == "Can't train.":
                raise ValueError("Can't train. Hyper-parameter problem.")

            if show_plots:
                show_training_plots(model_number=model_number, show_accuracy=False, show_lr=False,
                                    train_history_dict=self.multimodal.train_history)

            hist = self.multimodal.train_history[f'model{model_number}_history']
            self.hist_dict[f'model{model_number}'][f'hist_cv{fold}'] = hist

            best_val_metric.append(min(hist['hist_metrics_val']).item())
            trained_model_cv.append(self.multimodal.trained_models[f'model_trained{model_number}'])

        # summarize cv
        self.hist_dict[f'model{model_number}']['trained_models_cv'] = trained_model_cv

        mean_val = sum(best_val_metric) / len(best_val_metric)
        max_val = max(best_val_metric)
        min_val = min(best_val_metric)
        self.hist_dict[f'model{model_number}']['mean_val_metric'] = mean_val
        self.hist_dict[f'model{model_number}']['max_val_metric'] = max_val
        self.hist_dict[f'model{model_number}']['min_val_metric'] = min_val
        print("best_val_metric:", [round(metric, 4) for metric in best_val_metric])
        print(f'☑️ mean_val_metric: {round(mean_val, 4)}, '
              f'max_val_metric: {round(max_val, 4)}, '
              f'min_val_metric: {round(min_val, 4)}')

        return None

    def test(self, model_number):
        trained_model_cv = self.hist_dict[f'model{model_number}']['trained_models_cv']
        test_metric = ensemble_test(trained_model_cv=trained_model_cv,
                                    folds=self.folds,
                                    model_number=model_number,
                                    test_datasets=self.hist_dict['test_datasets'],
                                    device=self.device)

        self.hist_dict[f'model{model_number}']['test_metric'] = test_metric

        return None
