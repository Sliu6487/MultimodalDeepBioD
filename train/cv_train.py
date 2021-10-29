import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torchvision import transforms as T

from models.chemception_models import Chemception, Chemception_Small
from models.dnn_models import MLP_DNN
from train.metrics import clf_err_rate, get_accuracy

with open('config_files/default_config0.json') as f:
    config = json.load(f)

config["kf"] = KFold(n_splits=5, random_state=41, shuffle=True)
config["customized_metric"] = clf_err_rate


class CV_Train:
    def __init__(self, model_name='Chemception',
                 optimizer_name='RMSprop',
                 criterion=torch.nn.BCELoss(),
                 augmentation=False,
                 input_shape=(80, 80, 4),
                 num_neurons=128, num_hidden_layers=2,
                 drop_out_rate=0.5,
                 ):

        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.criterion = criterion

        # Chemception
        self.augmentation = augmentation
        self.augmentation = False

        # DNN
        self.input_shape = input_shape
        self.num_neurons = num_neurons
        self.num_hidden_layers = num_hidden_layers
        self.drop_out_rate = drop_out_rate

    def _train(self, X_train, y_train,
               kf=config['kf'],
               device='cpu',
               repeat_train=False,
               optimizer_name='RMSprop',
               learning_rate=config['learning_rate'],
               batch_size=config['batch_size'],
               num_epochs=config['num_epochs'],
               patience=config['patience'],
               verbose=0,
               use_lr_scheduler='batch',
               lr_scheduler_mode='triangular',
               max_lr=0.03,
               base_lr=0.001,
               step_size_up=120):

        #### all CV - avg best error rate in all CVs#####
        all_cv = []
        cv_s = 1
        rotater = T.RandomRotation(degrees=(0, 180))
        for t_index, v_index in kf.split(np.zeros(y_train.shape[0]),
                                         y_train):

            #### one CV - summarize all epochs #####
            print("CV split:", cv_s)

            X_t = X_train[t_index].to(device)
            y_t = y_train[t_index].to(device)
            if repeat_train:
                X_t = X_t.repeat(repeat_train, 1, 1, 1)
                y_t = y_t.repeat(repeat_train, 1)
            X_v = X_train[v_index].to(device)
            y_v = y_train[v_index].to(device)

            cv_s += 1
            # model, optimizer = get_model_and_optimizer()
            # model
            if self.model_name == 'Chemception':
                model = Chemception(augment=self.augmentation).to(device)

            elif self.model_name == 'Chemception_Small':
                model = Chemception_Small(augment=self.augmentation).to(device)

            elif self.model_name == 'MLP_DNN':
                model = MLP_DNN(input_shape=self.input_shape,
                                num_hidden_layers=self.num_hidden_layers,
                                num_neurons=self.num_neurons,
                                drop_out_rate=self.drop_out_rate).to(device)

            else:
                raise ValueError(f'No model named {self.model_name}!')
            # optimizer
            optimizer = None
            scheduler = None
            if optimizer_name == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(),
                                                lr=learning_rate,
                                                alpha=0.99,
                                                eps=1e-08)
            elif optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=learning_rate)
            elif optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=learning_rate)

            if use_lr_scheduler == 'batch':
                # iteration = train_size/batch_size = 30
                # step_size: 4*iteration = 120
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                              base_lr=base_lr,
                                                              max_lr=max_lr,
                                                              step_size_up=step_size_up,
                                                              mode=lr_scheduler_mode)
            elif use_lr_scheduler == 'epoch':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                              base_lr=base_lr,
                                                              max_lr=max_lr,
                                                              step_size_up=step_size_up,
                                                              mode=lr_scheduler_mode)

            hist_metric_all_epoch_one_cv = []
            hist_metric_val_all_epoch_one_cv = []
            hist_loss_all_epoch_one_cv = []
            hist_loss_val_all_epoch_one_cv = []
            hist_lr_all_epoch_one_cv = []
            hist_acc_all_epoch_one_cv = []
            hist_acc_val_all_epoch_one_cv = []

            tolerance = 0
            for epoch in range(num_epochs):
                if verbose > 1:
                    print('Epoch:', epoch + 1)

                ##### one epoch - batch training #######
                best_cls_err_rate = 100
                total_loss = 0
                y_pred = torch.tensor(np.empty(y_t.shape)).float().to(device)
                # 1.1 train
                model.train()
                permutation = torch.randperm(X_t.size()[0])
                steps = 0
                for i in range(0, X_t.size()[0], batch_size):
                    optimizer.zero_grad()
                    indices = permutation[i:i + batch_size]

                    batch_x, batch_y = rotater(X_t[indices]), y_t[indices]

                    outputs = model(batch_x)
                    if torch.isnan(outputs).any():
                        print('training prediction has null value')
                    loss = self.criterion(outputs, batch_y)
                    # use float(loss) to prevent GPU out of memory
                    total_loss += float(loss)
                    batch_y_pred = torch.round(outputs.data)
                    y_pred[indices] = batch_y_pred

                    loss.backward()
                    optimizer.step()
                    steps += 1
                    # change here: scheduler step among batches
                    if use_lr_scheduler == 'batch':
                        scheduler.step()
                        lr = optimizer.param_groups[0]["lr"]
                        hist_lr_all_epoch_one_cv.append(float(lr))
                if use_lr_scheduler == 'epoch':
                    scheduler.step()
                    lr = optimizer.param_groups[0]["lr"]
                    hist_lr_all_epoch_one_cv.append(float(lr))

                # 1.2 metrics
                cls_err_rate = clf_err_rate(y_t, y_pred).numpy()
                accuracy = get_accuracy(y_t, y_pred)
                # hist_acc_all_epoch_one_cv.append(accuracy)
                hist_metric_all_epoch_one_cv.append(float(cls_err_rate))
                # hist_loss_all_epoch_one_cv.append(float(loss))
                hist_loss_all_epoch_one_cv.append(total_loss / steps)
                hist_acc_all_epoch_one_cv.append(accuracy)

                del y_pred
                del loss
                del permutation
                del cls_err_rate

                #### one epoch - validation #####
                # 2.1 predict
                model.eval()
                outputs_val = model(X_v)
                if torch.isnan(outputs_val).any():
                    print('validation prediction has null value')
                # print(outputs_val.shape,y_v.shape)
                loss_val = self.criterion(outputs_val, y_v)
                y_pred_val = torch.round(outputs_val.data)

                # 2.2 metrics
                cls_err_rate_val = clf_err_rate(y_v, y_pred_val).numpy()
                accuracy_val = get_accuracy(y_v, y_pred_val)
                hist_acc_val_all_epoch_one_cv.append(accuracy_val)
                hist_metric_val_all_epoch_one_cv.append(float(cls_err_rate_val))
                hist_loss_val_all_epoch_one_cv.append(float(loss_val))

                # 3. print results
                if epoch >= 1:
                    if loss_val < hist_loss_val_all_epoch_one_cv[-2]:
                        if verbose > 1:
                            print(f'model improved from last epoch '
                                  f'{hist_loss_val_all_epoch_one_cv[-2]}.')
                        tolerance = 0
                        if loss_val < min(hist_loss_val_all_epoch_one_cv):
                            if verbose > 0:
                                print(f'best so far: {loss_val}.')
                    # early stopping
                    else:
                        if verbose > 1:
                            print(f'model did not improve from last epoch '
                                  f'{hist_loss_val_all_epoch_one_cv[-2]}.')
                        tolerance += 1
                        if verbose > 1:
                            print('tolerance', tolerance)
                        if tolerance == patience:
                            if verbose > 0:
                                print(f'Early stopping. No improvement in last '
                                      f'{patience} epochs')
                            break

                            #### one epoch - validation #####

            #### one CV - summarize all epochs #####
            if verbose > 0:
                plt.plot(hist_metric_all_epoch_one_cv, color="blue")
                plt.plot(hist_metric_val_all_epoch_one_cv, color="orange")
                plt.legend(['train_metric', 'val_metric'])
                plt.xlabel('epochs')
                plt.show()

                plt.plot(hist_acc_all_epoch_one_cv)
                plt.plot(hist_acc_val_all_epoch_one_cv)
                plt.legend(['train_acc', 'val_acc'])
                plt.xlabel('epochs')
                plt.show()

                plt.plot(hist_loss_all_epoch_one_cv)
                plt.plot(hist_loss_val_all_epoch_one_cv)
                plt.legend(['train_loss', 'val_loss'])
                plt.xlabel('epochs')
                plt.show()

                plt.plot(hist_lr_all_epoch_one_cv)
                plt.legend(['learning rate'])
                if use_lr_scheduler == 'epoch':
                    plt.xlabel('epochs')
                elif use_lr_scheduler == 'batch':
                    plt.xlabel('iterations(batchs)')
                plt.show()

            best_cls_err_rate_val_one_cv = min(hist_metric_val_all_epoch_one_cv)
            all_cv.append(best_cls_err_rate_val_one_cv)
            #### one CV - summarize all epochs #####

        #### all CV - avg best error rate in all CVs#####
        avg_aer = sum(all_cv) / len(all_cv)
        print('avg CV metric score:', avg_aer)
        return avg_aer

# avg_aer, avg_loss, *_ = cv_train(X_train, y_train, model,callbacks_list)
