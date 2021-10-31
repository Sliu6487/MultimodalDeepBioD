import copy
import time

import numpy as np
import torch
from IPython.display import display
from ipywidgets import IntProgress
from torchvision import transforms

from train.metrics import clf_err_rate, get_accuracy


def train_model(epochs, scheduler,
                train_loader, val_loader,
                model, optimizer, criterion,
                device='cpu', rotate=True):
    best_model = None
    best_val_metric = 1000
    hist_accuracy_train = []
    hist_metrics_train = []
    hist_losses_train = []
    hist_lr_all_batchs = []

    hist_metrics_val = []
    hist_accuracy_val = []
    hist_losses_val = []

    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    rotator = transforms.RandomRotation(degrees=(0, 180))
    # train_loader.dataset.tensors[1].shape[0]
    # val_loader.dataset.tensors[1].shape[0]
    # print("train_len, val_len:",train_len,val_len)

    # instantiate progress bar
    print(f'{epochs} epochs...')
    f = IntProgress(min=0, max=epochs)
    display(f)

    for epoch in range(epochs):
        # training
        model.train()
        index, steps, total_loss = (0, 0, 0)
        y_pred = torch.tensor(np.empty((train_len, 1))).float().to(device)
        y_true = torch.tensor(np.empty((train_len, 1))).float().to(device)
        for x_train, y_train in train_loader:
            if rotate & (len(x_train.shape) == 4):
                # random rotate image inputs
                x_train = rotator(x_train)
            batch_size = y_train.shape[0]
            optimizer.zero_grad()
            outputs = model(x_train.to(device))
            # print('outputs shape:', outputs.shape)
            # print('y_train shape:', y_train.shape)
            loss = criterion(outputs, y_train.to(device))
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            hist_lr_all_batchs.append(float(lr))
            total_loss += float(loss)
            batch_y_pred = torch.round(outputs.data)
            y_pred[index: index + batch_size] = batch_y_pred
            y_true[index: index + batch_size] = y_train

            index += batch_size
            steps += 1

        hist_metrics_train.append(clf_err_rate(y_true, y_pred).numpy())
        hist_accuracy_train.append(get_accuracy(y_true, y_pred))
        hist_losses_train.append(total_loss / steps)

        # validation
        model.eval()
        index, steps, total_loss = (0, 0, 0)
        y_pred = torch.tensor(np.empty((val_len, 1))).float().to(device)
        y_true = torch.tensor(np.empty((val_len, 1))).float().to(device)
        for x_val, y_val in val_loader:
            batch_size = y_val.shape[0]
            outputs = model(x_val.to(device))
            loss = criterion(outputs, y_val.to(device))
            total_loss += float(loss)
            batch_y_pred = torch.round(outputs.data)
            y_pred[index: index + batch_size] = batch_y_pred
            y_true[index: index + batch_size] = y_val
            index += batch_size
            steps += 1

        val_metric = clf_err_rate(y_true, y_pred).numpy()
        hist_metrics_val.append(val_metric)
        hist_accuracy_val.append(get_accuracy(y_true, y_pred))
        hist_losses_val.append(total_loss / steps)

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_model = copy.deepcopy(model)

        # update progress bar
        f.value += 1
        time.sleep(0.1)

    print('Best val_metric: {:.3f}'.format(best_val_metric))

    hist = {'hist_metrics_train': hist_metrics_train,
            'hist_losses_train': hist_losses_train,
            'hist_accuracy_train': hist_accuracy_train,
            'hist_lr_all_batchs': hist_lr_all_batchs,
            'hist_metrics_val': hist_metrics_val,
            'hist_losses_val': hist_losses_val,
            'hist_accuracy_val': hist_accuracy_val}

    return best_model, hist


def train_multi_model(epochs, scheduler,
                      train_loader, val_loader,
                      model, optimizer, criterion,
                      device='cpu', rotate=True):
    best_model = None
    best_val_metric = 1000
    hist_metrics_train = []
    hist_losses_train = []
    hist_metrics_val = []
    hist_losses_val = []

    rotator = transforms.RandomRotation(degrees=(0, 180))

    # instantiate progress bar
    print(f'{epochs} epochs...')
    f = IntProgress(min=0, max=epochs)
    display(f)

    for epoch in range(epochs):
        # training
        index, steps, total_loss = (0, 0, 0)
        train_len = len(train_loader.dataset)
        y_pred = torch.tensor(np.empty((train_len, 1))).float().to(device)
        y_true = torch.tensor(np.empty((train_len, 1))).float().to(device)
        for dataset1, dataset2 in train_loader:
            model.train()
            x_train_img = dataset1[0].to(device)
            if rotate:
                x_train_img = rotator(x_train_img)
            x_train_tbl = dataset2[0].to(device)
            y_train = dataset1[1].to(device)
            batch_size = y_train.shape[0]
            optimizer.zero_grad()
            outputs = model(x_train_img, x_train_tbl)  # (batch_size, 1)
            loss = criterion(outputs, y_train.to(device))
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += float(loss)
            batch_y_pred = torch.round(outputs.data)
            y_pred[index: index + batch_size] = batch_y_pred
            y_true[index: index + batch_size] = y_train
            index += batch_size
            steps += 1

        hist_metrics_train.append(clf_err_rate(y_true, y_pred).numpy())
        hist_losses_train.append(total_loss / steps)
        # validation
        index, steps, total_loss = (0, 0, 0)
        val_len = len(val_loader.dataset)
        y_pred = torch.tensor(np.empty((val_len, 1))).float().to(device)
        y_true = torch.tensor(np.empty((val_len, 1))).float().to(device)
        for dataset1, dataset2 in val_loader:
            model.eval()
            x_val_img = dataset1[0].to(device)
            x_val_tbl = dataset2[0].to(device)
            y_val = dataset1[1].to(device)
            batch_size = y_val.shape[0]
            outputs = model(x_val_img, x_val_tbl)
            loss = criterion(outputs, y_val.to(device))
            total_loss += float(loss)
            batch_y_pred = torch.round(outputs.data)
            y_pred[index: index + batch_size] = batch_y_pred
            y_true[index: index + batch_size] = y_val
            index += batch_size
            steps += 1

        val_metric = clf_err_rate(y_true, y_pred).numpy()
        hist_metrics_val.append(val_metric)
        hist_losses_val.append(total_loss / steps)

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_model = copy.deepcopy(model)

        # update progress bar
        f.value += 1  # signal to increment
        time.sleep(0.1)

    print('Best val_metric: {:.3f}'.format(best_val_metric))

    hist = {'hist_metrics_train': hist_metrics_train,
            'hist_losses_train': hist_losses_train,
            'hist_metrics_val': hist_metrics_val,
            'hist_losses_val': hist_losses_val}

    return best_model, hist
