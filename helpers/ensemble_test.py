import torch

from train.metrics import clf_err_rate


def ensemble_test(trained_model_cv: list, model_number: int, test_datasets: dict,
                  folds: int = 5, device: str = 'cpu') -> float:
    assert model_number in [1, 2, 3]
    X_test_img = test_datasets['X_test_img']
    X_test_tbl = test_datasets['X_test_tbl']
    y_test = test_datasets['y_test']
    soft_predictions = []
    hard_predictions = []

    for i in range(folds):
        if model_number == 3:
            outputs = trained_model_cv[i](X_test_img.to(device), X_test_tbl.to(device))
        elif model_number == 1:
            outputs = trained_model_cv[i](X_test_img.to(device))
        else:
            outputs = trained_model_cv[i](X_test_tbl.to(device))

        soft_predictions.append(outputs.data)
        y_pred = torch.round(outputs.data)
        hard_predictions.append(y_pred)

    ensemble_pred = sum(hard_predictions) / len(hard_predictions)
    y_pred = torch.empty(ensemble_pred.shape, dtype=torch.int32, device=device)
    # todo: customize this threshold to other fold numbers
    thresh_hold = 0.6
    pos_indx = (ensemble_pred >= thresh_hold).nonzero(as_tuple=True)[0]
    neg_indx = (ensemble_pred < thresh_hold).nonzero(as_tuple=True)[0]
    y_pred[pos_indx] = 1
    y_pred[neg_indx] = 0

    test_metric = clf_err_rate(y_test, y_pred).numpy().item()
    print(f'✅ test metric: {round(test_metric, 4)}')

    return test_metric


# def ensemble_test(cv_train_val, model_number, X_test_img, X_test_tbl, y_test, device):
#     assert model_number in [1, 2, 3]
#     trained_model_cv = cv_train_val.hist_cv_val[f'model{model_number}']['trained_models_cv']
#
#     predictions = []
#     for i in range(5):
#         if model_number == 3:
#             outputs = trained_model_cv[i](X_test_img.to(device), X_test_tbl.to(device))
#         elif model_number == 1:
#             outputs = trained_model_cv[i](X_test_img.to(device))
#         else:
#             outputs = trained_model_cv[i](X_test_tbl.to(device))
#
#         y_pred = torch.round(outputs.data)
#         predictions.append(y_pred)
#
#     ensemble_pred = sum(predictions) / len(predictions)
#     y_pred = torch.empty(ensemble_pred.shape, dtype=torch.int32, device=device)
#     pos_indx = (ensemble_pred >= 0.6).nonzero(as_tuple=True)[0]
#     neg_indx = (ensemble_pred < 0.6).nonzero(as_tuple=True)[0]
#     y_pred[pos_indx] = 1
#     y_pred[neg_indx] = 0
#
#     test_metric = clf_err_rate(y_test, y_pred).numpy().item()
#
#     print(f'✅ test metric: {round(test_metric, 4)}')
#
#     return None
