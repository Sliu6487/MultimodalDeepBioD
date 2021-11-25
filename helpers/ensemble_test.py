import torch

from train.metrics import clf_err_rate


def ensemble_test(cv_train_val, model_number, X_test_img, X_test_tbl, y_test, device):
    assert model_number in [1, 2, 3]
    trained_model_5cv = cv_train_val.hist_5cv_val[f'model{model_number}']['trained_models_5cv']

    predictions = []
    for i in range(5):
        if model_number == 3:
            outputs = trained_model_5cv[i](X_test_img.to(device), X_test_tbl.to(device))
        elif model_number == 1:
            outputs = trained_model_5cv[i](X_test_img.to(device))
        else:
            outputs = trained_model_5cv[i](X_test_tbl.to(device))

        y_pred = torch.round(outputs.data)
        predictions.append(y_pred)

    ensemble_pred = sum(predictions) / len(predictions)
    y_pred = torch.empty(ensemble_pred.shape, dtype=torch.int32, device=device)
    pos_indx = (ensemble_pred >= 0.6).nonzero(as_tuple=True)[0]
    neg_indx = (ensemble_pred < 0.6).nonzero(as_tuple=True)[0]
    y_pred[pos_indx] = 1
    y_pred[neg_indx] = 0

    test_metric = clf_err_rate(y_test, y_pred).numpy().item()

    print(f'✅ test metric: {round(test_metric, 4)}')

    return None
