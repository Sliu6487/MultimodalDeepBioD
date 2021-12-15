import torch


def clf_err_rate(y_true, y_pred):
    """
    Classification error rate in the paper has a mistake:
    Er = 1 - (specificity - sensitivity)/2
    based on the cited paper, it should be:
    Er = 1 - (specificity + sensitivity)/2
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred

    eps = 1e-10

    fp = torch.sum(neg_y_true * y_pred)  # (0->1, 1)  #ok
    tn = torch.sum(neg_y_true * neg_y_pred)  # (0->1,0->1)
    # added K.epsilon() to the denominator
    # to prevent a divide by zero error
    sp = tn / (tn + fp + eps)  # + K.epsilon()

    fn = torch.sum(y_true * neg_y_pred)  # (1, 0->1)
    tp = torch.sum(y_true * y_pred)  # (1, 1)
    sn = tp / (fn + tp + eps)

    er = 1 - (sp + sn) / 2
    return er


def get_accuracy(y_true, y_pred):
    total = y_pred.size(0)
    correct = (y_true == y_pred).sum().item()
    return 100 * correct / total
