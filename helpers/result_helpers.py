import matplotlib.pyplot as plt


def show_training_plots(model_number, train_history_dict,
                        show_accuracy=False, show_lr=False):
    train_history = train_history_dict[f'model{model_number}_history']
    plt.plot(train_history['hist_metrics_train'], color="blue")
    plt.plot(train_history['hist_metrics_val'], color="orange")
    plt.legend(['train_metric', 'val_metric'])
    plt.xlabel('epochs')
    plt.show()

    plt.plot(train_history['hist_losses_train'])
    plt.plot(train_history['hist_losses_val'])
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epochs')
    plt.show()

    if show_accuracy:
        plt.plot(train_history['hist_accuracy_train'])
        plt.plot(train_history['hist_accuracy_val'])
        plt.legend(['train_accuracy', 'val_accuracy'])
        plt.xlabel('epochs')
        plt.show()

    if show_lr:
        plt.plot(train_history['hist_lr_all_batchs'])
        plt.legend(['train_learning_rate'])
        plt.xlabel('batchs')
        plt.show()


def print_hyper_parameters(config: dict):
    for item in config:
        print(f'{item} = {config[item]}')


def merge_two_dict(dict1, dict2):
    return dict1.update(dict2)


def merge_all_dicts(dict_list):
    full_dict = dict_list[0]
    for i in range(1, len(dict_list)):
        dict2 = dict_list[i]
        merge_two_dict(full_dict, dict2)
    return full_dict


def get_best_models_5cv(model_number, search_hist):
    search_err = {}
    for key, value in search_hist.items():
        if f'model{model_number}' in key:
            search_err[key] = value['test_err']

    best_search = min(search_err, key=search_err.get)
    print("best_test_err:", search_hist[best_search]['test_err'])

    best_models = search_hist[best_search]['trained_models_5cv']
    best_hps = search_hist[best_search]['hps']

    return best_models, best_hps
