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