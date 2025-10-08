import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import numpy as np
import seaborn as sns
import torch
import src.model.discriminative_qbm as DQBM
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


class History:
    def __init__(self, loss_per_batch: list[float], loss_per_epoch: list[float], nll_per_batch: list[float],
                 nll_per_epoch: list[float], acc_per_epoch: list[float], auc_per_epoch: list[float], combined_acc_auc_per_epoch: list[float]):
        self.errors_per_batch = loss_per_batch
        self.error_per_epoch = loss_per_epoch
        self.nll_per_batch = nll_per_batch
        self.nll_per_epoch = nll_per_epoch
        self.distribution_per_epoch = []
        self.acc_per_epoch = acc_per_epoch
        self.auc_per_epoch = auc_per_epoch
        self.combined_acc_auc_per_epoch = combined_acc_auc_per_epoch


class Plots:
    def __init__(self, conf_matrix_fig: plt.Figure, loss_per_batch_fig: plt.Figure, loss_per_epoch_fig: plt.Figure,
                 nll_per_batch_fig: plt.Figure, nll_per_epoch_fig: plt.Figure):
        self.conf_matrix_fig = conf_matrix_fig
        self.loss_per_batch_fig = loss_per_batch_fig
        self.loss_per_epoch_fig = loss_per_epoch_fig
        self.nll_per_batch_fig = nll_per_batch_fig
        self.nll_per_epoch_fig = nll_per_epoch_fig



def get_loss_func_per_batch(loss_history_per_batch: list[float], show_plot=False):
    fig, ax = plt.subplots()
    ax.set_title("Average output node bias error per batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Average output node bias error")

    stage1_len = np.min((len(loss_history_per_batch)))
    stage1_x = np.linspace(1, stage1_len, stage1_len)
    stage1_y = loss_history_per_batch[:stage1_len]

    ax.plot(stage1_x, stage1_y, color="orange")

    if (show_plot):
        plt.show()

    return fig


def get_loss_func_per_epoch(loss_history_per_epoch: list[float], show_plot=False):
    fig, ax = plt.subplots()
    ax.set_title("Average output node bias error per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average output node bias error")

    stage1_len = np.min((len(loss_history_per_epoch)))
    stage1_x = np.linspace(1, stage1_len, stage1_len)
    stage1_y = loss_history_per_epoch[:stage1_len]

    ax.plot(stage1_x, stage1_y, color="purple")

    if (show_plot):
        plt.show()
    return fig


def get_nll_func_per_batch(nll_history_per_batch: list[float], show_plot=False):
    fig, ax = plt.subplots()
    ax.set_title("Negative Log Likelihood per batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Negative Log Likelihood")

    stage1_len = np.min((len(nll_history_per_batch)))
    stage1_x = np.linspace(1, stage1_len, stage1_len)
    stage1_y = nll_history_per_batch[:stage1_len]

    ax.plot(stage1_x, stage1_y, color="orange")

    if (show_plot):
        plt.show()

    return fig


def get_nll_func_per_epoch(nll_history_per_epoch: list[float], show_plot=False):
    fig, ax = plt.subplots()
    ax.set_title("Negative Log Likelihood per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Likelihood")

    stage1_len = np.min((len(nll_history_per_epoch)))
    stage1_x = np.linspace(1, stage1_len, stage1_len)
    stage1_y = nll_history_per_epoch[:stage1_len]

    ax.plot(stage1_x, stage1_y, color="purple")

    if (show_plot):
        plt.show()
    return fig


def get_plots(history: History, y, y_predict, class_titels, show_plot=False) -> Plots:
    conf_matrix= get_confusion_matrix(y, y_predict, class_titels, show_plot)
    loss_per_batch_fig = get_loss_func_per_batch(history.errors_per_batch, show_plot)
    loss_per_epoch_fig = get_loss_func_per_epoch(history.error_per_epoch, show_plot)
    nll_per_batch_fig = get_nll_func_per_batch(history.nll_per_batch, show_plot)
    nll_per_epoch_fig = get_nll_func_per_epoch(history.nll_per_epoch, show_plot)

    return Plots(conf_matrix, loss_per_batch_fig, loss_per_epoch_fig, nll_per_batch_fig, nll_per_epoch_fig)


def get_confusion_matrix(y, y_predict, classes_titels=None, show_plot=False):
    classes = np.unique(np.concatenate((y, y_predict)))
    cm = confusion_matrix(y, y_predict, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes if classes_titels is None else classes_titels)

    fig, ax = plt.subplots()

    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("Confusion Matrix for QBM Classifier")

    if show_plot:
        plt.show()

    return fig


def show_and_save_distribution(sorted_probs: list, x_ticks: list, file_path: str, title: str, legend_labels: list, save=False):
    num_probs = len(sorted_probs)
    num_categories = len(sorted_probs[0])
    bar_width = 0.6

    x_positions = np.arange(num_probs)
    stacked_values = np.array(sorted_probs).T

    colors = ["blue", "orange", "pink", "green"]

    bottom = np.zeros(num_probs)

    plt.figure(figsize=(8, 6))

    for i in range(num_categories):
        plt.bar(
            x_positions, stacked_values[i], width=bar_width,
            label=legend_labels[i] if i < len(legend_labels) else f"Category {i + 1}",
            bottom=bottom, color=colors[i]
        )
        bottom += stacked_values[i]

    plt.xticks(x_positions, x_ticks, rotation=45, ha="right")

    plt.xlabel("Time of Acquiring Distribution")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(0, 1.1)

    plt.legend(title="Output Units States", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if file_path and save:
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()



def get_result_as_txt(acc, f1, precision, recall, auc_score, num_classes, input_dim, n_output_nodes, n_hidden_nodes,
                      batch_size, epochs, optimizer, learning_rate, qpu_time_used="-", beta_eff="-"):
    return (f"Accuracy: {acc}\n"
            f"AUC ROC score: {auc_score}\n"
            f"F1 Score: {f1}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"Classes: {num_classes}\n"
            f"Input Dimension: {input_dim}\n" 
            f"Number of Output Nodes: {n_output_nodes}\n"
            f"Number of Hidden Nodes: {n_hidden_nodes}\n"
            f"Batch Size: {batch_size}\n"
            f"Epochs: {epochs}\n"
            f"Optimizer: {optimizer}\n"
            f"Learning Rate: {learning_rate}\n"
            f"QPU Time Used: {qpu_time_used}\n"
            f"Beta Eff: {beta_eff}\n")


def get_metrics(y_true, y_predict, class_titles):
    acc = accuracy_score(y_true, y_predict)
    if len(class_titles) == 2:
        f1 = f1_score(y_true, y_predict, average='binary')
        precision = precision_score(y_true, y_predict, average='binary')
        recall = recall_score(y_true, y_predict, average='binary')#
        auc = roc_auc_score(y_true, y_predict)
    else:
        f1 = f1_score(y_true, y_predict, average='macro')
        precision = precision_score(y_true, y_predict, average='macro')
        recall = recall_score(y_true, y_predict, average='macro')
        auc = 0#roc_auc_score(y_true, y_predict, multi_class='ovr')
    return acc, f1, precision, recall, auc


def save_result(file_path: str, qbm: DQBM, history: History, trained_params, y_true, y_predict,
                class_titles, batch_size, epochs, optimizer, learning_rate, qpu_time_used="-", show_plot=False, save=True):

    acc, f1, precision, recall, auc = get_metrics(y_true, y_predict, class_titles)
    result_txt = get_result_as_txt(acc, f1, precision, recall, auc, class_titles, qbm.dim_input, qbm.n_output_nodes,
                                   qbm.num_conv_units, batch_size, epochs, optimizer, learning_rate,
                                   qpu_time_used=qpu_time_used,
                                   beta_eff=qbm.beta_eff)

    if save:
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump(trained_params, f)
        txtfile = open(file_path + "_result.txt", "x")
        txtfile.write(result_txt)

    with open(file_path + "acc_auc.pkl", "wb") as f:
        pickle.dump((acc, auc), f)

    if save:
        plots = get_plots(history, y_true, y_predict, class_titles, show_plot)

    if save:
        plots.loss_per_batch_fig.savefig(file_path + "_loss_per_iteration.png")
        plots.loss_per_epoch_fig.savefig(file_path + "_loss_per_epoch.png")
        plots.nll_per_batch_fig.savefig(file_path + "_nll_per_iteration.png")
        plots.nll_per_epoch_fig.savefig(file_path + "_nll_per_epoch.png")
        plots.conf_matrix_fig.savefig(file_path + "_confusion_matrix.png")

    if save:
        save_history(file_path, history)

    return acc, f1, precision, recall, auc


def save_history(file_path_and_name, history: History):
    with open(file_path_and_name + "loss_per_batch.pkl", "wb") as f:
        pickle.dump(history.errors_per_batch, f)

    with open(file_path_and_name + "loss_per_epoch.pkl", "wb") as f:
        pickle.dump(history.error_per_epoch, f)

    with open(file_path_and_name + "nll_per_batch.pkl", "wb") as f:
        pickle.dump(history.nll_per_batch, f)

    with open(file_path_and_name + "nll_per_epoch.pkl", "wb") as f:
        pickle.dump(history.nll_per_epoch, f)

    with open(file_path_and_name + "acc_per_epoch.pkl", "wb") as f:
        pickle.dump(history.acc_per_epoch, f)

    with open(file_path_and_name + "auc_per_epoch.pkl", "wb") as f:
        pickle.dump(history.auc_per_epoch, f)

    with open(file_path_and_name + "combined_acc_auc_per_epoch.pkl", "wb") as f:
        pickle.dump(history.combined_acc_auc_per_epoch, f)


def load_history(file_path_experiment_name) -> History:
    with open(file_path_experiment_name + "loss_per_batch.pkl", 'rb') as f:
        loaded_loss_per_batch = pickle.load(f)

    with open(file_path_experiment_name + "loss_per_epoch.pkl", 'rb') as f:
        loaded_loss_per_epoch = pickle.load(f)

    with open(file_path_experiment_name + "acc_per_epoch.pkl", 'rb') as f:
        loaded_acc_per_epoch = pickle.load(f)

    with open(file_path_experiment_name + "nll_per_batch.pkl", 'rb') as f:
        loaded_nll_per_batch = pickle.load(f)

    with open(file_path_experiment_name + "nll_per_epoch.pkl", 'rb') as f:
        loaded_nll_per_epoch = pickle.load(f)

    return History(loaded_loss_per_batch, loaded_loss_per_epoch, loaded_nll_per_batch, loaded_nll_per_epoch)


