#!/usr/bin/env python

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, ConfusionMatrixDisplay
)

import src.data_loader as data_loader
import src.metrics as metrics

from src.model.cdqbm_state import Conv_Deep_QBM

from src.train.pipeline import run_unclamped
from src.train.train import train_model
import pickle



def main(seed=19, solver="SA", sample_count=100,
         anneal=1000, beta_eff=1.0, epochs=3, batch_size=10, learning_rate=0.01,
         restricted=True, data_set="mnist", num_classes=2, parallelize=False, save="", name="",
         kernel_size=5, num_kernels=10, sequential_layer_sizes=[16, 8], is_recurrent_weights=False,
         pooling_size=4, pooling_type="probabilistic", hidden_bias_type="shared",
         one_hot=False, ):

    print("Start")
    random.seed(seed)
    np.random.seed(seed)
    print("Seed is", seed)


    print("Loading data...")
    if data_set == "mnist":
        train_x, train_y = data_loader.get_mnist('src/data/mnist/train-images-idx3-ubyte.gz',
                                                 'src/data/mnist/train-labels-idx1-ubyte.gz', classes=[0, 1],
                                                 samples_per_class=50)
        test_x, test_y = data_loader.get_mnist('src/data/mnist/t10k-images-idx3-ubyte.gz',
                                               'src/data/mnist/t10k-labels-idx1-ubyte.gz', classes=[0, 1],
                                               samples_per_class=20)
    elif data_set == "breastmnist":
        (train_x, train_y), (val_X, val_y), (test_x, test_y) = data_loader.get_medmnist(
            'src/data/medmnist/breastmnist.npz')
    elif data_set == "pneumoniamnist":
        (train_x, train_y), (val_X, val_y), (test_x, test_y) = data_loader.get_medmnist(
            'src/data/medmnist/pneumoniamnist.npz')
    elif data_set == "fashionmnist":
        train_x, train_y = data_loader.get_fashionmnist('src/data/fashionmnist/train-images-idx3-ubyte',
                                                        'src/data/fashionmnist/train-labels-idx1-ubyte', classes=[0, 1])
        test_x, test_y = data_loader.get_fashionmnist('src/data/fashionmnist/t10k-images-idx3-ubyte',
                                                      'src/data/fashionmnist/t10k-labels-idx1-ubyte', classes=[0, 1])
    elif data_set == "miniimagenet":
        (train_x, train_y), (val_X, val_y), (test_x, test_y) = data_loader.get_imagenet(
            root="/Users/markseebode/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1",
            classes=["n02795169", "n03417042"],)

    elif data_set == "NEU-CLS-64":
        train_x, train_y, test_x, test_y = data_loader.get_NEU_CLS_64("src/data/NEU-CLS-64",
         train_test_percentage=0.8, seed=seed, image_size=(28, 28))
    else:
        raise ValueError("Invalid dataset")
    print("Data loaded")

    print("Preprocessing data...")
    train_x, train_y = data_loader.shuffle_images(train_x, train_y, seed)
    print("Data preprocessed")

    if isinstance(train_x, np.ndarray):
        img0 = train_x[0]
        image_shape = img0.shape[:2]
    else:
        image_shape = np.asarray(train_x[0]).shape[:2]
    num_visible_nodes = int(image_shape[0] * image_shape[1])

    if num_classes == 2 and not one_hot:
        num_label_nodes = 1
        class_names = ["0", "1"]
    else:
        num_label_nodes = int(num_classes)
        class_names = [str(i) for i in range(num_classes)]

    param_string = name
    print(param_string)

    # with open("src/secrets/luna_token.txt", "rb") as f:
    #     api_token = f.read().strip().decode("utf-8")
    #
    # with open("src/secrets/luna_group_token.txt", "rb") as f:
    #     groupQpuToken_name = f.read().strip().decode("utf-8")
    #
    # with open("src/secrets/dwave_key.txt", "rb") as f:
    #      dwave_token = f.read().strip().decode("utf-8")


    print('Creating QBM...')
    qbm = Conv_Deep_QBM(
        num_visible_nodes=num_visible_nodes,
        num_lable_nodes=num_label_nodes,
        image_shape=image_shape,
        seed=seed,
        kernel_size=kernel_size,
        pooling_size=pooling_size,
        pooling_type=pooling_type,   # "probabilistic" | "deterministic"
        stride=1,
        num_filter_kernels=num_kernels,
        is_recurrent_weights=is_recurrent_weights,
        sequential_layer_sizes=sequential_layer_sizes,
        param_string=param_string,
        load_path="",
        speicherort=save,
        is_restricted=bool(restricted),
        hidden_bias_type=hidden_bias_type,
        solver=solver,
        anneal=anneal,
        #api_token=api_token,
        #dwave_token=dwave_token,
        num_reads=sample_count,
        #groupQpuToken_name=groupQpuToken_name,
        example_image=train_x[0],
        parallelize=bool(parallelize)
    )



    print('QBM created with:\n'
          f'  active hidden nodes: {qbm.num_active_units_per_layer}\n'
          f'  label nodes: {qbm.num_label_nodes}\n'
          f'  total hidden nodes: {qbm.num_hidden_nodes}\n'
          f'  num params: {qbm.count_parameters()}\n')



    print('Training QBM...')
    epoch_loss_list, acc_list, auc_list = train_model(qbm, train_x, train_y, batch_size, epochs, learning_rate, sample_count, beta_eff, one_hot=one_hot, test_x=test_x, test_y=test_y)
    qbm.save_weights()
    print('QBM trained')

    with open(os.path.join(save, f"acc_per_epoch{seed}.pkl"), "wb") as f:
        pickle.dump(auc_list, f)

    with open(os.path.join(save, f"auc_per_epoch{seed}.pkl"), "wb") as f:
        pickle.dump(auc_list, f)


    # visualize the kernel weights
    # every k is a 5x5 kernel
    # for k in qbm.kernel_weights:
    #     plt.imshow(k.reshape(qbm.kernel_weights[0].shape), cmap='gray')
    #     plt.title('Learned Kernel')
    #     plt.colorbar()
    #     plt.show()



    # print("Predict on test data...")
    # predictions = []
    # probs_all = []
    # for i in tqdm(range(len(test_x)), desc="Predicting on test data", ncols=80, leave=False):
    #     run = run_unclamped(
    #         qbm, test_x[i],
    #         beta_eff=float(beta_eff),
    #         one_hot=bool(one_hot)
    #     )
    #     pred = int(np.argmax(run.probs))
    #     predictions.append(pred)
    #     probs_all.append(run.probs)
    # print("Predictions:", predictions)


    # acc = accuracy_score(test_y, predictions)
    # f1 = f1_score(test_y, predictions, average="binary" if num_classes == 2 else "macro")
    # precision = precision_score(test_y, predictions, average="binary" if num_classes == 2 else "macro")
    # recall = recall_score(test_y, predictions, average="binary" if num_classes == 2 else "macro")
    #
    # if num_label_nodes == 1 or num_label_nodes == 2:
    #     pos_scores = np.array([p[1] for p in probs_all])
    #     auc = roc_auc_score(test_y, predictions)
    # else:
    #     # macro-average AUC with one-vs-rest
    #     from sklearn.preprocessing import label_binarize
    #     Y_true = label_binarize(test_y, classes=list(range(num_classes)))
    #     auc = roc_auc_score(Y_true, np.stack(probs_all, axis=0), average="macro", multi_class="ovr")
    #
    # loss_fig = metrics.get_nll_func_per_batch(epoch_loss_list, show_plot=True)
    # loss_fig.savefig("" + os.path.join(save, f"nll_plot{param_string}.png"))
    # print(predictions)
    # cm = confusion_matrix(test_y, predictions)
    # disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    # disp.plot(values_format="d")
    # plt.title(f"Confusion Matrix ({data_set})")
    # plt.tight_layout()
    # plt.savefig("" + os.path.join(save, f"confusion_matrix{param_string}.png"))
    # plt.show()
    #
    # print("Accuracy: ", acc)
    # print("F1 Score: ", f1)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("AUC Score: ", auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Conv-Deep-QBM experiments.')


    parser.add_argument('-lr', '--learning_rate',
                        default=0.001,
                        type=float,
                        help='Learning rate for training')

    parser.add_argument('-r', '--restricted',
                        default=True,
                        type=bool,
                        help='Restricted weights between hidden nodes')

    parser.add_argument('-e', '--epochs',
                        default=20,
                        type=int,
                        help='Epochs for training')

    parser.add_argument('-b', '--batch_size',
                        default=100,
                        type=int,
                        help='Batchsize for training')

    parser.add_argument('-s', '--seed',
                        default=3492574,
                        type=int,
                        help='Seed for RNG')

    parser.add_argument('-sc', '--sample_count',
                        default=10,
                        type=int,
                        help='Number of samples to take from the solver_backend (reads)')

    parser.add_argument('--anneal',
                        default=1000,
                        type=int,
                        help='Num sweeps for SA (ignored for QPU)')

    parser.add_argument('--solver',
                        default='SA',
                        type=str,
                        help="Solver: 'SA' or a D-Wave solver_backend name (e.g., 'Advantage_system7.1', 'Advantage2_system1.8')")

    parser.add_argument('--data_set',
                        default='NEU-CLS-64',
                        type=str,
                        help="Dataset: 'mnist', 'breastmnist', 'pneumoniamnist', 'fashionmnist', 'cifar-10', 'miniimagenet', 'NEU-CLS-64'")

    parser.add_argument('--num_classes',
                        default=9,
                        type=int,
                        help='Number of classes in dataset')
    parser.add_argument('--parallelize',
                        default=True,
                        type=bool,
                        help='NOT IMPLEMENTED YET')
    parser.add_argument('--save',
                        default='out/',
                        type=str,
                        help='Output folder prefix')

    parser.add_argument('--name',
                        default='run',
                        type=str,
                        help='Name for run')

    parser.add_argument('--kernel_size',
                        default=5,
                        type=int,
                        help='Size of the convolutional kernel')

    parser.add_argument('--num_kernels',
                        default=10,
                        type=int,
                        help='number of convolutional kernels')

    parser.add_argument('--sequential_layer_sizes',
                        default=[64, 32, 16, 8],
                        help='Number of units in each sequential layer after convolution')

    parser.add_argument('--is_recurrent_weights',
                        default=False,
                        type=bool,
                        help='Should the sequential layer be build recurrently with each kernel having its own seq layer?')

    parser.add_argument('--pooling_size',
                        default=4,
                        type=int,
                        help='Pooling window size (0/1 disables)')

    parser.add_argument('--pooling_type',
                        default='deterministic',
                        type=str,
                        help="Pooling: 'probabilistic' or 'deterministic'")

    parser.add_argument('--hidden_bias_type',
                        default='shared',
                        type=str,
                        help="Hidden bias type: 'shared', 'none', or 'per-unit'")

    parser.add_argument('--one_hot',
                        default=True,
                        help='Use multi-node one-hot output (vs single-node binary)')

    flags = parser.parse_args()
    print("Running with solver_backend", flags.solver)

    os.makedirs(flags.save, exist_ok=True)

    main(
        seed=flags.seed,
        solver=flags.solver,
        sample_count=flags.sample_count,
        anneal=flags.anneal,
        beta_eff=1.0,
        epochs=flags.epochs,
        batch_size=flags.batch_size,
        learning_rate=flags.learning_rate,
        restricted=flags.restricted,
        data_set=flags.data_set,
        num_classes=flags.num_classes,
        parallelize=flags.parallelize,
        num_kernels=flags.num_kernels,
        kernel_size=flags.kernel_size,
        sequential_layer_sizes=flags.sequential_layer_sizes,
        is_recurrent_weights=flags.is_recurrent_weights,
        save=flags.save,
        name=flags.name,
        pooling_size=flags.pooling_size,
        pooling_type=flags.pooling_type,
        hidden_bias_type=flags.hidden_bias_type,
        one_hot=flags.one_hot,
    )


# TODO:
#  - fix probabilistic pooling bug
#  - refactor in the middle and keep clean

