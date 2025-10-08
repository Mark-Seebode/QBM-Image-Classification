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


def main(seed=19, solver="SA", sample_count=100,
         anneal=1000, beta_eff=1.0, epochs=3, batch_size=10, learning_rate=0.01,
         restricted=True, data_set="mnist", num_classes=2, parallelize=False, save="", name="",
         pooling_size=4, pooling_type="probabilistic", hidden_bias_type="shared",
         one_hot=False):

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
    elif data_set == "cifar-10":
        train_x, train_y = data_loader.get_cifar10_from_torch([3, 5], samples_per_class=200, train=True)
        test_x, test_y = data_loader.get_cifar10_from_torch([3, 5], samples_per_class=50, train=False)
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

    param_string = (
        f"_se{seed}_sol{solver}_sc{sample_count}_b{beta_eff}"
        f"_e{epochs}_bs{batch_size}_l{learning_rate}_r{restricted}"
        f"_data{data_set}_n_{name}"
    )
    print(param_string)


    print('Creating QBM...')
    qbm = Conv_Deep_QBM(
        num_visible_nodes=num_visible_nodes,
        num_lable_nodes=num_label_nodes,
        image_shape=image_shape,
        seed=seed,
        kernel_size=3,
        pooling_size=pooling_size,
        pooling_type=pooling_type,   # "probabilistic" | "deterministic"
        stride=1,
        sequential_layer_sizes=[4],
        param_string=param_string,
        load_path="",
        speicherort=save,
        is_restricted=bool(restricted),
        hidden_bias_type=hidden_bias_type,
        solver=solver,
        anneal=anneal,
        token="",
    )
    print('QBM created')


    print('Training QBM...')
    epoch_loss_list = train_model(qbm, train_x, train_y, batch_size, epochs, learning_rate, sample_count, beta_eff, one_hot=one_hot)
    print('QBM trained')


    print("Predict on test data...")
    predictions = []
    probs_all = []
    for i in tqdm(range(len(test_x)), desc="Predicting on test data", ncols=80, leave=False):
        run = run_unclamped(
            qbm, test_x[i],
            num_reads=int(sample_count), beta_eff=float(beta_eff),
            one_hot=bool(one_hot)
        )
        pred = int(np.argmax(run.probs))
        predictions.append(pred)
        probs_all.append(run.probs)


    acc = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions, average="binary" if num_classes == 2 else "macro")
    precision = precision_score(test_y, predictions, average="binary" if num_classes == 2 else "macro")
    recall = recall_score(test_y, predictions, average="binary" if num_classes == 2 else "macro")

    if num_label_nodes == 1:
        pos_scores = np.array([p[1] for p in probs_all])
        auc = roc_auc_score(test_y, pos_scores)
    else:
        # macro-average AUC with one-vs-rest
        from sklearn.preprocessing import label_binarize
        Y_true = label_binarize(test_y, classes=list(range(num_classes)))
        auc = roc_auc_score(Y_true, np.stack(probs_all, axis=0), average="macro", multi_class="ovr")

    metrics.get_nll_func_per_batch(epoch_loss_list, show_plot=True)
    cm = confusion_matrix(test_y, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix ({data_set})")
    plt.tight_layout()
    plt.show()

    print("Accuracy: ", acc)
    print("F1 Score: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("AUC Score: ", auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Conv-Deep-QBM experiments.')


    parser.add_argument('-lr', '--learning_rate',
                        default=0.01,
                        type=float,
                        help='Learning rate for training')

    parser.add_argument('-r', '--restricted',
                        default=False,
                        type=bool,
                        help='Restricted weights between hidden nodes')

    parser.add_argument('-e', '--epochs',
                        default=20,
                        type=int,
                        help='Epochs for training')

    parser.add_argument('-b', '--batch_size',
                        default=3,
                        type=int,
                        help='Batchsize for training')

    parser.add_argument('-s', '--seed',
                        default=44,
                        type=int,
                        help='Seed for RNG')

    parser.add_argument('-sc', '--sample_count',
                        default=100,
                        type=int,
                        help='Number of samples to take from the solver (reads)')

    parser.add_argument('--anneal',
                        default=1000,
                        type=int,
                        help='Num sweeps for SA (ignored for QPU)')

    parser.add_argument('--solver',
                        default='SA',
                        type=str,
                        help="Solver: 'SA' or a D-Wave solver name (e.g., 'Advantage_system4.1')")

    parser.add_argument('--data_set',
                        default='mnist',
                        type=str,
                        help="Dataset: 'mnist', 'breastmnist', 'pneumoniamnist', 'fashionmnist', 'cifar-10'")

    parser.add_argument('--num_classes',
                        default=2,
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
                        action='store_true',
                        help='Use multi-node one-hot output (vs single-node binary)')

    flags = parser.parse_args()
    print("Running with solver", flags.solver)

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
        save=flags.save,
        name=flags.name,
        pooling_size=flags.pooling_size,
        pooling_type=flags.pooling_type,
        hidden_bias_type=flags.hidden_bias_type,
        one_hot=flags.one_hot,
    )

