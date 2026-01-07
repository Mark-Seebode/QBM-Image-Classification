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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y', 't'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def main(seed=19, solver="SA", sample_count=100,
         anneal=1000, beta_eff=2.0, epochs=3, batch_size=10, learning_rate=0.01, conv_learning_rate=None,
         restricted=True, data_set="mnist", num_classes=2, parallelize=False, save="", name="",
         kernel_size=5, num_kernels=10, sequential_layer_sizes=[16, 8], is_recurrent_weights=False,
         pooling_size=4, pooling_type="probabilistic", hidden_bias_type="shared",
         one_hot=False, test_on_val=False ):

    print("Start")
    random.seed(seed)
    np.random.seed(seed)
    print("Seed is", seed)


    print("Loading data...")
    if data_set == "mnist":
        train_x, train_y = data_loader.get_mnist('src/data/mnist/train-images-idx3-ubyte.gz',
                                                 'src/data/mnist/train-labels-idx1-ubyte.gz', classes=[0, 1]
                                                 , samples_per_class=50)
        test_x, test_y = data_loader.get_mnist('src/data/mnist/t10k-images-idx3-ubyte.gz',
                                               'src/data/mnist/t10k-labels-idx1-ubyte.gz', classes=[0, 1],
                                               samples_per_class=20)
    elif data_set == "breastmnist":
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = data_loader.get_medmnist(
            'src/data/medmnist/breastmnist.npz')
        if test_on_val:
            test_x, test_y = val_x, val_y
    elif data_set == "pneumoniamnist":
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = data_loader.get_medmnist(
            'src/data/medmnist/pneumoniamnist.npz')
        if test_on_val:
            test_x, test_y = val_x, val_y
    elif data_set == "fashionmnist":
        train_x, train_y = data_loader.get_fashionmnist('src/data/fashionmnist/train-images-idx3-ubyte',
                                                        'src/data/fashionmnist/train-labels-idx1-ubyte', classes=[0, 1],
                                                        samples_per_class=100)
        test_x, test_y = data_loader.get_fashionmnist('src/data/fashionmnist/t10k-images-idx3-ubyte',
                                                      'src/data/fashionmnist/t10k-labels-idx1-ubyte', classes=[0, 1],
                                                      samples_per_class=50)
    elif data_set == "miniimagenet":
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = data_loader.get_imagenet(
            root="/Users/markseebode/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1",
            classes=["n02795169", "n03417042"],)
        if test_on_val:
            test_x, test_y = val_x, val_y

    elif data_set == "NEU-CLS-64": # /home/s/seebode/BIG/
        train_x, train_y, val_x, val_y, test_x, test_y = data_loader.get_NEU_CLS_64("/home/s/seebode/BIG/data/NEU-CLS-64",
                                                                      classes=["gg", "rp"], seed=seed,
                                                                        image_size=(28, 28), contrast_factor=1.5)
        if test_on_val:
            test_x, test_y = val_x, val_y
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

    try:
        with open("src/secrets/luna_token.txt", "rb") as f:
            api_token = f.read().strip().decode("utf-8")

        with open("src/secrets/luna_group_token.txt", "rb") as f:
            groupQpuToken_name = f.read().strip().decode("utf-8")

        with open("src/secrets/dwave_key.txt", "rb") as f:
             dwave_token = f.read().strip().decode("utf-8")
    except:
        api_token = ""
        groupQpuToken_name = ""
        dwave_token = ""


    print('Creating QBM...')
    qbm = Conv_Deep_QBM(
        num_visible_nodes=num_visible_nodes,
        num_lable_nodes=num_label_nodes,
        image_shape=image_shape,
        seed=seed,
        kernel_size=kernel_size,
        pooling_size=pooling_size,
        pooling_type=pooling_type,   # "probabilistic" | "deterministic"
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
        api_token=api_token,
        dwave_token=dwave_token,
        num_reads=sample_count,
        groupQpuToken_name=groupQpuToken_name,
        example_image=train_x[0],
        parallelize=bool(parallelize),
        centerize=False
    )

    print('QBM created with:\n'
          f'  active hidden nodes: {qbm.num_hidden_units_per_layer}\n'
          f'  label nodes: {qbm.num_label_nodes}\n'
          f'  total hidden nodes: {qbm.num_hidden_nodes}\n'
          f'  num params: {qbm.count_parameters()}\n')
    # import matplotlib.pyplot as plt
    # # plot initial kernel
    # plt.figure()
    # for k in range(num_kernels):
    #     plt.subplot(1, num_kernels, k+1)
    #     plt.imshow(qbm.kernel_weights[k], cmap='gray')
    #     plt.title(f'Initial Kernel {k+1}')
    #     plt.axis('off')
    # plt.show()



    print('Training QBM...')
    epoch_loss_list, acc_list, auc_list, kernel_change_history = train_model(qbm, train_x, train_y, batch_size, epochs, learning_rate, sample_count, beta_eff, conv_learning_rate=conv_learning_rate, one_hot=one_hot, test_x=test_x, test_y=test_y)
    #qbm.save_weights()
    print('QBM trained')

    with open(save + f"acc_per_epoch{seed}.pkl", "wb") as f:
        pickle.dump(acc_list, f)

    with open(save + f"auc_per_epoch{seed}.pkl", "wb") as f:
        pickle.dump(auc_list, f)

    #import matplotlib.pyplot as plt

    # line plot of epochsloss
    # plt.figure()
    # plt.plot(range(len(epoch_loss_list)), epoch_loss_list)
    # plt.title('Training Loss per Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.show()
    #
    # # plot trained kernels
    # plt.figure()
    # for k in range(num_kernels):
    #     plt.subplot(1, num_kernels, k+1)
    #     plt.imshow(qbm.kernel_weights[k], cmap='gray')
    #     plt.title(f'Trained Kernel {k+1}')
    #     plt.axis('off')
    # plt.show()






    if solver != 'SA':
        print("Total QPU time used (microseconds):", qbm.sampler.qpu_time_used)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Conv-Deep-QBM experiments.')


    parser.add_argument('-lr', '--learning_rate',
                        default=0.005,
                        type=float,
                        help='Learning rate for training')

    parser.add_argument('-clr', '--conv_learning_rate',
                        default=0.005,
                        type=float,
                        help='Learning rate for training')

    parser.add_argument('-r', '--restricted',
                        default="True",
                        type=str2bool,
                        help='Restricted weights between hidden nodes')

    parser.add_argument('-e', '--epochs',
                        default=20,
                        type=int,
                        help='Epochs for training')

    parser.add_argument('-b', '--batch_size',
                        default=2,
                        type=int,
                        help='Batchsize for training')

    parser.add_argument('-s', '--seed',
                        default=23094922,
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
                        default=2,
                        type=int,
                        help='Number of classes in dataset')

    parser.add_argument('--parallelize',
                        default=True,
                        type=bool,
                        help='NOT IMPLEMENTED YET')

    parser.add_argument('--save',
                        default='out/slurm/',
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
                        default=5,
                        type=int,
                        help='number of convolutional kernels')

    parser.add_argument("--sequential_layer_sizes",
                        type=int,
                        nargs="+",
                        default=[24, 16, 8],
                        help="List of sequential layer sizes",
    )

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
                        default=
                        False,
                        help='Use multi-node one-hot output (vs single-node binary)')

    parser.add_argument('--test_on_val',
                        default=True,
                        type=bool,
                        help='Test either on validation set (if available) instead of test set')

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
        conv_learning_rate=flags.conv_learning_rate,
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
        test_on_val=flags.test_on_val,
    )



# TODO:
#  - fix probabilistic pooling bug
#  - refactor in the middle and keep clean
#  - train test val split


