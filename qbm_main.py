#!/usr/bin/env python
import ssl
import argparse
import os
import random
import seaborn as sns
from collections import Counter
import time
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay)
from tqdm import tqdm
#from src.discriminative_qbm import Disc_QBM
from src.model.faster_dqbm import Disc_QBM
import src.data_loader as data_loader
import matplotlib.pyplot as plt
import src.metrics as metrics


CLUSTER = 10


def main(seed=19, n_hidden_nodes=10, solver="SA", sample_count=100,
         anneal=1000, beta_eff=1.0, epochs=3, batch_size=10, learning_rate=0.01,
         restricted=True, data_set="mnist", num_classes=2, parallelize=False, save="", name=""):

    print("Start")

    random.seed(seed)
    np.random.seed(seed)
    print("Seed is " + str(seed))

    print("Loading data...")
    if data_set == "mnist":
        train_X, train_y = data_loader.get_mnist('src/data/mnist/train-images-idx3-ubyte.gz', 'src/data/mnist/train-labels-idx1-ubyte.gz', classes=[0, 1])
        test_X, test_y = data_loader.get_mnist('src/data/mnist/t10k-images-idx3-ubyte.gz', 'src/data/mnist/t10k-labels-idx1-ubyte.gz', classes=[0, 1])
    elif data_set == "breastmnist":
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = data_loader.get_medmnist('src/data/medmnist/breastmnist.npz')
    elif data_set == "pneumoniamnist":
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = data_loader.get_medmnist('src/data/medmnist/pneumoniamnist.npz')
    elif data_set == "fashionmnist":
        train_X, train_y = data_loader.get_fashionmnist('src/data/fashionmnist/train-images-idx3-ubyte', 'src/data/fashionmnist/train-labels-idx1-ubyte', classes=[0, 1])
        test_X, test_y = data_loader.get_fashionmnist('src/data/fashionmnist/t10k-images-idx3-ubyte', 'src/data/fashionmnist/t10k-labels-idx1-ubyte', classes=[0, 1])
    elif data_set == "cifar-10":
        train_X, train_y = data_loader.get_cifar10_from_torch([3,5], samples_per_class=200, train=True)
        test_X, test_y = data_loader.get_cifar10_from_torch([3,5], samples_per_class=50, train=False)
    else:
        raise ValueError("Invalid dataset")
    print("Data loaded")

    print("Preprocessing data...")
    train_X, test_X, val_X = data_loader.preprocess_images(train_X, test_X)

    #train_y = data_loader.encode_labels_to_onehot(train_y, num_classes)
    print("Data preprocessed")

    param_string = "_se" + str(seed) + "_h" + str(n_hidden_nodes) + "_sol" + solver + "_sc" + str(sample_count) + "_b" + str(
        beta_eff) + "_e" + str(epochs) + "_bs"+ str(batch_size) +"_l" + str(learning_rate) + "_r" + str(restricted) + "_data" + data_set + "_n_" + name
    print(param_string)
    # create DQBM
    print('Creating QBM...')
    dqbm = Disc_QBM(seed=seed, epochs=epochs,
                    n_hidden_nodes=n_hidden_nodes, num_classes=num_classes,
                    solver=solver, sample_count=sample_count, anneal_steps=anneal,
                    beta_eff=beta_eff, restricted=restricted,
                    param_string=param_string, speicherort=save, dim_input=784,
                    parallelize=parallelize, use_one_hot_encoding=False)
    print('QBM created')

    # train
    print('Training QBM...')
    dqbm.train_model(train_X, train_y, test_X, test_y, batch_size=batch_size, learning_rate=learning_rate)
    print('QBM trained')

    print("Predict on test data...")
    predictions = []
    #samples_output_list = []
    for i in tqdm(range(len(test_X)), desc="Predicting on test data", ncols=80, leave=False):
       p, samples_output = dqbm.predict(test_X[i])
       predictions.append(p)
        # for sample in samples_output:
        #     samples_output_list.append(sample)

    # for i in tqdm(range(len(val_X)), desc="Predicting on val data", ncols=80, leave=False):
    #     p, samples_output = dqbm.predict(val_X[i])
    #     predictions.append(p)
        # for sample in samples_output:
        #     samples_output_list.append(sample)

    #all_possible_patterns = ["0", "1"]
    #sorted_probs = dqbm.get_result_distribution(samples_output_list, all_possible_patterns)

    #tes_y = data_loader.encode_labels_to_onehot(test_y, num_classes)
    # tt = []
    # for bro in test_y:
    #     #sample_str = ''.join(str(int(y)) for y in bro)
    #     sample_str = str(bro)
    #     tt.append(sample_str)
    # true_counts = Counter(tt)
    # total_true = sum(true_counts.values())
    # true_probs = {k: v / total_true for k, v in true_counts.items()}
    # true_probs = [true_probs.get(pattern, 0.0) for pattern in all_possible_patterns]
    # print(true_probs)
    # metrics.show_and_save_distribution([sorted_probs, true_probs], ["qbm", "true"],
    #                                    "src/experiments/integerbarssc100_nodoublenorm_nachabb" + param_string + "true_distribution.png",
    #                                    "Distribution of Test Data", all_possible_patterns, True)
    #

    acc, f1, precision, recall, auc = metrics.save_result(save + name, dqbm,
                                                          dqbm.training_history, dqbm.weight_objects,
                                                          test_y, predictions, ["healthy", "pneumonia"],
                                                          batch_size, epochs, solver, learning_rate, show_plot=False,
                                                          save=True)

    # acc, f1, precision, recall, auc = metrics.save_result(save + name, dqbm,
    #                                                       dqbm.training_history, dqbm.weight_objects,
    #                                                       val_y, predictions, ["negative", "positive"],
    #                                                       batch_size, epochs, solver, learning_rate, show_plot=False,
    #                                                       save=False)


    print("Accuracy: ", acc)
    print("F1 Score: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("AUC Score: ", auc)




if __name__ == '__main__':
    # solver = 'SA' for Simulated Annealing local simulator
    # 'BMS' for classical BoltzmannSampler as local simulator
    # 'QBSolv' for testing UQO without losing coins
    # 'DW_2000Q_6' [max hnodes==94 vnodes==21] or 'Advantage_system4.1' [max hnodes==634 vnodes==21] for D-Wave Quantum Annealer (needs coins)
    # 'FujitsuDAU' for accessing Fujitsu Digital Annealing Unit (needs coins)
    #  when using FujitsuDAU, sample_count must be a value in [16, 128]
    parser = argparse.ArgumentParser(
        description='Generate clustered datasets with outliers.')
    parser.add_argument('-hn', '--hnodes',
                        metavar='INT',
                        help='Amount of hidden units for RBM model',
                        default=2,
                        type=int)

    parser.add_argument('-lr', '--learning_rate',
                        metavar='FLOAT',
                        help='Learning rate for training',
                        default=0.4529451796571889,
                        type=float)

    parser.add_argument('-r', '--restricted',
                        help='Restricted weights between hidden nodes',
                        default=False,
                        type=bool)

    parser.add_argument('-e', '--epochs',
                        metavar='INT',
                        help='Epochs fr training',
                        default=20,
                        type=int)

    parser.add_argument('-b', '--batch_size',
                        metavar='INT',
                        help='Batchsize for training',
                        default=73,
                        type=int)

    parser.add_argument('-s', '--seed',
                        metavar='INT',
                        help='Seed for RNG',
                        default=3492574433,
                        type=int)
    parser.add_argument('-sc', '--sample_count',
                        metavar='INT',
                        help='number of samples to take from the solver, always in steps of 10:' +
                            "\'10\', \'20\', \'30\', \'40\', ..., \'1000\'",
                        default=100,
                        type=int)

    parser.add_argument('--solver',
                        help='Solver, options: \'SA\', \'DW_2000Q_6\', \'Advantage_system4.1\', \'FujitsuDAU\', '
                             '\'MyQLM\', \'BMS\'',
                        default='SA',
                        type=str)

    parser.add_argument('--data_set',
                        help='Dataset to use, options: \'mnist\', \'breastmnist\', \'pneumoniamnist\', \'fashionmnist\'',
                        default='pneumoniamnist',
                        type=str)

    parser.add_argument('--num_classes',
                        help='Number of classes in dataset',
                        default=2,
                        type=int)

    parser.add_argument('--parallelize',
                        help='Use Parallel Sampling',
                        default=True,#124886.45873963833
                        type=bool)#24745.96


    parser.add_argument('--load_path',
                        help='Filepath to numpy file with saved weights to initialize from',
                        default="out/",
                        type=str)

    parser.add_argument('--name',
                        help='Name for run',
                        default="qucun_3492574433",#23771.56
                        #13293.167110532522

                        type=str)


    flags = parser.parse_args()
    print("Running with solver", flags.solver)
    main(epochs=flags.epochs, n_hidden_nodes=flags.hnodes, learning_rate=flags.learning_rate,
         batch_size=flags.batch_size,  solver=flags.solver, restricted=flags.restricted,
         seed=flags.seed, data_set=flags.data_set, num_classes=flags.num_classes,
         parallelize=flags.parallelize, sample_count=flags.sample_count, save=flags.load_path, name=flags.name)

