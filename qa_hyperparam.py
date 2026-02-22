import argparse
from collections import defaultdict

from sklearn.metrics import (accuracy_score, roc_auc_score)
from tqdm import tqdm

import wandb
import random
import numpy as np
from functools import partial
import pickle
import matplotlib
import subprocess
import time
import src.data_loader as data_loader
import src.ClassificationRBM as ClassificationRBM
import os
import os
import shutil
from src.model import cdqbm_state
from src.model.cdqbm_state import Conv_Deep_QBM
from src.train.pipeline import run_unclamped
from src.train.train import train_model
from dwave.cloud import Client

def get_averages(list_of_lists):
    array_of_arrays = np.array(list_of_lists)
    averages = np.mean(array_of_arrays, axis=0)
    return averages

def increment_counter(counter, path):
    with open(path + "hyper_counter.pkl", "wb") as f:
        pickle.dump(counter + 1, f)


def get_counter(path):
    with open(path + "hyper_counter.pkl", 'rb') as f:
        return pickle.load(f)



def configure_hyperparams(run):
    global BATCH_SIZE
    global LEARNING_RATE
    global KERNEL_SIZE
    global NUM_KERNELS
    global SEQUENTIAL_LAYER_SIZES
    global RESTRICTED
    global ANNEAL
    global SAMPLE_COUNT

    if run:

        config_defaults = {'batch_size': args.batch_size,  'learning_rate': args.learning_rate, #'conv_learning_rate': args.conv_learning_rate,
                            'kernel_size': args.kernel_size, 'num_kernels': args.num_kernels, 'restricted': args.restricted, 'sample_count': args.sample_count,
                           'sequential_layer_sizes': args.sequential_layer_sizes,} #'is_recurrent_weights': args.is_recurrent_weights,'restricted': args.restricted,
                        #'anneal': args.anneal}

        run.config.setdefaults(config_defaults)

        BATCH_SIZE = wandb.config.batch_size
        LEARNING_RATE = wandb.config.learning_rate
        KERNEL_SIZE = wandb.config.kernel_size
        NUM_KERNELS = wandb.config.num_kernels
        SEQUENTIAL_LAYER_SIZES = wandb.config.sequential_layer_sizes
        RESTRICTED = wandb.config.restricted
        SAMPLE_COUNT = wandb.config.sample_count

    else:
        # Set default hyperparameters
        BATCH_SIZE = args.batch_size
        LEARNING_RATE = args.learning_rate
        KERNEL_SIZE = args.kernel_size
        NUM_KERNELS = args.num_kernels
        SEQUENTIAL_LAYER_SIZES = args.sequential_layer_sizes
        RESTRICTED = args.restricted
        SAMPLE_COUNT = args.sample_count

    sequential_layer_sizes_str = "_".join(map(str, SEQUENTIAL_LAYER_SIZES))


    return ("_b" + str(BATCH_SIZE) + "_l" + str(LEARNING_RATE) + "_ks" + str(KERNEL_SIZE) + "_nk" + str(NUM_KERNELS)
                + "_sls" + sequential_layer_sizes_str + "_r" + str(RESTRICTED) + "_sc" + str(SAMPLE_COUNT))

#globalcounter = 0


def pickup_seed(params_string_for_run, seed, epoch_data, dwave_token, new_run_path, client, until_epoch=20, restart_from_e=1):
    train_x, train_y, val_x, val_y, _, _ = data_loader.get_NEU_CLS_64("/home/s/seebode/BIG/data/NEU-CLS-64",
                                                                      classes=["gg", "rp"], seed=seed,
                                                                      image_size=(28, 28),
                                                                      contrast_factor=1.5)
    train_y = np.where(train_y == 0, -1, train_y)
    train_x, train_y = data_loader.shuffle_images(train_x, train_y, seed)
    param_string = params_string_for_run + "_seed" + str(seed)

    if isinstance(train_x, np.ndarray):
        img0 = train_x[0]
        image_shape = img0.shape[:2]
    else:
        image_shape = np.asarray(train_x[0]).shape[:2]
    num_visible_nodes = int(image_shape[0] * image_shape[1])
    acc_list1 = []
    auc_list1 = []

    for epoch in range(1, until_epoch + 1):
        qbm = Conv_Deep_QBM(
            num_visible_nodes=num_visible_nodes,
            num_lable_nodes=1,
            image_shape=image_shape,
            seed=seed,
            kernel_size=KERNEL_SIZE,
            pooling_size=4,
            pooling_type="deterministic",  # "probabilistic" | "deterministic"
            num_filter_kernels=NUM_KERNELS,
            is_recurrent_weights=False,
            sequential_layer_sizes=SEQUENTIAL_LAYER_SIZES,
            param_string=param_string,
            load_path="",
            stride=1,
            speicherort=new_run_path,
            is_restricted=bool(RESTRICTED),
            hidden_bias_type="shared",
            solver="Advantage2_system1.11",
            ising_or_qubo="ising",
            dwave_token=dwave_token,
            num_reads=SAMPLE_COUNT,
            example_image=train_x[0],
            parallelize=False,
            centerize=False
        )
        qbm.sampler.load_Dwave_client(client)
        qbm.load_weights(title=f'e{epoch}_seed{seed}',
                         path=f'out/{params_string_for_run}/')

        predictions = []
        probs_all = []
        for i in tqdm(range(len(val_x)), desc="Predicting on test data", ncols=80, leave=False):
            run = run_unclamped(
                qbm, val_x[i],
                beta_eff=float(1.0),
                one_hot=bool(False),
                do_conv_label_bias=True
            )
            pred = int(np.argmax(run.probs))
            predictions.append(pred)
            probs_all.append(run.probs)

        acc = accuracy_score(val_y, predictions)

        pos_scores = np.array([p[1] for p in probs_all])
        auc = roc_auc_score(val_y, pos_scores)
        acc_list1.append(acc)
        auc_list1.append(auc)

        epoch_data[epoch - 1]['acc_val'].append(acc)
        epoch_data[epoch - 1]['auc_val'].append(auc)

        client = qbm.sampler.client

    if restart_from_e > 1:
        qbm = Conv_Deep_QBM(
            num_visible_nodes=num_visible_nodes,
            num_lable_nodes=1,
            image_shape=image_shape,
            seed=seed,
            kernel_size=KERNEL_SIZE,
            pooling_size=4,
            pooling_type="deterministic",  # "probabilistic" | "deterministic"
            num_filter_kernels=NUM_KERNELS,
            is_recurrent_weights=False,
            sequential_layer_sizes=SEQUENTIAL_LAYER_SIZES,
            param_string=param_string,
            load_path="",
            stride=1,
            speicherort=new_run_path,
            is_restricted=bool(RESTRICTED),
            hidden_bias_type="shared",
            solver="Advantage2_system1.11",
            ising_or_qubo="ising",
            dwave_token=dwave_token,
            num_reads=SAMPLE_COUNT,
            example_image=train_x[0],
            parallelize=False,
            centerize=False
        )
        qbm.sampler.load_Dwave_client(client)
        qbm.load_weights(title=f'e{restart_from_e}_seed{seed}',
                         path=f'out/{params_string_for_run}/')

        epoch_loss_list, acc_list, auc_list, kernel_change_history = train_model(qbm, train_x, train_y, BATCH_SIZE,
                                                                                 20 - restart_from_e, LEARNING_RATE,
                                                                                 SAMPLE_COUNT, 1.0,
                                                                                 conv_learning_rate=LEARNING_RATE,
                                                                                 one_hot=False, test_x=val_x,
                                                                                 test_y=val_y, save_path=new_run_path,
                                                                                 restart_from_epoch=restart_from_e)

        qbm.save_weights(title=param_string, path=new_run_path)
        print('QBM trained')
        print(f"Results for seed {seed}: \nACC={acc_list}\n AUC={auc_list}")

        for i, (acc, auc) in enumerate(zip(acc_list, auc_list)):
            epoch_data[i + restart_from_e]['acc_val'].append(acc)
            epoch_data[i + restart_from_e]['auc_val'].append(auc)

        client = qbm.sampler.client

        for acc in acc_list:
            acc_list1.append(acc)
        for auc in auc_list:
            auc_list1.append(auc)

    with open(new_run_path + f"/acc_list_seed{seed}.pkl", "wb") as f:
        pickle.dump(acc_list, f)
    with open(new_run_path + f"/auc_list_seed{seed}.pkl", "wb") as f:
        pickle.dump(auc_list, f)

    return epoch_data, client




def main(args, resume=False, resume_id="6wv4w77p"):
    print("Starting current sweep")

    # start run
    if HYPERPARAM_OPT:
        if resume:
            run = wandb.init(
                project="NEU-CLS-64 CDQBM",
                entity="seebode-mark-ludwig-maximilianuniversity-of-munich",
                group=SWEEP_ID, id=resume_id, resume="must")
        else:
            run = wandb.init(reinit=True, group=SWEEP_ID)

    else:
        run = None



    params_string_for_run = configure_hyperparams(run)
    print("Params string for run: ", params_string_for_run)

    if HYPERPARAM_OPT:
        run.name = params_string_for_run

        print("Run name: ", run.name)

        num_metrics=4
        metrics_for_all_seeds = [[] for i in range(num_metrics)]


        seeds = [87634854]

        epoch_data = defaultdict(lambda: {
            'acc_val': [],
            'auc_val': [],
        })

        try:
            with open("src/secrets/dwave_key.txt", "rb") as f:
                dwave_token = f.read().strip().decode("utf-8")
        except:
            raise Exception("Error loading API tokens. Please ensure the token files exist and contain the correct tokens.")

        client = Client(token=dwave_token, solver="Advantage2_system1.11")

        # make new folder in path for this run
        new_run_path = os.path.join(args.path, params_string_for_run)
        os.makedirs(new_run_path, exist_ok=True)

        # for seed 88139577 trained model already exists, so we can load results from there and skip training for that seed
        # load model paramers per epoch run val acc and val auc for that seed and add to epoch_data
        if resume:
                epoch_data, client = pickup_seed(params_string_for_run, 88139577, epoch_data, dwave_token, new_run_path, client)
                print("Loaded results for seed 88139577")
                epoch_data, client = pickup_seed(params_string_for_run, 37523562, epoch_data, dwave_token, new_run_path, client)
                print("Loaded results for seed 37523562")
                pass

        for seed in seeds:
            train_x, train_y, val_x, val_y, _, _ = data_loader.get_NEU_CLS_64("/home/s/seebode/BIG/data/NEU-CLS-64",
                                                                              classes=["gg", "rp"], seed=seed,
                                                                              image_size=(28, 28),
                                                                              contrast_factor=1.5)
            train_y = np.where(train_y == 0, -1, train_y)
            train_x, train_y = data_loader.shuffle_images(train_x, train_y, seed)
            param_string =  params_string_for_run + "_seed" + str(seed)

            if isinstance(train_x, np.ndarray):
                img0 = train_x[0]
                image_shape = img0.shape[:2]
            else:
                image_shape = np.asarray(train_x[0]).shape[:2]
            num_visible_nodes = int(image_shape[0] * image_shape[1])

            print('Creating QBM...')
            qbm = Conv_Deep_QBM(
                num_visible_nodes=num_visible_nodes,
                num_lable_nodes=1,
                image_shape=image_shape,
                seed=seed,
                kernel_size=KERNEL_SIZE,
                pooling_size=4,
                pooling_type="deterministic",  # "probabilistic" | "deterministic"
                num_filter_kernels=NUM_KERNELS,
                is_recurrent_weights=False,
                sequential_layer_sizes=SEQUENTIAL_LAYER_SIZES,
                param_string=param_string,
                load_path="",
                stride=1,
                speicherort=new_run_path,
                is_restricted=bool(RESTRICTED),
                hidden_bias_type="shared",
                solver="Advantage2_system1.11",
                ising_or_qubo="ising",
                dwave_token=dwave_token,
                num_reads=SAMPLE_COUNT,
                example_image=train_x[0],
                parallelize=False,
                centerize=False
            )
            qbm.sampler.load_Dwave_client(client)
            print('QBM created with:\n'
                  f'  active hidden nodes: {qbm.num_hidden_units_per_layer}\n'
                  f'  label nodes: {qbm.num_label_nodes}\n'
                  f'  total hidden nodes: {qbm.num_hidden_nodes}\n'
                  f'  num params: {qbm.count_parameters()}\n')

            print('Training QBM...')
            # qbm.load_weights("e11_b22_error_backup", "out/CDQBM_QuCUN/")
            epoch_loss_list, acc_list, auc_list, kernel_change_history = train_model(qbm, train_x, train_y, BATCH_SIZE,
                                                                                     20, LEARNING_RATE,
                                                                                     SAMPLE_COUNT, 1.0,
                                                                                     conv_learning_rate=LEARNING_RATE,
                                                                                     one_hot=False, test_x=val_x,
                                                                                     test_y=val_y, save_path=new_run_path)
            qbm.save_weights(title=param_string, path=new_run_path)
            print('QBM trained')
            print(f"Results for seed {seed}: \nACC={acc_list}\n AUC={auc_list}")

            #save acc and auc list for this seed in a pickle files
            with open(new_run_path + f"/acc_list_seed{seed}.pkl", "wb") as f:
                pickle.dump(acc_list, f)
            with open(new_run_path + f"/auc_list_seed{seed}.pkl", "wb") as f:
                pickle.dump(auc_list, f)

            for epoch in range(20):
                epoch_data[epoch]['acc_val'].append(acc_list[epoch])
                epoch_data[epoch]['auc_val'].append(auc_list[epoch])

            client = qbm.sampler.client

        epochs_sorted = sorted(epoch_data.keys())
        avg_acc_list = [np.mean(epoch_data[e]['acc_val']) for e in epochs_sorted]
        avg_auc_list = [np.mean(epoch_data[e]['auc_val']) for e in epochs_sorted]

        combined_acc_auc = [0.5 * acc + 0.5 * auc for acc, auc in zip(avg_acc_list, avg_auc_list)]

        best_epoch = int(np.argmax(combined_acc_auc))

        best_acc = avg_acc_list[best_epoch]
        best_auc = avg_auc_list[best_epoch]
        best_combined = combined_acc_auc[best_epoch]

        metrics_for_all_seeds[0].append(best_acc)
        metrics_for_all_seeds[1].append(best_auc)
        metrics_for_all_seeds[2].append(best_combined)
        metrics_for_all_seeds[3].append(best_epoch)
        print(f"Loaded results: ACC={best_acc}, AUC={best_auc}, Combined={best_combined} at epoch={best_epoch}")
        print("All seeds finished")

        client.close()

        if HYPERPARAM_OPT:
            for metric_index in range(len(metrics_for_all_seeds)):
                metrics_for_all_seeds[metric_index] = get_averages(metrics_for_all_seeds[metric_index])

            wandb.log({"accuracy": metrics_for_all_seeds[0], "auc_score": metrics_for_all_seeds[1],
                       "combined_acc_auc": metrics_for_all_seeds[2], "best_epoch": metrics_for_all_seeds[3]})

            # with open(args.path + f"metrics_for_all_seeds_{counter}.txt", "w") as metrics_file:
            #     for metric_index, metric_values in enumerate(metrics_for_all_seeds):
            #         metrics_file.write(f"Metric {metric_index}: {metric_values}\n")


            run.finish()

        print("Run finished.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get optimmal cross entropy weights for chestmnist')


    parser.add_argument('-lr', '--learning_rate',
                        metavar='FLOAT',
                        help='Learning rate for the optimizer',
                        default=0.001,
                        type=float)

    parser.add_argument('-clr', '--conv_learning_rate',
                        metavar='FLOAT',
                        help='Learning rate for the optimizer',
                        default=0.001,
                        type=float)

    parser.add_argument('-e', '--epochs',
                        metavar='INT',
                        help='Number of epochs to train for',
                        default=50,
                        type=int)

    parser.add_argument('-b', '--batch_size',
                        metavar='INT',
                        help='Batch size for training',
                        default=25,
                        type=int)


    parser.add_argument('-hn', '--hnodes',
                        metavar='INT',
                        help='Number of hidden nodes for the QBM',
                        default=10,
                        type=int)

    parser.add_argument('-sc', '--sample_count',
                        metavar='INT',
                        help='Number of samples to take from the solver',
                        default=100,
                        type=int)

    parser.add_argument('-s', '--seed',
                        metavar='INT',
                        help='Seed for RNG',
                        default=42,
                        type=int)

    parser.add_argument('-nr', '--n_runs',
                        metavar='INT',
                        help='Number of runs to perform',
                        default=1,
                        type=int)

    parser.add_argument('-hpo', '--hyperparam_opt',
                        metavar='BOOL',
                        help='Whether to perform hyperparameter optimization',
                        default=True,
                        type=bool)

    parser.add_argument('--n_sweeps',
                        metavar='INT',
                        help='Number of sweeps to perform',
                        default=50,
                        type=int)

    parser.add_argument('--path',
                        metavar='STR',
                        help='Path to save the results',
                        default="out/",
                        type=str)

    parser.add_argument('--kernel_size',
                        metavar='INT',
                        help='Kernel size for convolutional layers',
                        default=3,
                        type=int)
    parser.add_argument('--num_kernels',
                        metavar='INT',
                        help='Number of kernels for convolutional layers',
                        default=2,
                        type=int)
    parser.add_argument('--sequential_layer_sizes',
                        metavar='INT',
                        help='Sizes of sequential layers',
                        nargs="+",
                        type=int,
                        default=[16, 8])
    parser.add_argument('--is_recurrent_weights',
                        metavar='BOOL',
                        help='Whether to use recurrent weights in sequential layers',
                        default=True,
                        type=bool)
    parser.add_argument('--restricted',
                        metavar='BOOL',
                        help='Whether to use restricted connections',
                        default=True,
                        type=bool)
    parser.add_argument('--anneal',
                        metavar='INT',
                        help='Number of annealing steps',
                        default=1000,
                        type=int)

    parser.add_argument('--sweep_id', type=str, default="3ldq58qy") #v8vwy5dq breast mnistk3i5g39d current rbm penumonia estex4pi   current sq qbm xzvm3exu
    parser.add_argument('--key', type=str, default=None)

    args = parser.parse_args()

    print("Starting Hyperparameter Optimization for")

    HYPERPARAM_OPT = args.hyperparam_opt
    NUM_SWEEPS = 100#args.n_sweeps

    if HYPERPARAM_OPT:
        if args.key:
            wandb.login(key=args.key)
        else:
            with open("wandb_key.txt", "r") as f:
                key = f.read().strip()
            wandb.login(key=key)
        print("Logged in to wandb")

        SWEEP_ID = args.sweep_id

        sweep_id_path = "seebode-mark-ludwig-maximilianuniversity-of-munich/NEU-CLS-64 CDQBM/" + SWEEP_ID
        print(sweep_id_path)
        #main_with_args = partial(main, args)
        print("Starting sweeping")
        #main(args, True, "6wv4w77p")
        main(args, True, "o44svo5q")
        #wandb.agent(sweep_id=sweep_id_path, function=main_with_args,
         #           count=100)

    else:
        main(args)







