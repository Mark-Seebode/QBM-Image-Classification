import argparse
from collections import defaultdict

from sklearn.metrics import (accuracy_score, roc_auc_score)
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





def run_slurm_with_hyperparams(learning_rate, conv_learning_rate, batch_size, sample_count,  seed, kernel_size, num_kernels, sequential_layer_sizes, restricted): #restricted, anneal):
    """Submits a SLURM job with hyperparameters and waits for it to complete."""

    shell_script = "slurm_hyperparam_single_seed.sh"
    slurm_command = [
        "sbatch", shell_script,
        str(learning_rate),
        str(conv_learning_rate),
        str(batch_size),
        str(sample_count),
        str(seed),
        str(kernel_size),
        str(num_kernels),
        str(restricted),
        *map(str, sequential_layer_sizes),
    ]

    print("Running running slurm command")
    job_submission = subprocess.run(slurm_command, capture_output=True, text=True)
    print(f"Submitted SLURM job:\n{job_submission.stdout}")
    print(f"Error (if any):\n{job_submission.stderr}")

    # Extract SLURM job ID
    job_id = job_submission.stdout.strip().split()[-1]  # Last word is the job ID
    return job_id

def wait_for_slurm_jobs(job_ids):
    """Wait for all SLURM jobs to finish before proceeding."""
    print(f"Waiting for SLURM jobs to finish: {job_ids}")
    while True:
        cmd = f"squeue -u seebode -h -o \"%i\""
        out = subprocess.getoutput(cmd)
        running = out.split()
        if not any(jid in running for jid in job_ids):
            print("All SLURM jobs have completed.")
            break
        time.sleep(10)  # Check every 10 seconds


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
    #global CONV_LEARNING_RATE
    global KERNEL_SIZE
    global NUM_KERNELS
    global SEQUENTIAL_LAYER_SIZES
    #global IS_RECURRENT_WEIGHTS
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
        #CONV_LEARNING_RATE = wandb.config.conv_learning_rate
        KERNEL_SIZE = wandb.config.kernel_size
        NUM_KERNELS = wandb.config.num_kernels
        SEQUENTIAL_LAYER_SIZES = wandb.config.sequential_layer_sizes
        #IS_RECURRENT_WEIGHTS = wandb.config.is_recurrent_weights
        RESTRICTED = wandb.config.restricted
        #ANNEAL = wandb.config.anneal
        SAMPLE_COUNT = wandb.config.sample_count

    else:
        # Set default hyperparameters
        BATCH_SIZE = args.batch_size
        LEARNING_RATE = args.learning_rate
        #CONV_LEARNING_RATE = args.conv_learning_rate
        KERNEL_SIZE = args.kernel_size
        NUM_KERNELS = args.num_kernels
        SEQUENTIAL_LAYER_SIZES = args.sequential_layer_sizes
        #IS_RECURRENT_WEIGHTS = args.is_recurrent_weights
        RESTRICTED = args.restricted
        #ANNEAL = args.anneal
        SAMPLE_COUNT = args.sample_count


        return ("_b" + str(BATCH_SIZE) + "_l" + str(LEARNING_RATE))

#globalcounter = 0



def main(args, resume=False, resume_id=""):
    print("Starting current sweep")

    # start run
    if HYPERPARAM_OPT:
        if resume:
            run = wandb.init(group=SWEEP_ID, id=resume_id, resume="must")
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


        seeds = [12995138, 88139577, 37523562, 87634854, 265871164, 210619836, 83934544, 55626886, 17682686, 11087010]
        # if resume:
        #     for seed in seeds:
        #         result_file = args.path + f"sweep_{counter}-seed_{seed}acc_auc.pkl"
        #         if os.path.exists(result_file):
        #             with open(result_file, "rb") as f:
        #                 acc, auc = pickle.load(f)
        #             combined_acc_auc = 0.5 * acc + 0.5 * auc
        #             metrics_for_all_seeds[0].append(acc)
        #             metrics_for_all_seeds[1].append(auc)
        #             metrics_for_all_seeds[2].append(combined_acc_auc)
        #             print(f"Loaded results for seed {seed}: ACC={acc}, AUC={auc}, Combined={combined_acc_auc}")
        #         else:
        #             raise FileNotFoundError(f"Result file not found: {result_file}")
        # else:
        job_list = []
        epoch_data = defaultdict(lambda: {
            'acc_val': [],
            'auc_val': [],
        })


        print("Submitting SLURM jobs for all seeds")
        for seed in seeds:
            job_id = run_slurm_with_hyperparams(LEARNING_RATE, LEARNING_RATE, BATCH_SIZE, SAMPLE_COUNT, seed, KERNEL_SIZE, NUM_KERNELS, SEQUENTIAL_LAYER_SIZES, RESTRICTED)#, ANNEAL)#RESTRICTED, ANNEAL)
            job_list.append(job_id)
        print(job_list)
        time.sleep(10)
        wait_for_slurm_jobs(job_list)
        time.sleep(10)

        print("All SLURM jobs completed. Collecting results.")
        for seed in seeds:
            param_string =  "-seed_" + str(seed)


            acc_list_file = args.path  +  f"acc_per_epoch{seed}.pkl"
            auc_list_file = args.path    +  f"auc_per_epoch{seed}.pkl"
            if os.path.exists(acc_list_file) and os.path.exists(auc_list_file):
                with open(acc_list_file, "rb") as f:
                    acc_list = pickle.load(f)

                with open(auc_list_file, "rb") as f:
                    auc_list = pickle.load(f)

                for epoch in range(20):
                    epoch_data[epoch]['acc_val'].append(acc_list[epoch])
                    epoch_data[epoch]['auc_val'].append(auc_list[epoch])

            else:
                raise FileNotFoundError(f"Result file not found: {acc_list_file} or {auc_list_file}")

        folder_path = os.path.dirname(args.path)
        shutil.rmtree(folder_path)

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
                        default="out/slurm/",
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

    parser.add_argument('--sweep_id', type=str, default="voq6cfrj") #v8vwy5dq breast mnistk3i5g39d current rbm penumonia estex4pi   current sq qbm xzvm3exu
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
        main_with_args = partial(main, args)
        print("Starting sweeping")
        #main(args, True, "09zz7ww6")
        #main(args, True, "8jzjsoe3")
        wandb.agent(sweep_id=sweep_id_path, function=main_with_args,
                    count=100)

    else:
        main(args)







