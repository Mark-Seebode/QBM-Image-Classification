import wandb
import argparse
import pickle

# process command line args
#parser = argparse.ArgumentParser()
#parser.add_argument("--name", type=str, required=True)
#parser.add_argument("--description", type=str, default="No description")
#args = parser.parse_args()

name = 'fully connected'

sweep_configuration = {'name': name,
                       'description': 'binary decoding 2 classes',
                       'project': "NEU-CLS-64 CDQBM", 'entity': "seebode-mark-ludwig-maximilianuniversity-of-munich",
                       'method': 'bayes',
                       'metric': {'goal': 'maximize', 'name': 'combined_acc_auc'},
                       'parameters': {'batch_size': {'values': [2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9]},
                                      #'kernel_size': {'values': [3, 5]},
                                      #'num_kernels': {'values': [1, 2, 3, 4, 5]},
                                      #'sequential_layer_sizes': {'values': [[24, 16, 8], [24, 16, 4], [24, 8, 4],
                                      #                                      [16, 8, 4], [24, 16], [24, 8], [24, 4],
                                       #                                     [16, 8], [16, 4], [8, 4], [24], [16], [8],
                                      #                                      [4]]},
                                      'n_hidden_nodes': {'values': [1,2,3,4,5,6]},
                                      #'is_recurrent_weights': {'values': [True, False]},
                                      #'restricted': {'values': [True, False]},
                                      'learning_rate': {'max': 0.5, 'min': 0.0005},
                                      #'conv_learning_rate': {'max': 0.5, 'min': 0.0005},
                                      'sample_count': {'values': list(range(10, 1010, 10))},
                                      #'anneal': {'values': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]},
                                      },
                       'early_terminate': {'type': 'hyperband', 'min_iter': 4}
                       }
def initialize_counter():
    with open("QuCUN_2hnodes/hyper_counter.pkl", "wb") as f:
        pickle.dump(1, f)

if __name__ == '__main__':
    with open("wandb_key.txt", "r") as f:
       key = f.read().strip()
    wandb.login(key=key)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="NEU-CLS-64 CDQBM", entity="seebode-mark-ludwig-maximilianuniversity-of-munich")
    print(list(range(10, 401, 10)))
    #initialize_counter()
    # Sweep generated. Sweep_id is: i12gdlp5




