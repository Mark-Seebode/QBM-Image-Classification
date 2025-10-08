import torch

import numpy as np
from src.ClassificationRBM import ClassificationRBM
from src import data_loader
import pickle


import argparse
USE_GPU = False

parser = argparse.ArgumentParser(description='classification_model text classificer')

parser.add_argument('--lr', type=float, default=0.08714599435919934, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train [default: 256]')
parser.add_argument('--batch-size', type=int, default=7, help='batch size for training [default: 64]')
parser.add_argument('--early-stop', type=int, default=15,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('--visible-units', type=int, default=784, help='Number of dimensions in input')
parser.add_argument('--hidden-units', type=int, default=9, help='Number of dimensions of the hidden representation')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable the gpu')
parser.add_argument('--cd-k', type=int, default=1, help='The K in the contrastive Divergence Algorithm')
parser.add_argument('--type', type=str, default='discriminative',
                    help='The type of training you want to start - discriminative, hybrid and generative')
parser.add_argument('--sparsity-coeffectient', type=float, default=0.00,
                    help='The amount that must be subtracted from bias after every update')
parser.add_argument('--data-folder', type=str, default='data', help='Folder in which the data needs to be stored')
parser.add_argument('--generative-factor', type=int, default=0.01)

args = parser.parse_args()

seeds = [1967690937, 2286980494, 3620295971, 1662044193, 1825595160, 3054779705, 900327972, 1620954898,
         3699850877, 3492574433]
acc_per_seed = []
auc_per_seed = []
for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)

    DATA_FOLDER = 'data/mnist'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    CUDA_DEVICE = 0

    if args.cuda:
        torch.cuda.set_device(CUDA_DEVICE)

    print("Loading data...")
    #train_x, train_y = data_loader.get_fashionmnist('src/data/fashionmnist/train-images-idx3-ubyte', 'src/data/fashionmnist/train-labels-idx1-ubyte', classes=[0, 1])
    #test_X, test_y = data_loader.get_fashionmnist('src/data/fashionmnist/t10k-images-idx3-ubyte', 'src/data/fashionmnist/t10k-labels-idx1-ubyte', classes=[0, 1])
    # train_x, train_y = data_loader.get_mnist('src/data/mnist/train-images-idx3-ubyte.gz',
    #                                          'src/data/mnist/train-labels-idx1-ubyte.gz', classes=[0, 1])
    # test_X, test_y = data_loader.get_mnist('src/data/mnist/t10k-images-idx3-ubyte.gz',
    #                                        'src/data/mnist/t10k-labels-idx1-ubyte.gz', classes=[0, 1])

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = data_loader.get_medmnist('src/data/medmnist/breastmnist.npz')
    print("Data loaded")
    print("Train train_x shape: ", train_X.shape)
    train_X, val_X, test_X = data_loader.preprocess_images(train_X, val_X, test_X)

    rbm = ClassificationRBM(args.visible_units, args.hidden_units, args.cd_k, num_classes=2, learning_rate=args.lr, use_cuda=False, seed=seed)
    device = rbm.get_device(USE_GPU)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    train_y = torch.from_numpy(train_y).to(device).long()
    test_y = torch.from_numpy(test_y).to(device).long()

    training_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_X).to(device), train_y)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_X).to(device), test_y)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(training_dataset), shuffle=False)

    loss_list, best_model, nll_list = rbm.train_rbm( train_loader=train_loader, test_loader=test_loader, epochs=args.epochs, method=args.type, generative_factor=args.generative_factor)

    with open(f'Breast_beta_hyper_RBM/acc_per_epoch{seed}.pkl', 'wb') as f:
        pickle.dump(rbm.acc_per_epoch_list, f)

    with open(f'Breast_beta_hyper_RBM/auc_per_epoch{seed}.pkl', 'wb') as f:
        pickle.dump(rbm.auc_per_epoch_list, f)

    #acc, auc = rbm.run_test_set(test_loader)
    #acc_per_seed.append(acc)
    #auc_per_seed.append(auc)
    # with open(f'RBM_results/acc_hnodes20_{seed}.pkl', 'wb') as f:
    #     pickle.dump(acc, f)
    #
    #
    #
    # with open(f'RBM_results/auc_hnodes20_{seed}.pkl', 'wb') as f:
    #     pickle.dump(auc, f)
#avg_acc = np.mean(acc_per_seed)
#avg_auc = np.mean(auc_per_seed)
#print("Average acc: ", avg_acc)
#print("Average auc: ", avg_auc)

#with open(f'RBM_performance_per_epoch/acc_per_epoch{seed}.pkl', 'wb') as f:
 #   pickle.dump(avg_acc, f)

#with open(f'RBM_performance_per_epoch/auc_per_epoch{seed}.pkl', 'wb') as f:
 #   pickle.dump(avg_auc, f)

