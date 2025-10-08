import dimod
import torch.nn
from collections import Counter
from neal import SimulatedAnnealingSampler
import pickle
import os
import dimod as di
import random
import dwave_networkx as dnx
import numpy as np
#import boltzmann_sampler
from pathlib import Path, PurePath
import dwave_networkx
import minorminer
from matplotlib import pyplot as plt
from dwave.cloud import Client
from dwave.embedding import embed_bqm, unembed_sampleset, EmbeddedStructure
import dwave.embedding as dwave_embedding
import src.embedding as emb
from itertools import repeat
import networkx as nx
from tqdm.auto import tqdm
import src.metrics as metrics
from concurrent.futures import ProcessPoolExecutor


def make_qubo_symmetric(qubo):
    for i in range(len(qubo)):
        for j in range(i):
            qubo[i, j] = qubo[j, i] / 2
            qubo[j, i] = qubo[j, i] / 2


def solve_linear_qubo(qubo_as_bqm, sample_count):
    # no quadratic term
    assert qubo_as_bqm.quadratic == {}
    solution = {}
    for key, value in qubo_as_bqm.linear.items():
        if value < 0.0:
            solution[key] = 1.0
        elif value > 0.0:
            solution[key] = 0.0
        else:  # == 0.0
            solution[key] = random.choice([0.0, 1.0])
    solution_list = list(repeat(solution, sample_count))
    sample_set = dimod.SampleSet.from_samples_bqm(solution_list, qubo_as_bqm)
    return sample_set


class Conv_Deep_Disc_QBM():
    def __init__(self, dim_input, num_classes, image_shape=(28,28), seed=77, solver="SA",
                 sample_count=20, anneal_steps=20, beta_eff=1.0, kernel_size=3, pooling_size=0,
                 pooling_type="deterministic", stride=1, sequential_layer_sizes=[6],
                 param_string="", load_path="", speicherort=None, parallelize=False, restricted=False,
                 use_one_hot_encoding=False, hidden_bias_type="none"):

        self.seed = seed
        self.training_history = metrics.History([], [], [], [], [], [], [])

        self.dim_input = dim_input
        self.image_shape = image_shape

        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.sequential_layer_sizes = sequential_layer_sizes
        self.restricted = restricted
        self.prob_penalty = 0.8225
        self.epoch_errors = 0
        self.epoch_nll = 0

        self.parallelize = parallelize
        with open("src/secrets/TOKEN.txt", "r") as f:
            TOKEN = f.read().strip()
        print(TOKEN)
        self.TOKEN = TOKEN

        self.hidden_bias_type = hidden_bias_type
        self.use_one_hot_encoding = use_one_hot_encoding
        if use_one_hot_encoding:
            self.n_output_nodes = num_classes
        else:
            self.n_output_nodes = 1
        (self.kernel_weights, self.weights_hidden_interlayer,
         self.weights_hidden_to_output, self.weights_output_output, self.biases_conv_units,
         self.biases_sequential_units, self.biases_output) =  self.init_weights()

        self.weight_objects = [self.kernel_weights, self.weights_hidden_interlayer,
                               self.weights_hidden_to_output, self.weights_output_output, self.biases_conv_units,
                               self.biases_sequential_units, self.biases_output]

        self.param_string = param_string

        if solver == "SA":
            self.sampler = SimulatedAnnealingSampler()
            self.sa_time_used = 0
            if self.parallelize:
                self.executor = ProcessPoolExecutor(max_workers=10)
                self.sampler_0 = SimulatedAnnealingSampler()
                self.sampler_1 = SimulatedAnnealingSampler()
                self.sampler_2 = SimulatedAnnealingSampler()
                self.sampler_3 = SimulatedAnnealingSampler()
                self.sampler_4 = SimulatedAnnealingSampler()
                self.sampler_5 = SimulatedAnnealingSampler()
                self.sampler_6 = SimulatedAnnealingSampler()
                self.sampler_7 = SimulatedAnnealingSampler()
                self.sampler_8 = SimulatedAnnealingSampler()
                self.sampler_9 = SimulatedAnnealingSampler()
        elif solver == "BMS":
            self.clamped_sampler = None
            self.unclamped_sampler = None
            self.temperatures = None
        else:  # use D-Wave Advantage
            # different embeddings for each phase, because then, we can
            # reuse the initial embedding
            self.embedding_clamped = None
            self.embedding_unclamped = None

            if self.parallelize:
                self.parallel_embeddings_clamped = None
                self.parallel_embeddings_unclamped = None
                self.subgraphs = None

            self.client = Client(token=self.TOKEN, solver=solver)
            # use an Advantage solver (first generation -> with 5000 Qubits)
            self.solver = self.client.get_solver(name=solver)
            self.qpu_time_used = 0
        # number of simulated annealing steps to create one sample
        if solver == "SA" or solver == "MyQLM":
            self.anneal_steps = anneal_steps
        self.solver_string = solver
        # number of samples from Annealing (i.e. number of anneals) TODO: find good default value
        self.sample_count = sample_count
        # 1/(k_b * T) "effective Temperature" TODO: find good value / way to calculate (Müller and Adachi have 2 -> ?)
        self.beta_eff = beta_eff

        if load_path != "":
            self.load_savepoint(load_path)
        self.speicherort = speicherort

        if (self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1") and self.parallelize\
                and (self.parallel_embeddings_clamped is None or self.parallel_embeddings_unclamped is None):
            print("Calculating embeddings...")
            # self.subgraphs = self.calcualte_parallel_subgraphs()
            self.subgraphs = self.load_subgraphs("src/embeddings/integer/subgraphs.pkl")
            print("Subgraphs loaded.")

            self.parallel_embeddings_clamped = self.load_embeddings(
                f"src/embeddings/integer/embeddings_clamped_{self.num_conv_units}.pkl")

            # self.parallel_embeddings_unclamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0])
            self.parallel_embeddings_unclamped = self.load_embeddings(
                f"src/embeddings/integer/embeddings_unclamped_{self.num_conv_units}.pkl")
            print("Embeddings loaded.")




    def load_savepoint(self, savepoint):
        savepoint = Path(savepoint)
        if savepoint.exists():
            with open(savepoint, "rb") as file:
                loaded_savepoint = pickle.load(file)
        else:
            raise FileNotFoundError("Savepoint file not found")

        (self.kernel_weights,
         self.weights_hidden_interlayer,
         self.weights_hidden_to_output, self.weights_output_output, self.biases_conv_units,
         self.biases_output) = loaded_savepoint

        self.weight_objects = [self.kernel_weights,
                               self.weights_hidden_interlayer,
                               self.weights_hidden_to_output, self.weights_output_output, self.biases_conv_units,
                               self.biases_output]

    def get_input_groups_coordinates(self):
        height, width = self.image_shape
        k = self.kernel_size
        s = self.stride

        groups = []
        for i in range(0, height - k + 1, s):
            rows = np.arange(i, i + k)
            for j in range(0, width - k + 1, s):
                cols = np.arange(j, j + k)
                groups.append((rows, cols))
        return groups

    def calculate_conv_layer_dimension(self):
        height, width = self.image_shape
        k = self.kernel_size
        s = self.stride
        num_vertical = (height - k) // s + 1
        num_horizontal = (width - k) // s + 1
        output_dimension = (num_vertical, num_horizontal)

        return output_dimension

    def init_weights_input_hidden(self):
        num_full_groups = self.dim_input // self.kernel_size
        remainder = self.dim_input % self.kernel_size

        if remainder == 0:
            num_groups = num_full_groups
        else:
            num_groups = num_full_groups + 1

        weights_visible_to_hidden = np.zeros((self.dim_input, num_groups))
        input_groups = []

        for i in range(num_full_groups):
            start_idx = i * self.kernel_size
            end_idx = start_idx + self.kernel_size
            group_weights = np.random.uniform(-1, 1, self.kernel_size)
            weights_visible_to_hidden[start_idx:end_idx, i] = group_weights
            input_groups.append(np.arange(start_idx, end_idx))

        if remainder > 0:
            # Final group reuses last kernel_size inputs
            start_idx = self.dim_input - self.kernel_size
            reused_indices = np.arange(start_idx, self.dim_input)
            group_weights = np.random.uniform(-1, 1, self.kernel_size)
            weights_visible_to_hidden[reused_indices, num_groups - 1] = group_weights
            input_groups.append(reused_indices)

        return weights_visible_to_hidden, input_groups

    def init_weights_input_hidden_kernel_like(self, stride=1):
        num_groups = (self.dim_input - self.kernel_size) // stride + 1

        weights_visible_to_hidden = np.zeros((self.dim_input, num_groups))
        input_groups = []

        for i in range(num_groups):
            start_idx = i * stride
            end_idx = start_idx + self.kernel_size
            if end_idx > self.dim_input:
                break

            group_indices = np.arange(start_idx, end_idx)
            group_weights = np.random.uniform(-1, 1, self.kernel_size)

            weights_visible_to_hidden[group_indices, i] = group_weights
            input_groups.append(group_indices)

        return weights_visible_to_hidden, input_groups


    def init_weights_hidden_to_output(self, last_hidden_layer_dim: int, num_output_units: int):
        weights = np.random.uniform(-1, 1, (last_hidden_layer_dim, num_output_units))
        return weights


    def _build_pool_windows_first_hidden(self, pool_size: int, first_hidden_layer_ouput_dim):
        """Non-overlapping p×p windows on first hidden layer (layer 0)."""
        self.pool_windows, self.pooled_units = [], []
        if pool_size in (0, 1): return
        H, W = first_hidden_layer_ouput_dim
        p = pool_size
        for i in range(0, H - p + 1, p):
            for j in range(0, W - p + 1, p):
                idxs = []
                for di in range(p):
                    for dj in range(p):
                        idxs.append((i + di) * W + (j + dj))
                win = np.array(idxs, dtype=int)
                self.pool_windows.append(win)
        self.pooled_units = [int(w[0]) for w in self.pool_windows] # place holder to get number of pooled units



    def pool_units_deterministically(self, eff_bias_conv_layer):
        """select pooled_units biased on max p(h|v)"""
        # should also respect the bias of the units
        pooled_units = []
        for win in self.pool_windows:
            win_biases = eff_bias_conv_layer[win]

            max_idx = np.argmin(win_biases)
            pooled_units.append(win[max_idx])
        return pooled_units





    def init_weights(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.num_total_units = 0
        self.num_active_units = 0

        kernel_weights = np.random.uniform(-1, 1, (self.kernel_size, self.kernel_size))

        # TODO: adjust this for pooling and sequential layers
        self.num_hidden_units_per_layer = []
        self.num_active_units_per_layer = []

        input_groups = self.get_input_groups_coordinates()

        first_hidden_layer_ouput_dim = self.calculate_conv_layer_dimension()

        self.num_hidden_units_per_layer.append(len(input_groups))
        if self.pooling_type == "probabilistic":
            self.num_active_units_per_layer.append(len(input_groups))


        if getattr(self, "pooling_size", 0) not in (0, 1):
            self._build_pool_windows_first_hidden(self.pooling_size, first_hidden_layer_ouput_dim)
            self.num_hidden_units_per_layer.append(len(self.pooled_units))
            self.num_active_units_per_layer.append(len(self.pooled_units))

        self.num_conv_units = len(input_groups)
        self.conv_layer_dim = first_hidden_layer_ouput_dim
        self.input_groups = input_groups  # now 2-D (rows, cols) tuples
        self.num_total_units += self.num_conv_units
        self.num_active_units += len(self.pooled_units)

        # Inter-layer (within-layer) connections (if not restricted)
        # TODO: is interlayer connections in the conv layer good?
        # if not self.restricted:
        #     if self.pooling_type == "determinisitic":
        #         pass #weights_hidden_interlayer = np.triu(np.random.uniform(-1, 1, len(self.pooled_units)), k=1)
        #     else:
        #         weights_hidden_interlayer = np.triu(np.random.uniform(-1, 1, self.conv_layer_dim), k=1)
        # else:
        #     weights_hidden_interlayer = None



        # Sequential Layer
        self.weights_sequential_layer = []
        for i, num_units in enumerate(self.sequential_layer_sizes):
            self.weights_sequential_layer.append(np.random.uniform(-1, 1, (self.num_hidden_units_per_layer[-1], num_units)))
            self.num_hidden_units_per_layer.append(num_units)
            self.num_total_units += num_units
            self.num_active_units += num_units
            self.num_active_units_per_layer.append(num_units)
            self.num_hidden_units_per_layer.append(num_units)

        if not self.restricted:
            self.weights_interlayer_sequential = []
            for size in self.sequential_layer_sizes:
                weights = np.triu(np.random.uniform(-1, 1, size))
                self.weights_interlayer_sequential.append(weights)

        # Last hidden -> output; output -> output
        weights_hidden_to_output = self.init_weights_hidden_to_output(
            self.num_hidden_units_per_layer[-1], self.n_output_nodes
        )
        weights_output_output = np.triu(
            np.random.uniform(-1, 1, (self.n_output_nodes, self.n_output_nodes)), k=1
        )

        # Biases
        if self.hidden_bias_type == "shared":
            biases_conv_units = np.random.uniform(-1, 1, 1) #TODO: currently only one Conv filter supported
        elif self.hidden_bias_type == "none":
            biases_conv_units = np.zeros(self.sequential_layer_sizes) #TODO: not working
        else:
            biases_conv_units = np.random.uniform(-1, 1, self.num_conv_units)

        biases_sequential_units = np.random.uniform(-1, 1, sum(self.sequential_layer_sizes))

        biases_output = np.random.uniform(-1, 1, self.n_output_nodes)

        weights_hidden_interlayer = None


        return (
            kernel_weights,
            weights_hidden_interlayer,
            weights_hidden_to_output,
            weights_output_output,
            biases_conv_units,
            biases_sequential_units,
            biases_output
        )

    def prepare_hidden_bias_for_qubo(self):
        if self.hidden_bias_type == "shared":
            # deprecated only worked with old pyramid structure
            # conv_units_biases = np.array([])
            # for i in range(self.sequential_layer_sizes):
            #     hidden_biases_one_layer = np.repeat(self.biases_conv_units[i], self.num_hidden_units_per_layer[i])
            #     conv_units_biases = np.append(conv_units_biases, hidden_biases_one_layer)
            conv_units_biases = np.repeat(self.biases_conv_units[0], self.num_conv_units)
        elif self.hidden_bias_type == "none":
            conv_units_biases = np.zeros(self.num_conv_units)
        else:
            conv_units_biases = self.biases_conv_units

        return conv_units_biases

    def _conv2d_valid_stride(self, img2d: np.ndarray, kernel2d: np.ndarray, stride: int) -> np.ndarray:
        kh, kw = kernel2d.shape
        ih, iw = img2d.shape
        sh = sw = int(stride)
        out_h = (ih - kh) // sh + 1
        out_w = (iw - kw) // sw + 1
        out = np.zeros((out_h, out_w), dtype=float)
        for i in range(out_h):
            ii = i * sh
            for j in range(out_w):
                jj = j * sw
                out[i, j] = np.sum(img2d[ii:ii + kh, jj:jj + kw] * kernel2d)
        return out

    def add_first_hidden_layer_biases_to_qubo(self, qubo_matrix, input_image_2d, _unused):
        """
        Compute linear terms for layer-0 from a valid 2-D conv of the input image
        with the first kernel. Insert them on the diagonal of the first hidden block.
        """
        k2d = np.asarray(self.kernel_weights)
        s = self.stride
        fmap = self._conv2d_valid_stride(input_image_2d, k2d, s)
        flat = fmap.reshape(-1)

        n0 = self.num_hidden_units_per_layer[0]
        assert flat.size == n0, f"Conv output {flat.size} must equal first hidden units {n0}"

        block = qubo_matrix[0:n0, 0:n0]
        block.flat[::n0 + 1] += flat

        diag_mat = np.diag(flat)
        return qubo_matrix, diag_mat

    # def add_first_hidden_layer_biases_to_qubo(self, qubo_matrix, input_image, weights_input_to_first_hidden):
    #     first_hidden_layer_biases = np.matmul(input_image, weights_input_to_first_hidden)
    #     input_weights_and_input_values_to_first_hidden_layer = np.diag(first_hidden_layer_biases.flatten())
    #     qubo_matrix[0:self.num_hidden_units_per_layer[0], 0:self.num_hidden_units_per_layer[0]] \
    #         += input_weights_and_input_values_to_first_hidden_layer
    #     return qubo_matrix, input_weights_and_input_values_to_first_hidden_layer

    def prepare_clamped_qubo_pool(self, initial_qubo, label, input_image,
                                  conv_units_biases):
        if not self.use_one_hot_encoding:
            label = np.array([label])

        k2d = np.asarray(self.kernel_weights)  # (k, k)
        s = self.stride
        fmap = self._conv2d_valid_stride(input_image, k2d, s)  # shape (H', W')
        flat_fmap = fmap.reshape(-1)

        # TODO: add pooling here and different options
        if self.pooling_type == "deterministic":
            self.pooled_units = self.pool_units_deterministically(flat_fmap)
            num_conv_units = len(self.pooled_units)
        else:
            num_conv_units = self.num_conv_units

        qubo_matrix = initial_qubo.copy()

        # add image bias to pooled units
        if self.pooling_type == "deterministic":
            eff_bias_conv_layer = flat_fmap[self.pooled_units]
        else:
            eff_bias_conv_layer = flat_fmap

        eff_bias_conv_layer = np.diag(eff_bias_conv_layer)
        qubo_matrix[:num_conv_units, :num_conv_units] += eff_bias_conv_layer


        #if not self.restricted:
        #    qubo_matrix[:num_conv_units, :num_conv_units] += self.weights_hidden_interlayer #TODO are these weights also shared?

        for i, size in enumerate(self.sequential_layer_sizes):
            if self.pooling_type == "probabilistic":
                i += 1
                j = i -1
                if i >= len(self.num_active_units_per_layer):
                    continue
                start_idx = sum(self.num_active_units_per_layer[1:i]) + self.num_conv_units
            else:
                j = i
                start_idx = sum(self.num_active_units_per_layer[:i])
            end_idx = start_idx + self.num_active_units_per_layer[i]
            qubo_matrix[start_idx:end_idx, end_idx:end_idx+size] += self.weights_sequential_layer[j]
            if not self.restricted:
                qubo_matrix[end_idx:end_idx + size, end_idx:end_idx + size] += self.weights_interlayer_sequential[j]

        eff_label_bias_last_layer = np.matmul(label, np.transpose(self.weights_hidden_to_output))
        eff_label_bias_last_layer = np.diag(eff_label_bias_last_layer.flatten())

        num_units_till_last_layer = sum(self.num_active_units_per_layer[:-1])
        qubo_matrix[num_units_till_last_layer:, num_units_till_last_layer:] += eff_label_bias_last_layer

        # np.add.at(qubo_matrix, (idx, idx), bias_vec)
        if self.pooling_type == "probabilistic":
            pooled_hidden_biases = conv_units_biases
        else:
            pooled_hidden_biases = conv_units_biases[self.pooled_units]

        pooled_hidden_biases = np.diag(pooled_hidden_biases)
        qubo_matrix[:num_conv_units, :num_conv_units] += pooled_hidden_biases

        if self.pooling_type == "deterministic":
            qubo_matrix[num_conv_units:, num_conv_units:] += np.diag(self.biases_sequential_units)
        else:
            qubo_matrix[num_conv_units + len(self.pooled_units):, num_conv_units + len(self.pooled_units):] += np.diag(self.biases_sequential_units)


        qubo_matrix = qubo_matrix / self.beta_eff

        return qubo_matrix



    def prepare_clamped_qubo(self, label, input_vector, weights_input_to_first_hidden,
                             hidden_biases):
        if not self.use_one_hot_encoding:
            label = np.array([label])
        qubo_matrix = np.zeros((self.num_conv_units, self.num_conv_units))

        qubo_matrix, eff_bias_conv_layer = self.add_first_hidden_layer_biases_to_qubo(qubo_matrix, input_vector,
                                                                                      weights_input_to_first_hidden)

        if not self.restricted:
            qubo_matrix += self.weights_hidden_interlayer

        #TODO: add pooling here and different options
        self.pooled_units = self.pool_units_deterministically(eff_bias_conv_layer)




        eff_label_bias_last_layer = np.matmul(label, np.transpose(self.weights_hidden_to_output))
        #eff_label_bias_last_layer = np.diag(last_hidden_layer_biases.flatten())
        pooled_units_indexes = np.asarray(self.pooled_units, dtype=int)  # e.g. [7, 19, 23, ...]
        eff_label_bias_last_layer = np.asarray(eff_label_bias_last_layer, dtype=float)  # same length as idx

        qubo_matrix[pooled_units_indexes, pooled_units_indexes] += eff_label_bias_last_layer

        # np.add.at(qubo_matrix, (idx, idx), bias_vec)

        hidden_biases = np.diag(hidden_biases)
        qubo_matrix += hidden_biases

        qubo_matrix = qubo_matrix / self.beta_eff

        return qubo_matrix

    def prepare_unclamed_qubo_pool(self, initial_qubo, input_image,
                                   conv_units_biases):

        if self.pooling_type == "probabilistic":
            num_active_units = self.num_total_units + len(self.pooled_units)
        else:
            num_active_units = self.num_active_units

        k2d = np.asarray(self.kernel_weights)  # (k, k)
        s = self.stride
        fmap = self._conv2d_valid_stride(input_image, k2d, s)  # shape (H', W')
        flat_fmap = fmap.reshape(-1)

        # TODO: add pooling here and different options
        if self.pooling_type == "deterministic":
            self.pooled_units = self.pool_units_deterministically(flat_fmap)
            num_conv_units = len(self.pooled_units)
        else:
            num_conv_units = self.num_conv_units

        qubo_matrix = initial_qubo.copy()

        if self.pooling_type == "deterministic":
            eff_bias_conv_layer = flat_fmap[self.pooled_units]
        else:
            eff_bias_conv_layer = flat_fmap

        eff_bias_conv_layer = np.diag(eff_bias_conv_layer)
        qubo_matrix[:num_conv_units, :num_conv_units] += eff_bias_conv_layer

        # if not self.restricted:
        #    qubo_matrix[:num_conv_units, :num_conv_units] += self.weights_hidden_interlayer #TODO are these weights also shared?

        for i, size in enumerate(self.sequential_layer_sizes):
            if self.pooling_type == "probabilistic":
                i += 1
                j = i - 1
                if i >= len(self.num_active_units_per_layer):
                    continue
                start_idx = sum(self.num_active_units_per_layer[1:i]) + self.num_conv_units
            else:
                j = i
                start_idx = sum(self.num_active_units_per_layer[:i])
            end_idx = start_idx + self.num_active_units_per_layer[i]
            okok = qubo_matrix[start_idx:end_idx, end_idx:end_idx + size]
            okoko = self.weights_sequential_layer[j]
            qubo_matrix[start_idx:end_idx, end_idx:end_idx + size] += self.weights_sequential_layer[j]
            if not self.restricted:
                qubo_matrix[end_idx:end_idx + size, end_idx:end_idx + size] += self.weights_interlayer_sequential[j]

        num_units_till_last_layer = sum(self.num_active_units_per_layer[:-1])
        qubo_matrix[num_units_till_last_layer:num_active_units, num_active_units:] \
            += self.weights_hidden_to_output

        qubo_matrix[num_active_units:, num_active_units:] += self.weights_output_output


        if self.pooling_type == "probabilistic":
            pooled_hidden_biases = conv_units_biases
        else:
            pooled_hidden_biases = conv_units_biases[self.pooled_units]

        pooled_hidden_biases = np.diag(pooled_hidden_biases)
        qubo_matrix[:num_conv_units, :num_conv_units] += pooled_hidden_biases

        if self.pooling_type == "probabilistic":
            qubo_matrix[num_conv_units + len(self.pooled_units):num_active_units, num_conv_units + len(self.pooled_units):num_active_units] \
                += np.diag(self.biases_sequential_units)
        else:
            qubo_matrix[num_conv_units:num_active_units, num_conv_units:num_active_units] \
            += np.diag(self.biases_sequential_units)

        output_biases = np.diag(self.biases_output)
        qubo_matrix[num_active_units:, num_active_units:] += output_biases

        qubo_matrix = qubo_matrix / self.beta_eff

        return qubo_matrix

    def prepare_unclamed_qubo(self, input_vector, weights_input_to_first_hidden,
                              hidden_biases):
        qubo_matrix = np.zeros((self.num_conv_units + self.n_output_nodes, self.num_conv_units + self.n_output_nodes))

        qubo_matrix, eff_bias_conv_layer= self.add_first_hidden_layer_biases_to_qubo(qubo_matrix, input_vector, weights_input_to_first_hidden)

        if not self.restricted:
            qubo_matrix[:self.num_conv_units, :self.num_conv_units] += self.weights_hidden_interlayer

        #TODO: add pooling here and different options
        self.pooled_units = self.pool_units_deterministically(eff_bias_conv_layer)

        pooled_units_indexes = np.asarray(self.pooled_units, dtype=int)
        qubo_matrix[pooled_units_indexes, self.num_conv_units:] += self.weights_hidden_to_output

        qubo_matrix[self.num_conv_units:, self.num_conv_units:] += self.weights_output_output

        hidden_biases = np.diag(hidden_biases)
        qubo_matrix[:self.num_conv_units, :self.num_conv_units] += hidden_biases

        output_biases = np.diag(self.biases_output)
        qubo_matrix[self.num_conv_units:, self.num_conv_units:] += output_biases

        qubo_matrix = qubo_matrix / self.beta_eff

        return qubo_matrix

    def create_qubo_matrix_from(self, input_vector: np.ndarray, initial_qubo, label: np.ndarray = None):
        # clamped phase
        # 3 visible, 4 hidden, clamped, upper right triangular embedding_matrix
        #
        #          (hb1 + v1h1w*v1 + v2h1w*v2.......) h1h2w + h2h2w .....
        #             (hb2  + v1h2w*v1 + v2h2w*v2.......)-----
        #                (hb3 + v1h3w*v1 + v2h3w*v2.......)-----
        #                   hb4

        conv_unit_biases = self.prepare_hidden_bias_for_qubo()
        if label is not None:
            qubo_matrix = self.prepare_clamped_qubo_pool(initial_qubo, label, input_vector, conv_unit_biases)

        # unclamped_phase
        #            vb1 v1v2w v1h1w v1h2w .....
        #                 vb2 v2h1w v2h2w .....
        #                     hb1 h1h2w h2h2w    -
        #                           hb2 h2h3w    -
        #                               hb3   -
        #                                    hb4
        else:
                qubo_matrix = self.prepare_unclamed_qubo_pool(initial_qubo, input_vector, conv_unit_biases)
        return qubo_matrix


    def duplicate_and_concatenate_qubo(self, qubo_matrix, n_times):
        size = qubo_matrix.shape[0]

        zero_matrix = np.zeros((n_times * size, n_times * size))

        for i in range(n_times):
            start_idx = i * size
            end_idx = start_idx + size
            zero_matrix[start_idx:end_idx, start_idx:end_idx] = qubo_matrix

        return zero_matrix

    def sample_sa(self, sa_sampler, qubo_as_bqm, sample_count, anneal_steps, samples=None):
        sample = list(sa_sampler.sample(
            qubo_as_bqm, num_reads=sample_count, num_sweeps=anneal_steps, seed=self.seed).samples())

        if samples is not None:
            samples.put(sample)
        else:
            return sample

    @staticmethod
    def parallel_sa_sample(args):
        qubo_matrix, label, sample_count, anneal_steps, sa_sampler, seed = args
        qubo_as_bqm = di.BQM(qubo_matrix, "BINARY")
        return list(sa_sampler.sample(qubo_as_bqm, num_reads=sample_count, num_sweeps=anneal_steps, seed=seed).samples())


    def run_parralel_qa_sampling(self, combined_embedded_bqm, combined_qubo_bqm, embeddings):
        bqm = dimod.BQM(combined_embedded_bqm, "BINARY")
        try:
            sampleset = self.solver.sample_bqm(bqm, answer_mode='raw', num_reads=10).sampleset
            self.qpu_time_used += sampleset.info['timing']['qpu_access_time']

        except (
                ConnectionError, ConnectionResetError, ConnectionAbortedError,
                ConnectionRefusedError):
            self.refresh_connection()

            sampleset = self.solver.sample_bqm(bqm, answer_mode='raw', num_reads=10).sampleset
            self.qpu_time_used += sampleset.info['timing']['qpu_access_time']

        combined_embedding = {}

        new_key = 0

        for d in embeddings:
            for key, value in d.items():
                combined_embedding[new_key] = value
                new_key += 1

        answer = dwave_embedding.unembed_sampleset(sampleset, combined_embedding, combined_qubo_bqm)
        print(list(answer.samples()))


    def run_qa_sampling(self, embedded_bqm, this_embedding, source_bqm_unembedded, sample_count):
        try:
            embedded_answer = self.solver.sample_bqm(embedded_bqm,
                                                     num_reads=sample_count,
                                                     answer_mode='raw'
                                                     ).sampleset
            self.qpu_time_used += embedded_answer.info['timing']['qpu_access_time']
        except (
                ConnectionError, ConnectionResetError, ConnectionAbortedError,
                ConnectionRefusedError):
            self.refresh_connection()
            embedded_answer = self.solver.sample_bqm(embedded_bqm,
                                                     num_reads=self.sample_count,
                                                     answer_mode='raw'
                                                     ).sampleset
            self.qpu_time_used += embedded_answer.info['timing']['qpu_access_time']
        answer = unembed_sampleset(target_sampleset=embedded_answer,
                                   embedding=this_embedding,
                                   source_bqm=source_bqm_unembedded)

        return answer

    def get_qa_samples(self, qubo_as_bqm, label = None):
            # uqo: problem.embedding = ...
        if label is None:
            this_embedding = self.find_embedding_with_client(
                qubo_as_bqm, False, label) if self.embedding_unclamped is None else self.embedding_unclamped
        else:
            this_embedding = self.find_embedding_with_client(
                qubo_as_bqm, False, label) if self.embedding_clamped is None else self.embedding_clamped

        embedded_q = embed_bqm(source_bqm=qubo_as_bqm,
                                embedding=EmbeddedStructure(
                                target_edges=self.solver.edges,
                                embedding=this_embedding
                                    )
                                )

        answer = self.run_qa_sampling(embedded_q, this_embedding, qubo_as_bqm, self.sample_count)
        samples = list(answer.samples())

        return samples

    def split_and_rename_dicts(self, input_list, chunk_size):
        result = []
        for original_dict in input_list:
            keys = list(original_dict.keys())
            for i in range(0, len(keys), chunk_size):
                # Create a chunk of the dictionary
                chunk = {j: original_dict[keys[i + j]] for j in range(min(chunk_size, len(keys) - i))}
                result.append(chunk)
        return result

    def canonicalize_bqm_keys(self, bqm):
        linear = {var: bqm.linear[var] for var in bqm.variables}
        quadratic = {
            (min(u, v), max(u, v)): bqm.quadratic[(u, v)]
            for u, v in bqm.quadratic
        }
        return di.BQM(linear, quadratic, bqm.vartype)


    def get_parallel_qa_samples(self, qubo_matrix, label=None):
        embeddings = self.parallel_embeddings_unclamped if label is None else self.parallel_embeddings_clamped

        bqm = di.BQM(qubo_matrix, "BINARY")

        list_embedded_bqm = []
        #chain_strength_fixed = dwave_embedding.chain_strength.uniform_torque_compensation(bqm, 0.2)
        for subgraph, embedding in zip(self.subgraphs, embeddings):
            embedded_bqm = dwave_embedding.embed_bqm(source_bqm=bqm,
                                                      embedding=EmbeddedStructure(
                                                            target_edges=self.solver.edges,
                                                            embedding=embedding),
                                                      target_adjacency=subgraph)
                                                      #chain_strength=chain_strength_fixed)
            list_embedded_bqm.append(embedded_bqm)
            #combined_bqm_dict = {**combined_bqm_dict, **embedded_bqm}

        combined_bqm = list_embedded_bqm[0]
        for i in range(1, len(list_embedded_bqm)):
            combined_bqm.update(list_embedded_bqm[i])

        combined_qubo = self.duplicate_and_concatenate_qubo(qubo_matrix, len(embeddings))
        combined_source_qubo_bqm = di.BQM(combined_qubo, "BINARY")
        combined_source_qubo, _ = combined_source_qubo_bqm.to_qubo()

        combined_embedding = {}
        new_key = 0
        for d in embeddings:
            for i in range(len(d.items())):
            #for key, value in d.items():
                #combined_embedding[new_key] = value
                combined_embedding[new_key] = d[i]
                new_key += 1


        combined_embedded_bqm =combined_bqm

        answer = self.run_qa_sampling(combined_embedded_bqm, combined_embedding, combined_source_qubo_bqm,
                                      int(self.sample_count/10)) # sample_count ist immer in 10ner Schritten
        samples = list(answer.samples())
        splitted_samples = self.split_and_rename_dicts(samples, qubo_matrix.shape[0])


        return splitted_samples

    def get_samples_batch(self, input_batch, label_batch=None):
        samples_all = []
        for i in range(len(input_batch)):
            x = input_batch[i]
            y = label_batch[i] if label_batch is not None else None
            samples = self.get_samples(x, label=y)
            samples_all.append(samples)
        return samples_all

    def get_samples(self, input_vector, initial_qubo, label=None):
        qubo_matrix = self.create_qubo_matrix_from(input_vector, initial_qubo, label)
        if self.solver_string == "SA":
            if self.parallelize:
                tasks = [
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_0, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_1, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_2, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_3, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_4, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_5, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_6, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_7, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_8, self.seed),
                    (qubo_matrix, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_9, self.seed),

                ]
                futures = [self.executor.submit(self.parallel_sa_sample, task) for task in tasks]
                samples = []
                for future in futures:
                    samples.extend(future.result())

            else:
                qubo_as_bqm = di.BQM(qubo_matrix, "BINARY")
                samples = self.sample_sa(self.sampler, qubo_as_bqm, self.sample_count, self.anneal_steps)

        elif self.solver_string == "BMS":
            qubo = qubo_matrix.astype("float32")
            qubo_assertion = qubo.astype(np.float32)
            assert qubo_assertion.shape[0] == qubo_assertion.shape[1], "QUBO embedding_matrix must be square"

            # we are in unclamped phase
            if label is None:
                if self.unclamped_sampler is None:
                    if self.temperatures:
                        # "reach equilibrium" (i.e. anneal with temperature schedule) only once, like in paper
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                     seed=self.seed, num_total_anneals_per_sampling=1,
                                                                     temperatures=self.temperatures,
                                                                     parallel_annealing=self.parallelize)
                    else:
                        # "reach equilibrium" (i.e. anneal with temperature schedule) only once, like in paper
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                     seed=self.seed, num_total_anneals_per_sampling=1,
                                                                     parallel_annealing=self.parallelize)
                    self.unclamped_sampler, samples_array = sampler.draw_samples()
                else:
                    self.unclamped_sampler, samples_array = self.unclamped_sampler.draw_samples(qubo)
                # we are in clamped phase
            else:
                if self.clamped_sampler is None:
                    if self.temperatures:
                        # "reach equilibrium" (i.e. anneal with temperature schedule) twice, like in paper
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                     seed=self.seed, num_total_anneals_per_sampling=2,
                                                                     temperatures=self.temperatures,
                                                                     parallel_annealing=self.parallelize)
                    else:
                        # "reach equilibrium" (i.e. anneal with temperature schedule) twice, like in paper
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                     seed=self.seed, num_total_anneals_per_sampling=2,
                                                                     parallel_annealing=self.parallelize)
                    self.clamped_sampler, samples_array = sampler.draw_samples()
                else:
                    self.clamped_sampler, samples_array = self.clamped_sampler.draw_samples(qubo)
            samples = [dict(enumerate(sample)) for sample in samples_array.tolist()]
        else:
            qubo_as_bqm = di.BQM(qubo_matrix, "BINARY")
            # config = Config(configpath='src/secrets/config.json')
            if (self.solver_string == "Advantage_system4.1" or self.solver_string == "DW_2000Q_6"
                    or self.solver_string == "Advantage_system7.1"):
                if not self.parallelize:
                    if self.restricted:
                        if label is not None:
                            answer = solve_linear_qubo(qubo_as_bqm, self.sample_count)
                            samples = list(answer.samples())
                        else:
                            samples = self.get_qa_samples(qubo_as_bqm, label)
                    else:
                        samples = self.get_qa_samples(qubo_as_bqm, label)
                else:
                    if self.restricted and label is not None:
                        answer = solve_linear_qubo(qubo_as_bqm, self.sample_count)
                        samples = list(answer.samples())
                    else:
                        try:
                            samples = self.get_parallel_qa_samples(qubo_matrix, label)
                        except (
                                ConnectionError, ConnectionResetError, ConnectionAbortedError,
                                ConnectionRefusedError):
                            self.refresh_connection()
                            samples = self.get_parallel_qa_samples(qubo_matrix, label)
            else:
                raise Exception(
                    'No valid solver specified. Valid solvers are "SA", "BMS", "MyQLM", "QBSolv", "DW_2000Q_6", "Advantage_system4.1", "FujitsuDAU"')
        return samples

        # done



    def get_average_configuration_single(self, samples: list, x_input: np.ndarray, y: np.ndarray = None):
        unclamped = y is None
        label = None if unclamped else np.array(y).flatten()

        n_hidden = self.num_active_units
        num_active_conv_units = len(self.pooled_units)
        sizes_active = [len(self.pooled_units)] + list(self.sequential_layer_sizes)
        starts = np.cumsum([0] + sizes_active[:-1])  # local starts per active layer

        avgs_biases_conv_units = np.zeros_like(self.biases_conv_units)
        avgs_biases_sequential = np.zeros_like(self.biases_sequential_units)
        avgs_biases_output = np.zeros_like(self.biases_output)
        avgs_kernel_weights = np.zeros_like(self.kernel_weights)
        avgs_weights_hidden_interlayer = np.zeros_like(self.weights_hidden_interlayer)

        avgs_weights_interlayer_sequential = [np.zeros_like(w) for w in self.weights_interlayer_sequential] if not self.restricted else 0
        avgs_weights_sequential_layers = [np.zeros_like(w) for w in self.weights_sequential_layer]
        avgs_weights_hidden_to_output = np.zeros_like(self.weights_hidden_to_output)
        avgs_weights_output_output = np.zeros_like(self.weights_output_output)

        sample_matrix = np.vstack([np.array(list(s.values())) for s in samples])

        if self.pooling_type == "probabilistic":
            #remove the conv units from the sampel matrix
            sample_matrix = sample_matrix[:, self.num_conv_units:]
        n_reads = sample_matrix.shape[0]

        avg_biases = sample_matrix.mean(axis=0)

        if self.hidden_bias_type == "shared":
            # deprecated only worked with old pyramid structure
            # sum per layer into a shared bias slot
            # start = 0
            # for li in range(self.sequential_layer_sizes):
            #     cnt = self.num_hidden_units_per_layer[li]
            #     avgs_biases_conv_units[li] += np.sum(avg_biases[start:start + cnt])
            #     start += cnt
            avgs_biases_conv_units[0] += np.sum(avg_biases[:num_active_conv_units]) #/ num_active_conv_units
        elif self.hidden_bias_type == "none":
            pass  # keep zeros
        else:
            pooled_idx = np.asarray(self.pooled_units, dtype=int)
            pooled_marginals = avg_biases[:num_active_conv_units]
            avgs_biases_conv_units[pooled_idx] += pooled_marginals.astype(avgs_biases_conv_units.dtype)

        avgs_biases_sequential += avg_biases[num_active_conv_units:n_hidden]

        if unclamped:
            avgs_biases_output += avg_biases[n_hidden:]
        else:
            avgs_biases_output += label


        # Input to first hidden layer
        for i, h in enumerate(self.pooled_units):
            rows, cols = self.input_groups[h]
            patch = x_input[np.ix_(rows, cols)]
            Eh = float(sample_matrix[:, i].mean())
            avgs_kernel_weights += patch * Eh

        #avgs_kernel_weights /= len(self.pooled_units)

        if not self.restricted:
            #within layer connections in the sequential layers:
            for li, W in enumerate(self.weights_interlayer_sequential):
                # indices of the li-th sequential layer block inside the hidden slice
                cur_size = sizes_active[li + 1]  # sequential layer size
                cur_s = int(starts[li + 1])  # start index of that layer in sample_matrix
                cur_e = cur_s + cur_size

                # samples over this layer: shape (num_reads, cur_size)
                cur_block = sample_matrix[:, cur_s:cur_e]

                # E[h_i h_j] as an average outer product (upper triangle only)
                avg_outer = (cur_block.T @ cur_block) / n_reads
                triu = np.triu_indices(cur_size, k=1)
                avgs_weights_interlayer_sequential[li][triu] = avg_outer[triu]

        # Sequential Layer
        for li, _ in enumerate(self.weights_sequential_layer):
            prev_size = sizes_active[li]
            cur_size = sizes_active[li + 1]
            prev_s = int(starts[li])
            cur_s = int(starts[li + 1])
            prev_block = sample_matrix[:, prev_s:prev_s + prev_size]  # (N, prev)
            cur_block = sample_matrix[:, cur_s:cur_s + cur_size]  # (N, cur)
            avgs_weights_sequential_layers[li][:] = (prev_block.T @ cur_block) / n_reads

        # Hidden -> Output
        if self.pooling_type == "probabilistic":
            last_hidden = sum(self.num_active_units_per_layer[1:-1]) + np.arange(self.num_active_units_per_layer[-1])
        else:
            last_hidden = sum(self.num_active_units_per_layer[:-1]) + np.arange(self.num_active_units_per_layer[-1])
        if unclamped:
            for o in range(self.n_output_nodes):
                x_out = sample_matrix[:, n_hidden + o]  # E[y_o]
                y_last = sample_matrix[:, last_hidden]
                avgs_weights_hidden_to_output[:, o] += (y_last.T @ x_out) / n_reads
        else:
            for o in range(self.n_output_nodes):
                for i_h, h in enumerate(last_hidden):
                    y_h = sample_matrix[:, h]
                    avgs_weights_hidden_to_output[i_h, o] += np.average(y_h * label[o])

        # Output -> Output
        if unclamped:
            yvars = sample_matrix[:, n_hidden:]
            avg_outer = np.einsum('ni,nj->ij', yvars, yvars) / n_reads
            triu = np.triu_indices(self.n_output_nodes, k=1)
            avgs_weights_output_output[triu] += avg_outer[triu]
        else:
            outer = np.outer(label, label)
            triu = np.triu_indices(self.n_output_nodes, k=1)
            avgs_weights_output_output[triu] += outer[triu]

        return (
            avgs_biases_conv_units,
            avgs_biases_sequential,
            avgs_biases_output,
            avgs_kernel_weights,
            avgs_weights_hidden_interlayer,
            avgs_weights_interlayer_sequential,
            avgs_weights_sequential_layers,
            avgs_weights_hidden_to_output,
            avgs_weights_output_output
        )


    def calcualte_parallel_subgraphs(self, num_partitions=10):
        pegasus_graph = dnx.pegasus_graph(16)
        #edges = pegasus_graph.edges
        A = self.solver.edges
        connectivity_graph = nx.Graph(list(A))
        subgraph_paritions = emb.partition_graph(pegasus_graph, num_partitions)
        #subgraphs = emb.create_subgraphs(pegasus_graph, subgraph_paritions)
        subgraphs, buffer_subgraph = emb.create_subgraphs_with_buffer(connectivity_graph, subgraph_paritions)

        emb.plot_subgraphs(pegasus_graph, subgraphs,
                           (self.speicherort + self.param_string if self.speicherort is not None else None))
        # with open(f"src/embeddings/advantage_7_1/integer/subgraphs.pkl", "wb") as f:
        #      pickle.dump(subgraphs, f)
        return subgraphs

    def calcualte_parallel_embeddings(self, subgraphs, input_data, label=None):
        qubo_matrix = self.create_qubo_matrix_from(input_data, label)
        qubo_bqm = di.BQM(qubo_matrix, "BINARY")
        qubo_graph = nx.Graph(qubo_bqm.quadratic)
        print("qubo_graph", qubo_graph)
        print("qubo_bqm", qubo_bqm)

        if qubo_matrix.shape[0] == 1:
            qubo_graph = nx.Graph([(0, 0)])

        embeddings = []
        # TODO bug with embedding a single node if hidden nodes are 1
        for i in range(len(subgraphs)):
            print(f"Looking for embedding for subgraph {i}")
            embedding = minorminer.find_embedding(
                qubo_graph.edges,
                subgraphs[i].edges,
                random_seed=self.seed,
                tries=1000,
                chainlength_patience=100,
                threads=100,
                max_no_improvement=100,
            )
            embeddings.append(embedding)

            # while not embedding_found:
            #     print(f"Looking for embedding for subgraph {i} again")
            #     embedding, embedding_found = minorminer.find_embedding(
            #         qubo_graph.edges,
            #         subgraphs[i].edges,
            #         random_seed=self.seed,
            #         return_overlap=True,
            #         tries=1000,
            #         chainlength_patience=1000,
            #         threads=20,
            #         max_no_improvement=10,
            #     )

        combined_embeddings = {}

        for d in embeddings:
            for key, value in d.items():
                new_key = key
                while new_key in combined_embeddings:
                    new_key += 10

                if new_key in combined_embeddings:
                    combined_embeddings[new_key].extend(value)
                else:
                    combined_embeddings[new_key] = value

        pegasus_graph = dnx.pegasus_graph(16)
        plt.figure(figsize=(8, 6))
        dnx.draw_pegasus_embedding(pegasus_graph, emb=combined_embeddings, node_size=4, width=.3,
                                   unused_color="lightgray",
                                   show_labels=False)
        plt.title(f'All Embeddings')

        if self.speicherort:
            plt.savefig("src/" + self.speicherort + self.param_string + "combined_emb.png")

        plt.show()
        return embeddings


    def load_subgraphs(self, path):
        with open(path, "rb") as f:
            subgraphs = pickle.load(f)

        return subgraphs


    def load_embeddings(self, path):
        with open(path, "rb") as f:
            embeddings = pickle.load(f)

        return embeddings

    def find_and_save_embedding(self, example_data, example_label, subgraphs=None):
        if subgraphs is None:
            self.subgraphs = self.load_subgraphs("src/embeddings/advantage_7_1/integer/subgraphs.pkl")
        else:
            self.subgraphs = subgraphs

        # with open(f"src/embeddings/integer/subgraphs.pkl", "wb") as f:
        #      pickle.dump(self.subgraphs, f)

        embeddings_clamped = self.calcualte_parallel_embeddings(self.subgraphs, example_data, example_label)
        embeddings_unclamped = self.calcualte_parallel_embeddings(self.subgraphs, example_data)
        print("embeddings clamped", embeddings_clamped)
        print("embeddings unclamped", embeddings_unclamped)
        with open(f"src/embeddings/advantage_4_1/conv_deep_qbm/embeddings_clamped_{self.num_conv_units}.pkl", "wb") as f:
             pickle.dump(embeddings_clamped, f)
        with open(f"src/embeddings/advantage_4_1/conv_deep_qbm/embeddings_unclamped_{self.num_conv_units}.pkl", "wb") as f:
             pickle.dump(embeddings_unclamped, f)


    def nll_from_samples_single(self, samples_unclamped: list, y, nll) -> float:
        """
        Compute NLL for a single input from its unclamped samples.
          - samples_unclamped: list[dict] (one dict per sample: var -> {0,1})
          - y: scalar 0/1 for binary, or one-hot / class index for multi-class
          - nll: torch.nn.NLLLoss() instance
        Returns:
          - scalar loss (float)
        """
        # Stack samples into matrix: rows=samples, cols=variables (hidden + output)
        np_samples = np.vstack([np.array(list(s.values())) for s in samples_unclamped])
        if self.pooling_type == "probabilistic":
            num_active_units = self.num_total_units + len(self.pooled_units)
        else:
            num_active_units = self.num_active_units
        # Output marginals = mean over samples of output vars (after the hidden block)
        out_probs = np_samples[:, num_active_units:].mean(axis=0)  # shape: (n_output_nodes,)#TODO: only works if last layer is pooling layer

        # Build a proper probability vector for NLLLoss
        if self.n_output_nodes == 1:
            # Binary case: turn p into [1-p, p]
            p = float(out_probs[0])
            prob_vec = np.array([1.0 - p, p], dtype=np.float32)
            target_idx = int(y)  # y should be 0 or 1
        else:
            # Multi-class: use mean marginals; (optionally) renormalize to sum to 1
            prob_vec = out_probs.astype(np.float32)
            s = prob_vec.sum()
            if s > 0:
                prob_vec = prob_vec / s
            # y can be one-hot or class index
            try:
                target_idx = int(np.argmax(y))
            except Exception:
                target_idx = int(y)

        # NLLLoss expects log-probs and a class index
        prob_t = torch.tensor(prob_vec, dtype=torch.float32)
        log_probs = torch.log(prob_t + 1e-12)  # safeguard against log(0)
        label = torch.tensor([target_idx], dtype=torch.long)
        loss = nll(log_probs.unsqueeze(0), label)  # shape (1, C) vs (1,)
        return float(loss.item())


    def add_at_most_one_penalty_upper(self, qubo, penalty):
        for g in self.pool_windows:
            ids = np.asarray(g, dtype=int)
            m = ids.size
            if m <= 1:
                continue
            ii, jj = np.triu_indices(m, k=1)
            qubo[ids[ii], ids[jj]] += penalty  # only upper triangle
        return qubo

    def add_link_penalty_upper(self, qubo: np.ndarray, penalty_B: float):
        p_start = self.num_conv_units  # first pooling var index
        for g_idx, g in enumerate(self.pool_windows):
            p = p_start + g_idx  # pooling var for this group
            ids = np.asarray(g, dtype=int)  # conv-unit indices in this pool
            if ids.size == 0:
                # still add B * p_g to keep the penalty well-defined
                qubo[p, p] += penalty_B
                continue

            # Linear terms: B * p_g  and  B * sum_i x_{g,i}
            qubo[p, p] += penalty_B
            qubo[ids, ids] += penalty_B

            # Quadratic terms: -2B * sum_i p_g x_{g,i}  (upper triangle only)
            lo = np.minimum(ids, p)
            hi = np.maximum(ids, p)
            mask = lo != hi
            if np.any(mask):
                qubo[lo[mask], hi[mask]] += -2.0 * penalty_B

        return qubo

    def train_for_one_iteration(self, x_batch, y_batch, learning_rate, nll):
        errors_biases_conv = 0
        errors_biases_sequential = 0
        errors_biases_output = 0
        errors_weights_kernels = 0
        errors_weights_hidden_interlayer = 0
        if not self.restricted:
            errors_weights_interlayer_sequential = [0 for _ in self.weights_interlayer_sequential]
        errors_weights_sequential = [0 for _ in self.sequential_layer_sizes]
        errors_weights_hidden_to_output = 0
        errors_weights_output_output = 0
        total_nll_loss = 0

        if self.pooling_type == "probabilistic":
            clamped_qubo_len = self.num_total_units + len(self.pooled_units)
            unclamped_qubo_len = self.num_total_units + len(self.pooled_units) + self.n_output_nodes
        elif self.pooling_type == "deterministic":
            clamped_qubo_len = self.num_active_units
            unclamped_qubo_len = self.num_active_units + self.n_output_nodes
        else:
            raise NotImplementedError("Pooling type None not implemented for training yet")
        initial_clamped_qubo = np.zeros((clamped_qubo_len, clamped_qubo_len))
        initial_unclamped_qubo = np.zeros((unclamped_qubo_len, unclamped_qubo_len))

        if self.pooling_type == "probabilistic":
            initial_clamped_qubo = self.add_at_most_one_penalty_upper(initial_clamped_qubo, self.prob_penalty) #TODO: is this a good value?
            initial_unclamped_qubo = self.add_at_most_one_penalty_upper(initial_unclamped_qubo, self.prob_penalty)
            initial_clamped_qubo = self.add_link_penalty_upper(initial_clamped_qubo, self.prob_penalty)
            initial_unclamped_qubo = self.add_link_penalty_upper(initial_unclamped_qubo, self.prob_penalty)


        for x, y in zip(x_batch, y_batch):
            samples_clamped = self.get_samples(x, initial_clamped_qubo, label=y)
            avgs_clamped_biases_conv, avgs_clamped_biases_sequential, avgs_clamped_biases_output, avgs_clamped_weights_kernels, avgs_clamped_weights_hidden_interlayer, avgs_clamped_weights_interlayer_sequential, avgs_clamped_weights_sequential, avgs_clamped_weights_hidden_to_output, avgs_clamped_weights_output_output = self.get_average_configuration_single(
                samples_clamped, x, y)

            samples_unclamped = self.get_samples(x, initial_unclamped_qubo)
            avgs_unclamped_biases_conv, avgs_unclamped_biases_sequential, avgs_unclamped_biases_output, avgs_unclamped_weights_kernels, avgs_unclamped_weights_hidden_interlayer, avgs_unclamped_weights_interlayer_sequential, avgs_unclamped_weights_sequential, avgs_unclamped_weights_hidden_to_output, avgs_unclamped_weights_output_output= self.get_average_configuration_single(samples_unclamped, x)


            total_nll_loss += self.nll_from_samples_single(samples_unclamped, y, nll)

            errors_biases_output += (avgs_clamped_biases_output -
                                         avgs_unclamped_biases_output)

            errors_biases_conv += (avgs_clamped_biases_conv -
                                         avgs_unclamped_biases_conv)

            errors_biases_sequential += (avgs_clamped_biases_sequential -
                                            avgs_unclamped_biases_sequential)

            errors_weights_kernels += (avgs_clamped_weights_kernels - avgs_unclamped_weights_kernels)


            if not self.restricted:
                #errors_weights_hidden_interlayer += (
                 #           avgs_clamped_weights_hidden_interlayer - avgs_unclamped_weights_hidden_interlayer)
                errors_weights_interlayer_sequential = \
                    [errors_weights_interlayer_sequential[i] +
                     (avgs_clamped_weights_interlayer_sequential[i] - avgs_unclamped_weights_interlayer_sequential[i])
                     for i in range(len(self.weights_interlayer_sequential))]

            errors_weights_sequential = \
                [errors_weights_sequential[i] +
                                         (avgs_clamped_weights_sequential[i] - avgs_unclamped_weights_sequential[i])
                 for i in range(len(self.sequential_layer_sizes))]

            errors_weights_hidden_to_output += (
                    avgs_clamped_weights_hidden_to_output - avgs_unclamped_weights_hidden_to_output)

            errors_weights_output_output += (
                    avgs_clamped_weights_output_output- avgs_unclamped_weights_output_output)

        errors_biases_conv /= x_batch.shape[0]
        errors_biases_sequential /= x_batch.shape[0]
        errors_biases_output /= x_batch.shape[0]
        errors_weights_kernels /= x_batch.shape[0]

        if not self.restricted:
            #errors_weights_hidden_interlayer /= x_batch.shape[0]
            errors_weights_interlayer_sequential = [error / x_batch.shape[0] for error in errors_weights_interlayer_sequential]

        errors_weights_sequential = [error / x_batch.shape[0] for error in errors_weights_sequential]
        errors_weights_hidden_to_output /= x_batch.shape[0]
        errors_weights_output_output /= x_batch.shape[0]

        self.biases_output -= learning_rate * errors_biases_output
        self.biases_conv_units -= learning_rate * errors_biases_conv
        self.biases_sequential_units -= learning_rate * errors_biases_sequential

        self.kernel_weights -= learning_rate * errors_weights_kernels

        if not self.restricted:
            #self.weights_hidden_interlayer -= learning_rate * errors_weights_hidden_interlayer
            self.weights_interlayer_sequential = [weights - learning_rate * errors_weights
                                                    for weights, errors_weights in zip(self.weights_interlayer_sequential, errors_weights_interlayer_sequential)]

        self.weights_sequential_layer = [weights - learning_rate * errors_weights
                                         for weights, errors_weights in zip(self.weights_sequential_layer, errors_weights_sequential)]
        self.weights_hidden_to_output -= learning_rate * errors_weights_hidden_to_output
        self.weights_output_output -= learning_rate * errors_weights_output_output

        avg_batch_loss = total_nll_loss / len(x_batch)
        self.training_history.nll_per_batch.append(avg_batch_loss)
        #avg_batch_loss = 0
        return errors_biases_output, avg_batch_loss

    def split_into_batches(self, lst, batch_size):
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

    def save_weights(self, title, path="out"):
        with open(f"{path}/{title}.pkl", "wb") as f:
            pickle.dump(self.weight_objects, f)

    def train_model(self, train_X, train_Y, val_X=None, val_Y=None, batch_size=8, learning_rate=0.005, epochs=20):
        if self.speicherort != "":
            save_folder = "e" + str(epochs) + "_" + self.speicherort + self.param_string
            os.makedirs(save_folder, exist_ok=True)
        print("Training with \n"
                f"batch size: {batch_size}\n",
                f"learning rate: {learning_rate}\n",
                f"conv units: {self.num_conv_units}\n",
                f"sample count: {self.sample_count}\n",
                f"beta eff: {self.beta_eff}\n",
                f"layer dimensions: {self.conv_layer_dim}\n",
                f"num pooled units: {len(self.pool_windows)}\n",
                f"num active units: {self.num_active_units}\n",
              )


        for epoch in tqdm(range(1, epochs + 1), desc="Training", ncols=80):
            batchnum = 1
            num_batches = len(train_X) // batch_size
            nll = torch.nn.NLLLoss()
            self.epoch_nll = 0

            for b in tqdm(range(0, len(train_X), batch_size), desc=f"Training current epoch {epoch}", ncols=80, leave=False):
                if (b + batch_size) <= len(train_X):
                    x_batch = train_X[b:b + batch_size]
                    y_batch = train_Y[b:b + batch_size]
                else:
                    x_batch = train_X[b:]
                    y_batch = train_Y[b:]

                if len(x_batch) == 0:
                    print("Batch is empty")
                    continue

                try:
                    output_bias_errors_batch, avg_nll_batch = self.train_for_one_iteration(x_batch, y_batch, learning_rate, nll)
                    avg_output_bias_errors_batch = np.mean(output_bias_errors_batch)
                    self.epoch_errors += avg_output_bias_errors_batch
                    self.epoch_nll += avg_nll_batch
                    self.training_history.errors_per_batch.append(avg_output_bias_errors_batch)
                    if epoch == 1 and batchnum == 1 and (self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1"):
                        print(f"QPU time used for one iteration: {self.qpu_time_used} microseconds")
                    batchnum += 1
                except Exception as e:
                    self.save_weights(
                        f'e{epoch}_b{batchnum}_{self.param_string}', save_folder)
                    metrics.save_history(f"{save_folder}/", self.training_history)
                    raise e
            if self.speicherort != "":
                self.save_weights(
                    f'e{epoch}_{self.param_string}', save_folder)
            if val_X is not None:
                val_predictions = []
                for val_x in tqdm(val_X, desc="predict validation set", ncols=80, leave=False):
                    prediction, _ = self.predict(val_x)
                    val_predictions.append(prediction)

                acc, _, _, _, auc = metrics.get_metrics(val_Y, val_predictions, 2)
                combined_acc_auc = 0.5*acc + 0.5*auc
                self.training_history.acc_per_epoch.append(acc)
                self.training_history.auc_per_epoch.append(auc)
                self.training_history.combined_acc_auc_per_epoch.append(combined_acc_auc)

            avg_epoch_errors = self.epoch_errors / num_batches
            avg_epoch_nll = self.epoch_nll / num_batches

            self.training_history.error_per_epoch.append(avg_epoch_errors)
            self.training_history.nll_per_epoch.append(avg_epoch_nll)

            if self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1":
                print(f"QPU time used after {epoch} epochs: {self.qpu_time_used} microseconds")

        if self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1":
            print(f"QPU time used for one training run: {self.qpu_time_used} microseconds")

        if self.solver_string == "SA":
            print(f"SA time used for one training run: {self.sa_time_used} microseconds")
        if self.speicherort != "":
            with open(f"{save_folder}/acc_per_epoch{self.seed}.pkl", "wb") as f:
                pickle.dump(self.training_history.acc_per_epoch, f)
            with open(f"{save_folder}/auc_per_epoch{self.seed}.pkl", "wb") as f:
                pickle.dump(self.training_history.auc_per_epoch, f)
            with open(f"{save_folder}/combined_acc_auc_per_epoch{self.seed}.pkl", "wb") as f:
                pickle.dump(self.training_history.combined_acc_auc_per_epoch, f)
        print(self.kernel_weights)


    def find_embedding_with_client(self, bqm, save, label = None):
        if bqm.quadratic == {}:
            qubo_graph = nx.Graph([(0, 0)])
            target_edges = qubo_graph.edges
        else:
            target_edges = bqm.quadratic
        embedding, embedding_found = minorminer.find_embedding(target_edges,
                                                               self.solver.edges,
                                                               return_overlap=True,
                                                               random_seed=self.seed,
                                                               threads=4
                                                               )
        while not embedding_found:
            print("No embedding found. Trying again...")
            embedding, embedding_found = minorminer.find_embedding(
                target_edges, self.solver.edges, return_overlap=True
            )
        # adapted from: https://support.dwavesys.com/hc/en-us/community
        # /posts/1500001417242-Solved-How-to-find-out-the-number-of-qubits
        # -used-to-solve-a-problem
        print(f"Number of logical variables: {len(embedding.keys())}")
        print(f"Number of physical qubits used in embedding: "
              f"{sum(len(chain) for chain in embedding.values())}"
                )

        # save embedding as pdf if parameter save is set to True
        if save:
            if self.solver_string == "DW_2000Q_6":
                dwave_networkx.draw_chimera_embedding(dwave_networkx.chimera_graph(16), emb=embedding, node_size=3,
                                                      width=.3)
            elif self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1":
                dwave_networkx.draw_pegasus_embedding(dwave_networkx.pegasus_graph(16), emb=embedding, node_size=3,
                                                      width=.3)
            elif self.solver_string == "Advantage2_prototype2.6":
                dwave_networkx.draw_zephyr_embedding(dwave_networkx.zephyr_graph(16), emb=embedding, node_size=3,
                                                     width=.3)
            path = PurePath()
            path = Path(path / 'embeddings')
            path.mkdir(mode=0o770, exist_ok=True)
            plt.savefig("embeddings/embeddingccc.pdf")

        if label is None:
            self.embedding_unclamped = embedding
        else:
            self.embedding_clamped = embedding
        return embedding

    def refresh_connection(self):
        """
        If there are problems with the connection to the D-Wave, this method
        can be used to close the client object and create a new one.
        :return: No return value, adapts the attributes of the DQBM object
        directly.
        """
        solver_id = self.solver.id
        self.client.close()
        # get new connection to client
        self.client = Client(token=self.TOKEN, solver=solver_id)
        # make sure to get the same solver from this connection
        self.solver = self.client.get_solver(name=solver_id)


    def predict(self, data):
        if self.pooling_type == "probabilistic":
            unclamped_qubo_len = self.num_total_units + len(self.pooled_units) + self.n_output_nodes
        elif self.pooling_type == "deterministic":
            unclamped_qubo_len = self.num_active_units + self.n_output_nodes
        else:
            raise NotImplementedError("Pooling type None not implemented for training yet")
        initial_unclamped_qubo = np.zeros((unclamped_qubo_len, unclamped_qubo_len))

        if self.pooling_type == "probabilistic":
            initial_unclamped_qubo = self.add_at_most_one_penalty_upper(initial_unclamped_qubo, self.prob_penalty)
            initial_unclamped_qubo = self.add_link_penalty_upper(initial_unclamped_qubo, self.prob_penalty)
        samples = self.get_samples(data, initial_unclamped_qubo)

        np_samples = np.vstack(
            tuple([np.array(list(sample.values())) for sample in samples]))

        if self.pooling_type == "probabilistic":
            num_active_units = self.num_total_units + len(self.pooled_units)
        else:
            num_active_units = self.num_active_units
        samples_of_output = np_samples[:, num_active_units:] #TODO only works if there is nothing but conv or conv pool
        average_output = np.mean(samples_of_output, axis=0)
        rounded_output = np.round(average_output).astype(int)
        one_hot = np.argmax(average_output)

        if self.use_one_hot_encoding:
            return one_hot, samples_of_output.tolist()
        else:
            return rounded_output[0], samples_of_output.flatten()


    def get_result_distribution(self, samples_of_output_list, all_possible_patterns):

        if self.use_one_hot_encoding:
            all_samples = []
            for sample in samples_of_output_list:
                sample_str = ''.join(str(int(v)) for v in sample)
                all_samples.append(sample_str)
            samples_of_output_list = all_samples
        else:
            samples_of_output_list = [str(int(v)) for v in samples_of_output_list]

        sample_counts = Counter(samples_of_output_list)

        # Calculate total number of samples
        total_samples = sum(sample_counts.values())

        # Normalize to probabilities
        probabilities = {k: v / total_samples for k, v in sample_counts.items()} if total_samples > 0 else {}

        # Ensure all patterns from `all_possible_patterns` exist in `probabilities`
        sorted_probs = [probabilities.get(pattern, 0.0) for pattern in all_possible_patterns]

        return sorted_probs

    def get_annealing_time(self, train_X, train_Y, batch_size, learning_rate,):

            # all_possible_patterns = ["0", "1"]
            # sorted_probs_list = []
            # random data point just to get embedding
            if (
                self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1") and self.parallelize:
                print("Calculating embeddings...")
                # self.subgraphs = self.calcualte_parallel_subgraphs()
                self.subgraphs = self.load_subgraphs("src/embeddings/integer/subgraphs.pkl")
                print("Subgraphs loaded.")

                if not self.restricted:
                    # self.parallel_embeddings_clamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0], train_y[0])
                    self.parallel_embeddings_clamped = self.load_embeddings(
                        f"src/embeddings/integer/embeddings_clamped_{self.num_conv_units}.pkl")

                # self.parallel_embeddings_unclamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0])
                self.parallel_embeddings_unclamped = self.load_embeddings(
                    f"src/embeddings/integer/embeddings_unclamped_{self.num_conv_units}.pkl")
                print("Embeddings loaded.")
                # print("Embeddings calculated.")


            print("Training with \n"
                  f"batch size: {batch_size}\n",
                  f"learning rate: {learning_rate}\n",
                  f"hidden nodes: {self.num_conv_units}\n",
                  f"sample count: {self.sample_count}\n",
                  f"beta eff: {self.beta_eff}\n",
                  )

            nll = torch.nn.NLLLoss()
            batchnum = 1
            for b in tqdm(range(0, len(train_X), batch_size), desc=f"Training current epoch ", ncols=80,
                              leave=False):
                if (b + batch_size) <= len(train_X):
                    x_batch = train_X[b:b + batch_size]  # [X_train[i] for i in range(b, b + batch_size)]
                    y_batch = train_Y[b:b + batch_size]
                else:
                    x_batch = train_X[b:]  # [X_train[i] for i in range(b, len(X_train))]
                    y_batch = train_Y[b:]

                if len(x_batch) == 0:
                    print("Batch is empty")
                    continue


                output_bias_errors_batch, avg_nll_batch = self.train_for_one_iteration(x_batch, y_batch,
                                                                                               learning_rate, nll)

                batchnum += 1

                if batchnum == 3:
                    break

            if self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1":
                print(f"QPU time used for one training run: {self.qpu_time_used} microseconds")
                return self.qpu_time_used

            else:
                print(f"SA time used for one training run: {self.sa_time_used} microseconds")
                return self.sa_time_used




    def get_best_combined_acc_auc(self):
        best_epoch = np.argmax(self.training_history.combined_acc_auc_per_epoch)
        best_combined_acc_auc = self.training_history.combined_acc_auc_per_epoch[best_epoch]
        best_acc = self.training_history.acc_per_epoch[best_epoch]
        best_auc = self.training_history.auc_per_epoch[best_epoch]
        return best_epoch, best_acc, best_auc, best_combined_acc_auc

    import matplotlib.pyplot as plt
    import networkx as nx

    def visualize_architecture(self, save_path=None):
        G = nx.DiGraph()

        layer_offset = 0
        pos = {}

        # Input layer
        input_nodes = [f"I{i}" for i in range(self.dim_input)]
        for i, node in enumerate(input_nodes):
            pos[node] = (0, -i)
            G.add_node(node, layer='input')

        # First hidden layer
        h_nodes = []
        layer = 1
        for i in range(self.conv_layer_dim[0]):
            node = f"H{layer}_{i}"
            h_nodes.append(node)
            pos[node] = (layer, -i)
            G.add_node(node, layer='hidden1')

            # connect from input_groups
            for input_idx in self.input_groups[i]:
                G.add_edge(f"I{input_idx}", node)

        # Deeper hidden layers
        prev_nodes = h_nodes
        for layer in range(2, len(self.conv_layer_dim)):
            h_nodes = []
            for i in range(self.conv_layer_dim[layer - 1]):
                node = f"H{layer}_{i}"
                h_nodes.append(node)
                pos[node] = (layer, -i)
                G.add_node(node, layer=f'hidden{layer}')

            # connect all-to-all (as your inter-conv layers do)
            for p in prev_nodes:
                for h in h_nodes:
                    G.add_edge(p, h)

            prev_nodes = h_nodes

        # Output layer
        out_layer = len(self.conv_layer_dim)
        output_nodes = [f"O{i}" for i in range(self.n_output_nodes)]
        for i, node in enumerate(output_nodes):
            pos[node] = (out_layer, -i)
            G.add_node(node, layer='output')

        # connect last hidden layer to output
        for h in prev_nodes:
            for o in output_nodes:
                G.add_edge(h, o)

        # Draw
        plt.figure(figsize=(12, 6))
        nx.draw(G, pos, with_labels=True, node_size=600, node_color='lightblue', font_size=8, arrowsize=10)
        plt.title("Conv-Deep-QBM Architecture")

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def get_last_hidden_embedding(self, X):
        feats = []
        for i in tqdm(range(0, len(X), 10), desc=f"Getting hidden embeddings", ncols=80,
                              leave=False):
            xb = X[i:i + 10]
            # get_samples_batch returns list[list[dict]]; we average hidden part
            samples_batch = self.get_samples_batch(xb)  # unclamped
            for samples in samples_batch:
                mat = np.vstack([np.array(list(s.values())) for s in samples])
                h = mat[:, :self.num_conv_units].mean(axis=0)  # hidden mean
                feats.append(h)
        return np.vstack(feats)  # (N, D)

