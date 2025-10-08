import dimod
import torch.nn
from collections import Counter
from neal import SimulatedAnnealingSampler
import pickle
import os


import numpy as np
import dimod as di

#import boltzmann_sampler
import random
import dwave_networkx as dnx
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


class Disc_QBM():
    def __init__(self, dim_input, num_classes, epochs=2, n_hidden_nodes=4, seed=77, solver="SA", restricted=False,
                 sample_count=20, anneal_steps=20, beta_eff=1,
                 param_string="", load_path="", speicherort=None, parallelize=False,
                 use_one_hot_encoding=False, use_old_parallization=True):

        self.epochs = epochs
        self.seed = seed
        self.training_history = metrics.History([], [], [], [], [], [], [])
        self.temperatures = None
        self.dim_input = dim_input
        self.n_hidden_nodes = n_hidden_nodes

        self.restricted = restricted
        self.parallelize = parallelize
        with open("src/secrets/TOKEN.txt", "r") as f:
            TOKEN = f.read().strip()
        print(TOKEN)
        self.TOKEN = TOKEN

        # TODO: gucken ob integer encoding nicht genauso gut wäre. Würde ziemlich viele Parameter sparen.
        self.use_one_hot_encoding = use_one_hot_encoding
        if use_one_hot_encoding:
            self.n_output_nodes = num_classes
        else:
            self.n_output_nodes = 1
        if not restricted:
            self.weights_hidden_hidden = self.init_weights_hidden_hidden()
        else:
            self.weights_hidden_hidden = None
        (self.weights_all_visible_to_hidden, self.weights_clamped_visible_to_output, self.weights_output_output,
         self.biases_hidden, self.biases_output) = self.init_weights_for_supervised()
        self.param_string = param_string

        self.weight_objects = [
            self.weights_all_visible_to_hidden, self.weights_clamped_visible_to_output,
            self.biases_hidden, self.biases_output,
            self.weights_output_output, self.weights_hidden_hidden if self.weights_hidden_hidden is not None else None
        ]

        if solver == "SA":
            self.sa_time_used = 0
            if self.parallelize:
                self.executor = ProcessPoolExecutor(max_workers=10)
                self.use_old_parallization = use_old_parallization
                if use_old_parallization:
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
            else:
                self.sampler = SimulatedAnnealingSampler()
        elif solver == "BMS":
            self.clamped_sampler = None
            self.unclamped_sampler = None

        else:  # use D-Wave Advantage
            # different embeddings for each phase, because then, we can
            # reuse the initial embedding
            self.embedding_clamped = None
            self.embedding_unclamped = None

            if self.parallelize:
                if not self.restricted:
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

            if not self.restricted:
                # self.parallel_embeddings_clamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0], train_y[0])
                self.parallel_embeddings_clamped = self.load_embeddings(
                    f"src/embeddings/integer/embeddings_clamped_{self.n_hidden_nodes}.pkl")

            # self.parallel_embeddings_unclamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0])
            self.parallel_embeddings_unclamped = self.load_embeddings(
                f"src/embeddings/integer/embeddings_unclamped_{self.n_hidden_nodes}.pkl")
            print("Embeddings loaded.")

        if (self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1") and self.parallelize\
                and (self.parallel_embeddings_clamped is None or self.parallel_embeddings_unclamped is None):
            print("Calculating embeddings...")
            # self.subgraphs = self.calcualte_parallel_subgraphs()
            self.subgraphs = self.load_subgraphs("src/embeddings/integer/subgraphs.pkl")
            print("Subgraphs loaded.")

            if not self.restricted:
                # self.parallel_embeddings_clamped = self.calcualte_parallel_embeddings(self.subgraphs, X[0], Y[0])
                self.parallel_embeddings_clamped = self.load_embeddings(
                    f"src/embeddings/integer/embeddings_clamped_{self.n_hidden_nodes}.pkl")

            # self.parallel_embeddings_unclamped = self.calcualte_parallel_embeddings(self.subgraphs, X[0])
            self.parallel_embeddings_unclamped = self.load_embeddings(
                f"src/embeddings/integer/embeddings_unclamped_{self.n_hidden_nodes}.pkl")
            print("Embeddings loaded.")



    def load_savepoint(self, savepoint):
        savepoint = Path(savepoint)
        if savepoint.exists():
            with open(savepoint, "rb") as file:
                loaded_savepoint = pickle.load(file)
        else:
            raise FileNotFoundError("Savepoint file not found")

        assert len(loaded_savepoint) in [5, 6]
        self.weight_objects = [
            self.weights_all_visible_to_hidden, self.weights_clamped_visible_to_output,
            self.biases_hidden, self.biases_output,
            self.weights_output_output
        ]

        if len(loaded_savepoint) == 6:  # fully connected
            (self.weights_all_visible_to_hidden, self.weights_clamped_visible_to_output, self.biases_hidden,
             self.biases_output,
             self.weights_output_output, self.weights_hidden_hidden) = loaded_savepoint
        else: # semi restricted
            (self.weights_all_visible_to_hidden, self.weights_clamped_visible_to_output, self.biases_hidden, self.biases_output,
             self.weights_output_output) = loaded_savepoint

    def init_weights_for_supervised(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        # weights from visible to hidden layer
        weights_visible_to_hidden = np.random.uniform(
            -1, 1, (self.n_output_nodes + self.dim_input, self.n_hidden_nodes))

        # weights from visible to output layer
        weights_visible_to_output = np.random.uniform(
            -1, 1, (self.dim_input, self.n_output_nodes))

        # weights from output to output
        weights_output_output = np.triu(np.random.uniform(
            -1, 1, (self.n_output_nodes, self.n_output_nodes)), k=1)
        # bias hidden layer
        biases_hidden = np.random.uniform(-1, 1, self.n_hidden_nodes)

        # bias visible layer
        # initialization according to tipps from Hinton (2012): penalty Practical guide to training Restricted Boltzmann Machines
        # output_bias_value = - \
        #     math.log((1 / self.n_output_nodes) / (1 - (1 / self.n_output_nodes)))
        # biases_output = np.array(
        #     [output_bias_value for _ in range(self.n_output_nodes)])

        biases_output = np.random.uniform(-1, 1, self.n_output_nodes)

        return weights_visible_to_hidden, weights_visible_to_output, weights_output_output, biases_hidden, biases_output

    def init_weights_hidden_hidden(self):
        weights_hidden_hidden = np.triu(np.random.uniform(
            -1, 1, (self.n_hidden_nodes, self.n_hidden_nodes)), k=1)
        return weights_hidden_hidden

    def create_qubo_matrix_from(self, input_vector: np.ndarray, label: np.ndarray = None):
        # clamped phase
        # 3 visible, 4 hidden, clamped, upper right triangular embedding_matrix
        #
        #          (hb1 + v1h1w*v1 + v2h1w*v2.......) h1h2w + h2h2w .....
        #             (hb2  + v1h2w*v1 + v2h2w*v2.......)-----
        #                (hb3 + v1h3w*v1 + v2h3w*v2.......)-----
        #                   hb4
        if label is not None:
            if not self.use_one_hot_encoding:
                label = np.array([label])
            visible_values = np.array(np.concatenate((label, input_vector), axis=0))

            biased_visible_to_hidden_weights = np.diag(
                np.matmul(visible_values, self.weights_all_visible_to_hidden).flatten())
            qubo_matrix = (np.diag(self.biases_hidden) + biased_visible_to_hidden_weights)

            if not self.restricted:
                qubo_matrix = qubo_matrix + self.weights_hidden_hidden

            qubo_matrix = qubo_matrix / self.beta_eff


        # unclamped_phase
        #            vb1 v1v2w v1h1w v1h2w .....
        #                 vb2 v2h1w v2h2w .....
        #                     hb1 h1h2w h2h2w    -
        #                           hb2 h2h3w    -
        #                               hb3   -
        #                                    hb4
        else:
            weight_output_to_hidden = self.weights_all_visible_to_hidden[0:self.n_output_nodes, :]
            weight_output_to_hidden = np.pad(weight_output_to_hidden,
                                             ((0, self.n_hidden_nodes), (self.n_output_nodes, 0)), 'constant',
                                             constant_values=0)
            # weight_output_to_hidden.reshape((1, self.num_conv_units))

            qubo_matrix_size = self.n_output_nodes + self.n_hidden_nodes
            upper_part = np.zeros((qubo_matrix_size, qubo_matrix_size))
            upper_part += weight_output_to_hidden
            padded_output_output = np.pad(self.weights_output_output,
                                          ((0, self.n_hidden_nodes), (0, self.n_hidden_nodes)), 'constant',
                                          constant_values=0)
            upper_part += padded_output_output

            if not self.restricted:
                upper_part[self.n_output_nodes:, self.n_output_nodes:] += self.weights_hidden_hidden

            biases = np.concatenate((self.biases_output, self.biases_hidden), axis=0)
            diagonal_biases = np.diag(biases)

            weights_input_to_hidden = self.weights_all_visible_to_hidden[self.n_output_nodes:, :]
            weights_input_to_out_and_hidden = np.concatenate((self.weights_clamped_visible_to_output, weights_input_to_hidden),
                                                             axis=1)
            matmul = np.matmul(input_vector, weights_input_to_out_and_hidden)
            c = np.diag(matmul.flatten())
            diagonal_biases_with_input_weights = diagonal_biases + c

            qubo_matrix = (diagonal_biases_with_input_weights + upper_part) / self.beta_eff
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
        qubo_as_bqm, label, sample_count, anneal_steps, sa_sampler, seed = args
        #qubo_as_bqm = di.BQM(qubo_matrix, "BINARY")
        #sa_sampler = SimulatedAnnealingSampler()
        return list(sa_sampler.sample(qubo_as_bqm, num_reads=sample_count, num_sweeps=anneal_steps, seed=seed).samples())

    @staticmethod
    def parallel_bms_sample(args):
        qubo_matrix, bms_sampler = args
        # sa_sampler = SimulatedAnnealingSampler()
        sampler, samples_array = bms_sampler.draw_samples(qubo_matrix)
        return sampler, samples_array

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

        #raise Exception("Not implemented")


    def run_qa_sampling(self, embedded_bqm, this_embedding, source_bqm_unembedded, sample_count):
        try:
            embedded_answer = self.solver.sample_bqm(embedded_bqm,
                                                     num_reads=sample_count,
                                                     answer_mode='raw'
                                                     ).sampleset
            #print(f"    QPU time used: {embedded_answer.info['timing']['qpu_access_time']} microseconds")
            self.qpu_time_used += embedded_answer.info['timing']['qpu_access_time']
            #print("QPU time used: ", self.qpu_time_used)
            #raise Exception("Not implemented")
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
        # if self.current_batch_index == 50:
        #print(samples)
        #     raise Exception("stop")
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

    # def split_and_rename_dicts(self, input_list, chunk_size):
    #     result = []
    #     for original_dict in input_list:
    #         keys = list(original_dict.keys())
    #         for i in range(0, len(keys), chunk_size):
    #             # Create a chunk of the dictionary
    #             chunk = {j: original_dict[keys[i + j]] for j in range(min(chunk_size, len(keys) - i))}
    #
    #             # Switch the first two elements in the new dictionary
    #             chunk_keys = list(chunk.keys())
    #             if len(chunk_keys) >= 2:  # Ensure at least two elements exist
    #                 # Switch the first two keys
    #                 chunk[chunk_keys[0]], chunk[chunk_keys[1]] = chunk[chunk_keys[1]], chunk[chunk_keys[0]]
    #
    #             result.append(chunk)
    #     return result

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
        #combined_bqm_dict = {}
        #chain_strength_fixed = dwave_embedding.chain_strength.uniform_torque_compensation(bqm, 0.2)
        #print(chain_strength_fixed)
        for subgraph, embedding in zip(self.subgraphs, embeddings):
            embedded_bqm = dwave_embedding.embed_bqm(source_bqm=bqm,
                                                      embedding=EmbeddedStructure(
                                                            target_edges=self.solver.edges,
                                                            embedding=embedding),
                                                      target_adjacency=subgraph)
                                                      #chain_strength=chain_strength_fixed)
            #print("\nembedded_bqm", embedded_bqm)
            list_embedded_bqm.append(embedded_bqm)
            #combined_bqm_dict = {**combined_bqm_dict, **embedded_bqm}
        #print("ksskss", ksskss)

        combined_bqm = list_embedded_bqm[0]
        for i in range(1, len(list_embedded_bqm)):
            combined_bqm.update(list_embedded_bqm[i])

        #print("\n\nqubo_matrix\n", qubo_matrix)
        combined_qubo = self.duplicate_and_concatenate_qubo(qubo_matrix, len(embeddings))
        #print("\n\ncombined qubomatrix", combined_qubo)
        combined_source_qubo_bqm = di.BQM(combined_qubo, "BINARY")
        combined_source_qubo, _ = combined_source_qubo_bqm.to_qubo()
        #print("\ncombined_source_qubo", combined_source_qubo)
        #print("\ncombined_source_qubo", combined_source_qubo)
        combined_embedding = {}

        new_key = 0

        for d in embeddings:
            for i in range(len(d.items())):
            #for key, value in d.items():
                #combined_embedding[new_key] = value
                combined_embedding[new_key] = d[i]
                new_key += 1

        #print("\ncombined_bqm_dict", combined_bqm_dict)
        #combined_embedded_bqm = dimod.BQM(combined_bqm_dict, "BINARY")
        combined_embedded_bqm =combined_bqm
        #print("\ncombined_embedded_bqm", combined_embedded_bqm.to_qubo())
        #print(qubo_matrix.shape[0])
        #print("\n\nembeddings list", embeddings)
        #print("\n\n", combined_embedding)
        #raise Exception("stop")
        answer = self.run_qa_sampling(combined_embedded_bqm, combined_embedding, combined_source_qubo_bqm,
                                      int(self.sample_count/10)) # sample_count ist immer in 10ner Schritten
        samples = list(answer.samples())
        #print("\n\nsamples",samples)
        splitted_samples = self.split_and_rename_dicts(samples, qubo_matrix.shape[0])
        #print("\n\nslpittedsamples",splitted_samples)
        #raise Exception("stop")

        return splitted_samples

    def create_parallel_bms_samplers(self, qubo, num_tasks, label = None):
        if label is None:
                if self.temperatures:
                    # "reach equilibrium" (i.e. anneal with temperature schedule) only once, like in paper
                    sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count, seed=self.seed,
                                                                 num_total_anneals_per_sampling=1,
                                                                 temperatures=self.temperatures)
                else:
                    # "reach equilibrium" (i.e. anneal with temperature schedule) only once, like in paper
                    sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                 seed=self.seed, num_total_anneals_per_sampling=1)
        else:
                if self.temperatures:
                    # "reach equilibrium" (i.e. anneal with temperature schedule) twice, like in paper
                    sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count, seed=self.seed,
                                                                 num_total_anneals_per_sampling=2,
                                                                 temperatures=self.temperatures)
                else:
                    # "reach equilibrium" (i.e. anneal with temperature schedule) twice, like in paper
                    sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                 seed=self.seed, num_total_anneals_per_sampling=2)
        return sampler

    def get_parallel_bms_samples(self, qubo, label = None):

        num_tasks = int(self.sample_count / 8)
        rest_of_tasks = self.sample_count % 8
        if (len(self.parallel_clamped_samplers) == 0) and  (len(self.parallel_unclamped_samplers)== 0):
            for _ in range(8):
                self.parallel_clamped_samplers.append(self.create_parallel_bms_samplers(qubo, num_tasks, label))
                self.parallel_unclamped_samplers.append(self.create_parallel_bms_samplers(qubo, num_tasks))
            self.parallel_clamped_samplers.append(self.create_parallel_bms_samplers(qubo, rest_of_tasks, label))
            self.parallel_unclamped_samplers.append(self.create_parallel_bms_samplers(qubo, rest_of_tasks))

        if label is None:
            samplers = self.parallel_unclamped_samplers
        else:
            samplers = self.parallel_clamped_samplers
        tasks = [
            (qubo, samplers[0]),
            (qubo, samplers[1]),
            (qubo, samplers[2]),
            (qubo, samplers[3]),
            (qubo, samplers[4]),
            (qubo, samplers[5]),
            (qubo, samplers[6]),
            (qubo, samplers[7]),
            (qubo, samplers[8]),
        ]

        futures = [self.executor.submit(self.parallel_bms_sample, task) for task in tasks]
        samples = []
        new_samplers = []
        for future in futures:
            sample, sampler = future.result()
            samples.extend(sample)
            new_samplers.append(sampler)

        if label is None:
            self.parallel_unclamped_samplers = new_samplers
        else:
            self.parallel_clamped_samplers = new_samplers
        samples = [samples_array.tolist() for samples_array in samples]
        return [dict(enumerate(sample)) for sample in samples]




    def get_samples(self, input_vector, label=None):
        # TODO all solvers (except SA) respond with only one sample
        #     Except DAU who seems to map equal samples upon each other
        qubo_matrix = self.create_qubo_matrix_from(input_vector, label)
        # TODO: try whether this works better with qubo or ising?
        if self.solver_string == "SA":
            qubo_as_bqm = di.BQM(qubo_matrix, "BINARY")
            if self.parallelize and self.use_old_parallization:
                tasks = [
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_0, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_1, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_2, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_3, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_4, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_5, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_6, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_7, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_8, self.seed),
                    (qubo_as_bqm, label, int(self.sample_count / 10), self.anneal_steps, self.sampler_9, self.seed),

                ]
                # Use ProcessPoolExecutor to parallelize the sampling
                futures = [self.executor.submit(self.parallel_sa_sample, task) for task in tasks]
                samples = []
                for future in futures:
                    samples.extend(future.result())
            elif self.parallelize and not self.use_old_parallization:
                num_tasks = int(self.sample_count / 8)
                rest_of_tasks = self.sample_count % 8
                tasks = [
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, num_tasks, self.anneal_steps, self.sampler, self.seed),
                    (qubo_matrix, label, rest_of_tasks, self.anneal_steps, self.sampler, self.seed),

                ]

                futures = [self.executor.submit(self.parallel_sa_sample, task) for task in tasks]
                samples = []
                for future in futures:
                    samples.extend(future.result())

            else:
                qubo_as_bqm = di.BQM(qubo_matrix, "BINARY")
                #print(qubo_as_bqm)
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
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count, seed=self.seed, num_total_anneals_per_sampling=1, temperatures=self.temperatures, parallel_annealing=self.parallelize)
                    else:
                            # "reach equilibrium" (i.e. anneal with temperature schedule) only once, like in paper
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                         seed=self.seed, num_total_anneals_per_sampling=1,  parallel_annealing=self.parallelize)
                    self.unclamped_sampler, samples_array = sampler.draw_samples()
                else:
                    self.unclamped_sampler, samples_array = self.unclamped_sampler.draw_samples(qubo)
                # we are in clamped phase
            else:
                if self.clamped_sampler is None:
                    if self.temperatures:
                            # "reach equilibrium" (i.e. anneal with temperature schedule) twice, like in paper
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count, seed=self.seed, num_total_anneals_per_sampling=2, temperatures=self.temperatures,  parallel_annealing=self.parallelize)
                    else:
                            # "reach equilibrium" (i.e. anneal with temperature schedule) twice, like in paper
                        sampler = boltzmann_sampler.BoltzmannSampler(qubo, num_samples=self.sample_count,
                                                                         seed=self.seed, num_total_anneals_per_sampling=2,  parallel_annealing=self.parallelize)
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
        #print("\n\n", samples)
        return samples

        # done

    def get_average_configuration(self, samples: list, input_vector, label=None):
        ''' Takes samples from Annealer and averages for each neuron and connection
        '''

        # unclamped if label == None
        unclamped = label is None
        label = np.array(label) if label is not None else label
        label = label.flatten() if label is not None else label
        # biases (row = sample, column = neuron)
        np_samples = np.vstack(
            tuple([np.array(list(sample.values())) for sample in samples]))

        avgs_biases = np.average(np_samples, axis=0)
        avgs_biases_hidden = avgs_biases[self.n_output_nodes:] if unclamped else avgs_biases
        avgs_biases_output = avgs_biases[:self.n_output_nodes] if unclamped else label

        # weights
        avgs_weights_visible_to_hidden = np.zeros(
            self.weights_all_visible_to_hidden.shape)
        avgs_weights_visible_to_output = np.zeros(
            self.weights_clamped_visible_to_output.shape)
        avgs_weights_output_output = np.zeros(
            self.weights_output_output.shape)

        if unclamped:
            visible_vector = input_vector
            n_visible_nodes = self.dim_input
        else:
            visible_vector = np.concatenate((input_vector, label))
            n_visible_nodes = self.n_output_nodes + self.dim_input

        for h in range(self.n_hidden_nodes):
            # visible to hidden connections
            for v in range(n_visible_nodes):
                x, y = (visible_vector[v], np_samples[:, self.n_output_nodes + h]) if unclamped else (
                    visible_vector[v], np_samples[:, h])
                avgs_weights_visible_to_hidden[v, h] = np.average(x * y)

        for v in range(self.dim_input):
            # visible to output connections
            for out in range(self.n_output_nodes):
                x, y = (input_vector[v], np_samples[:, out]) if unclamped else (
                    input_vector[v], label[out])
                avgs_weights_visible_to_output[v, out] = np.average(x * y)

        for o in range(self.n_output_nodes):
            # output to output connections
            for o2 in range(o + 1, self.n_output_nodes):
                x, y = (np_samples[:, o], np_samples[:, o2]) if unclamped else (
                    label[o], label[o2])
                avgs_weights_output_output[o, o2] = np.average(x * y)

        if not self.restricted:
            avgs_weights_hidden_hidden = np.zeros(
                self.weights_hidden_hidden.shape)
            for h1 in range(self.n_hidden_nodes):
                for h2 in range(h1 + 1, self.n_hidden_nodes):
                    x, y = (np_samples[:, self.n_output_nodes + h1],
                            np_samples[:, self.n_output_nodes + h2]) if unclamped else (
                        np_samples[:, h1], np_samples[:, h2])
                    avgs_weights_hidden_hidden[h1, h2] = np.average(x * y)

        else:
            avgs_weights_hidden_hidden = None
        return avgs_biases_hidden, avgs_biases_output, avgs_weights_visible_to_hidden, avgs_weights_visible_to_output, avgs_weights_output_output, avgs_weights_hidden_hidden  # avgs_weights_output_to_hidden, avgs_weights_output_to_output



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
                    new_key += len(subgraphs)#10

                if new_key in combined_embeddings:
                    combined_embeddings[new_key].extend(value)
                else:
                    combined_embeddings[new_key] = value

        pegasus_graph = dnx.pegasus_graph(16)
        plt.figure(figsize=(8, 6))
        dnx.draw_pegasus_embedding(pegasus_graph, emb=combined_embeddings, node_size=4, width=.3,
                                   #unused_color="lightgray",
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
        with open(f"src/embeddings/advantage_4_1/integer/embeddings_clamped_{self.n_hidden_nodes}.pkl", "wb") as f:
            pickle.dump(embeddings_clamped, f)
        with open(f"src/embeddings/advantage_4_1/integer/embeddings_unclamped_{self.n_hidden_nodes}.pkl", "wb") as f:
            pickle.dump(embeddings_unclamped, f)

    def train_for_one_iteration(self, x_batch, y_batch, learning_rate, nll):
        errors_biases_hidden = 0
        errors_biases_output = 0
        errors_weights_visible_to_hidden = 0
        errors_weights_visible_to_output = 0
        errors_weights_output_output = 0
        errors_weights_hidden_hidden = 0
        total_nll_loss = 0

        for i in tqdm(range(len(x_batch)), desc="Training current batch", ncols=80, leave=False):
            samples_clamped = self.get_samples(x_batch[i], label=y_batch[i])
            # avgs_weights_visible_to_visible_clamped only has a value if not restricted   avgs_weights_output_to_hidden_clamped, avgs_weights_output_to_output_clamped
            avgs_bias_hidden_clamped, avgs_bias_output_clamped, avgs_weights_visible_to_hidden_clamped, avgs_weights_visible_to_output_clamped, avgs_weights_output_output_clamped, avgs_weights_hidden_hidden_clamped = self.get_average_configuration(
                samples_clamped, x_batch[i], [y_batch[i]])

            samples_unclamped = self.get_samples(x_batch[i])
            # avgs_weights_visible_to_visible_unclamped only has a value if not restricted avgs_weights_output_to_hidden_unclamped, avgs_weights_output_to_output_unclamped
            avgs_bias_hidden_unclamped, avgs_bias_output_unclamped, avgs_weights_visible_to_hidden_unclamped, avgs_weights_visible_to_output_unclamped, avgs_weights_output_output_unclamped, avgs_weights_hidden_hidden_unclamped = self.get_average_configuration(
                samples_unclamped, x_batch[i])

            # # Compute predictions from samples (first output node)
            # np_samples = np.vstack([np.array(list(sample.values())) for sample in samples_unclamped])
            # output_probs = np.mean(np_samples[:, :self.n_output_nodes], axis=0)  # Average over samples
            # output_probs = [1-output_probs[0], output_probs[0]]
            #
            # # Convert to log probabilities (for NLLLoss)
            # output_probs = torch.tensor(output_probs, dtype=torch.float32)
            # log_probs = torch.log(output_probs + 1e-12)  # Add epsilon to avoid log(0) #log_probs = torch.nn.functional.log_softmax(output_probs, dim=0)
            #
            #
            # # Compute NLL loss
            # label = torch.tensor([y[i]], dtype=torch.long)
            # loss = nll(log_probs.unsqueeze(0), label)  # Add batch dimension
            # total_nll_loss += loss.item()

            errors_biases_output += (avgs_bias_output_clamped -
                                     avgs_bias_output_unclamped)

            errors_biases_hidden += (avgs_bias_hidden_clamped -
                                     avgs_bias_hidden_unclamped)

            errors_weights_visible_to_hidden += (
                    avgs_weights_visible_to_hidden_clamped - avgs_weights_visible_to_hidden_unclamped)

            errors_weights_visible_to_output += (
                    avgs_weights_visible_to_output_clamped - avgs_weights_visible_to_output_unclamped)

            errors_weights_output_output += (
                    avgs_weights_output_output_clamped - avgs_weights_output_output_unclamped)

            if not self.restricted:
                errors_weights_hidden_hidden += (
                        avgs_weights_hidden_hidden_clamped - avgs_weights_hidden_hidden_unclamped)

        errors_biases_hidden /= x_batch.shape[0]
        errors_biases_output /= x_batch.shape[0]
        errors_weights_visible_to_hidden /= x_batch.shape[0]
        errors_weights_visible_to_output /= x_batch.shape[0]
        errors_weights_output_output /= x_batch.shape[0]

        if not self.restricted:
            errors_weights_hidden_hidden /= x_batch.shape[0]

        self.biases_output -= learning_rate * errors_biases_output
        self.biases_hidden -= learning_rate * errors_biases_hidden

        self.weights_all_visible_to_hidden -= learning_rate * errors_weights_visible_to_hidden
        self.weights_clamped_visible_to_output -= learning_rate * errors_weights_visible_to_output
        self.weights_output_output -= learning_rate * errors_weights_output_output

        if not self.restricted:
            self.weights_hidden_hidden -= learning_rate * errors_weights_hidden_hidden

        avg_batch_loss = total_nll_loss / len(x_batch)
        self.training_history.nll_per_batch.append(avg_batch_loss)
        #avg_batch_loss = 0
        return errors_biases_output, avg_batch_loss

    def split_into_batches(self, lst, batch_size):
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

    def save_weights(self, title, path="out"):
        # path = PurePath()
        # path = Path(path / 'saved_weights')
        # path.mkdir(mode=0o770, exist_ok=True)
        #
        # np.savez_compressed(path / f'{title}', w_vh=self.weights_all_visible_to_hidden,
        #                     b_v=self.biases_visible, b_h=self.biases_conv_units, lat=self.weights_clamped_visible_to_output)
        with open(f"{path}/{title}.pkl", "wb") as f:
            pickle.dump(self.weight_objects, f)

    def train_model(self, train_X, train_Y, val_X, val_Y, batch_size=8, learning_rate=0.005):
        #all_possible_patterns = ["0", "1"]
        #sorted_probs_list = []
        # random data point just to get embedding
        if (self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1") and self.parallelize:
            print("Calculating embeddings...")
            # self.subgraphs = self.calcualte_parallel_subgraphs()
            self.subgraphs = self.load_subgraphs("src/embeddings/integer/subgraphs.pkl")
            print("Subgraphs loaded.")

            if not self.restricted:
                # self.parallel_embeddings_clamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0], train_y[0])
                self.parallel_embeddings_clamped = self.load_embeddings(
                    f"src/embeddings/integer/embeddings_clamped_{self.n_hidden_nodes}.pkl")

            # self.parallel_embeddings_unclamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0])
            self.parallel_embeddings_unclamped = self.load_embeddings(
                f"src/embeddings/integer/embeddings_unclamped_{self.n_hidden_nodes}.pkl")
            print("Embeddings loaded.")
            # print("Embeddings calculated.")


        save_folder = self.speicherort + self.param_string
        os.makedirs(save_folder, exist_ok=True)
        print("Training with \n"
              f"batch size: {batch_size}\n",
              f"learning rate: {learning_rate}\n",
              f"hidden nodes: {self.n_hidden_nodes}\n",
              f"sample count: {self.sample_count}\n",
              f"beta eff: {self.beta_eff}\n",
              )

        for epoch in tqdm(range(1, self.epochs + 1), desc="Training", ncols=80):
            # print(f'Epoch {epoch}')
            batchnum = 1
            # batches = zip(batches_data, batches_labels)
            # batches = list(batches)
            # num_batches = len(batches)
            num_batches = len(train_X) // batch_size
            epoch_errors = 0
            epoch_nll = 0
            nll = torch.nn.NLLLoss()

            for b in tqdm(range(0, len(train_X), batch_size), desc=f"Training current epoch {epoch}", ncols=80, leave=False):
                if (b + batch_size) <= len(train_X):
                    x_batch = train_X[b:b + batch_size]  # [X_train[i] for i in range(b, b + batch_size)]
                    y_batch = train_Y[b:b + batch_size]
                else:
                    x_batch = train_X[b:]  # [X_train[i] for i in range(b, len(X_train))]
                    y_batch = train_Y[b:]

                if len(x_batch) == 0:
                    print("Batch is empty")
                    continue

                try:
                    output_bias_errors_batch, avg_nll_batch = self.train_for_one_iteration(x_batch, y_batch, learning_rate, nll)
                    avg_output_bias_errors_batch = np.mean(output_bias_errors_batch)
                    epoch_errors += avg_output_bias_errors_batch
                    epoch_nll += avg_nll_batch
                    self.training_history.errors_per_batch.append(avg_output_bias_errors_batch)
                    # self.save_weights(
                    # f'e{epoch}_b{batchnum}_{self.param_string}')
                    if epoch == 1 and batchnum == 1 and (self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1"):
                        print(f"QPU time used for one iteration: {self.qpu_time_used} microseconds")
                    batchnum += 1
                except Exception as e:
                    self.save_weights(
                        f'e{epoch}_b{batchnum}_{self.param_string}', save_folder)
                    metrics.save_history(f"{save_folder}/", self.training_history)
                    raise e
            self.save_weights(
                f'e{epoch}_{self.param_string}', save_folder)
            val_predictions = []
            for val_x in tqdm(val_X, desc="predict validation set", ncols=80, leave=False):
                prediction, _ = self.predict(val_x)
                val_predictions.append(prediction)

            acc, _, _, _, auc = metrics.get_metrics(val_Y, val_predictions, 2)
            combined_acc_auc = 0.5*acc + 0.5*auc
            self.training_history.acc_per_epoch.append(acc)
            self.training_history.auc_per_epoch.append(auc)
            self.training_history.combined_acc_auc_per_epoch.append(combined_acc_auc)

            avg_epoch_errors = epoch_errors / num_batches
            avg_epoch_nll = epoch_nll / num_batches

            self.training_history.error_per_epoch.append(avg_epoch_errors)
            self.training_history.nll_per_epoch.append(avg_epoch_nll)

            if self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1":
                print(f"QPU time used after {epoch} epochs: {self.qpu_time_used} microseconds")

        if self.solver_string == "Advantage_system4.1" or self.solver_string == "Advantage_system7.1":
            print(f"QPU time used for one training run: {self.qpu_time_used} microseconds")

        if self.solver_string == "SA":
            print(f"SA time used for one training run: {self.sa_time_used} microseconds")

        with open(f"{save_folder}/acc_per_epoch{self.seed}.pkl", "wb") as f:
            pickle.dump(self.training_history.acc_per_epoch, f)
        with open(f"{save_folder}/auc_per_epoch{self.seed}.pkl", "wb") as f:
            pickle.dump(self.training_history.auc_per_epoch, f)
        with open(f"{save_folder}/combined_acc_auc_per_epoch{self.seed}.pkl", "wb") as f:
            pickle.dump(self.training_history.combined_acc_auc_per_epoch, f)


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
            path = PurePath()
            path = Path(path / 'embeddings')
            path.mkdir(mode=0o770, exist_ok=True)
            plt.savefig("embeddings/embedding.pdf")

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
        samples = self.get_samples(data)

        np_samples = np.vstack(
            tuple([np.array(list(sample.values())) for sample in samples]))

        samples_of_output = np_samples[:, 0:self.n_output_nodes]
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

        total_samples = sum(sample_counts.values())

        probabilities = {k: v / total_samples for k, v in sample_counts.items()} if total_samples > 0 else {}

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
                        f"src/embeddings/integer/embeddings_clamped_{self.n_hidden_nodes}.pkl")

                # self.parallel_embeddings_unclamped = self.calcualte_parallel_embeddings(self.subgraphs, train_x[0])
                self.parallel_embeddings_unclamped = self.load_embeddings(
                    f"src/embeddings/integer/embeddings_unclamped_{self.n_hidden_nodes}.pkl")
                print("Embeddings loaded.")
                # print("Embeddings calculated.")


            print("Training with \n"
                  f"batch size: {batch_size}\n",
                  f"learning rate: {learning_rate}\n",
                  f"hidden nodes: {self.n_hidden_nodes}\n",
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


