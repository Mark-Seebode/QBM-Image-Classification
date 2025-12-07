# src/samplers.py
from __future__ import annotations

from pathlib import PurePath, Path

import numpy as np
import dimod as di
from neal import SimulatedAnnealingSampler
from luna_quantum import LunaSolve
from luna_quantum.client.schemas.qpu_token.qpu_token import GroupQpuToken
from luna_quantum.solve.parameters.backends import DWaveQpu
from luna_quantum.translator import QuboTranslator
from luna_quantum._core import Vtype
from luna_quantum.solve.parameters.algorithms import QuantumAnnealing
from concurrent.futures import ProcessPoolExecutor
import logging
import networkx as nx
from dwave.cloud import Client
from dwave.embedding import embed_bqm, unembed_sampleset, EmbeddedStructure
import dwave.embedding as dwave_embedding
import dwave_networkx as dnx
import dwave_networkx
import minorminer
import matplotlib.pyplot as plt

import pickle

def _to_bqm(Q: np.ndarray) -> di.BQM:
    return di.BQM(Q, "BINARY")

def _is_linear_only(bqm: di.BQM) -> bool:
    return len(bqm.quadratic) == 0

def _solve_linear_only(bqm: di.BQM, num_reads: int, seed: int | None) -> di.SampleSet:
    rng = np.random.default_rng(seed)
    sol = {v: (1 if h < 0 else (0 if h > 0 else int(rng.integers(0, 2))))
           for v, h in bqm.linear.items()}
    return di.SampleSet.from_samples_bqm([sol] * int(num_reads), bqm)

class LocalSASampler:
    # TODO: parallel sampling support
    def __init__(self, num_reads: int, num_sweeps: int = 1000, parallelize=False, seed: int | None = None):
        if not parallelize:
            self.sa = SimulatedAnnealingSampler()
            self.sa_list = []
        else:
            self.executor = ProcessPoolExecutor(max_workers=10)
            sampler_0 = SimulatedAnnealingSampler()
            sampler_1 = SimulatedAnnealingSampler()
            sampler_2 = SimulatedAnnealingSampler()
            sampler_3 = SimulatedAnnealingSampler()
            sampler_4 = SimulatedAnnealingSampler()
            sampler_5 = SimulatedAnnealingSampler()
            sampler_6 = SimulatedAnnealingSampler()
            sampler_7 = SimulatedAnnealingSampler()
            sampler_8 = SimulatedAnnealingSampler()
            sampler_9 = SimulatedAnnealingSampler()
            self.sa_list = [sampler_0, sampler_1, sampler_2, sampler_3, sampler_4,
                                              sampler_5, sampler_6, sampler_7, sampler_8, sampler_9]
            self.sa = None
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.num_reads = num_reads

    def sample_Q(self, Q: np.ndarray, label=None) -> np.ndarray:
        bqm = _to_bqm(Q)
        if self.sa is None:
            tasks = [
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[0], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[1], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[2], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[3], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[4], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[5], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[6], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[7], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[8], self.seed),
                (bqm,  int(self.num_reads / 10), self.num_sweeps, self.sa_list[9], self.seed),

            ]
            # Use ProcessPoolExecutor to parallelize the sampling
            futures = [self.executor.submit(self.parallel_sa_sample, task) for task in tasks]
            sample_set = []
            for future in futures:
                sample_set.extend(future.result())
            sample_set = [list(s.values()) for s in sample_set]
            sample_set = np.array(sample_set)
        else:
            sample_set = self.sa.sample(bqm, num_reads=self.num_reads,
                                num_sweeps=self.num_sweeps, seed=self.seed)
            sample_set = sample_set.record.sample
        return sample_set

    @staticmethod
    def parallel_sa_sample(args):
        qubo_as_bqm,  sample_count, anneal_steps, sa_sampler, seed = args
        # qubo_as_bqm = di.BQM(qubo_matrix, "BINARY")
        # sa_sampler = SimulatedAnnealingSampler()
        return list(sa_sampler.sample(qubo_as_bqm, num_reads=sample_count, num_sweeps=anneal_steps, seed=seed).samples())


class DWaveAdapter:
    # TODO: PQA support
    def __init__(self, solver, api_token: str, dwave_token, groupQpuToken_name: str, num_reads: int, embedding=None, seed: int | None = None, luna: bool = True):
        self.solver_backend = solver
        self.solver = solver
        self.embedding = embedding
        self.num_reads = num_reads
        self.seed = seed
        self.TOKEN = api_token
        self.luna = luna
        self.embedding_clamped = None
        self.embedding_unclamped = None

        if luna:
            try:
                self.group_qpu_token = self.connect_to_luna(api_token, groupQpuToken_name)
                self.client = Client(token=dwave_token, solver=solver)
                self.solver_backend = self.client.get_solver(name=solver)
                print(f"Connected to D-Wave solver '{solver}' via Luna Quantum.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize D-Wave solver '{solver}'. "
                    f"Use --solver SA or ensure TOKEN and solver name are valid. Error: {e}"
                )
            self.clamped_backend = None
            self.unclamped_backend = None
            self.algorithm = None
        else:
            self.client = Client(token=dwave_token, solver=solver)
            # use an Advantage solver_backend (first generation -> with 5000 Qubits)
            self.solver_backend = self.client.get_solver(name=solver)

    def sample_Q(self, Q: np.ndarray, label=None) -> np.ndarray:
        qubo_as_bqm = _to_bqm(Q)
        if self.luna:
            if label is None:
                if self.unclamped_backend is None:
                    self.unclamped_backend = self.get_backend(self.group_qpu_token, label, qubo_as_bqm)
                    self.algorithm = self.prepare_algorithm(backend=self.unclamped_backend)
            else:
                if self.clamped_backend is None:
                    self.clamped_backend = self.get_backend(self.group_qpu_token, label, qubo_as_bqm)
                    self.algorithm = self.prepare_algorithm(backend=self.clamped_backend)

            return self.get_qa_samples_luna(Q)
        else:
            if label is None:
                this_embedding = self.find_embedding_with_client(
                    qubo_as_bqm, True, label) if self.embedding_unclamped is None else self.embedding_unclamped
                self.embedding_unclamped = this_embedding
            else:
                this_embedding = self.find_embedding_with_client(
                    qubo_as_bqm, False, label) if self.embedding_clamped is None else self.embedding_clamped
                self.embedding_clamped = this_embedding
            return self.get_qa_samples_Dwave(qubo_as_bqm, self.num_reads, this_embedding)


    def connect_to_luna(self, api_token: str, groupQpuToken_name: str):
        LunaSolve.authenticate(api_token)

        group_qpu_token = GroupQpuToken(name=groupQpuToken_name)
        return group_qpu_token


    def get_backend(self, group_qpu_token, label, qubo_as_bqm):
        # if label is None:
        #     this_embedding = self.find_embedding_with_client(
        #         qubo_as_bqm, True, label) if self.embedding_unclamped is None else self.embedding_unclamped
        # else:
        #     this_embedding = self.find_embedding_with_client(
        #         qubo_as_bqm, False, label) if self.embedding_clamped is None else self.embedding_clamped
        #
        # print(this_embedding)

        backend = DWaveQpu(qpu_backend=self.solver, token=group_qpu_token)

        return backend



    def prepare_algorithm(self, backend):
        algorithm = QuantumAnnealing(
            backend=backend,
            anneal_offsets=None,
            anneal_schedule=None,
            annealing_time=None,
            auto_scale=None,
            fast_anneal=False,
            flux_biases=None,
            flux_drift_compensation=True,
            h_gain_schedule=None,
            initial_state=None,
            max_answers=None,
            num_reads=self.num_reads,
            programming_thermalization=None,
            readout_thermalization=None,
            reduce_intersample_correlation=False,
            reinitialize_state=None
        )

        # Mute Luna's active waiting INFO spam
        #logging.getLogger("luna_quantum.util.active_waiting").setLevel(logging.WARNING)

        return algorithm

    def get_qa_samples_Dwave(self, qubo_as_bqm, sample_count, embedding):
            # uqo: problem.embedding = ...

        embedded_q = embed_bqm(source_bqm=qubo_as_bqm,
                                embedding=EmbeddedStructure(
                                target_edges=self.solver_backend.edges,
                                embedding=embedding
                                    )
                                )

        answer = self.run_qa_sampling_Dwave(embedded_q, embedding, qubo_as_bqm, sample_count)
        samples = list(answer.samples())
        # if self.current_batch_index == 50:
        #print(samples)
        #     raise Exception("stop")
        return np.array(samples)


    def run_qa_sampling_Dwave(self, embedded_bqm, this_embedding, source_bqm_unembedded, sample_count):
        try:
            embedded_answer = self.solver_backend.sample_bqm(embedded_bqm,
                                                             num_reads=sample_count,
                                                             answer_mode='raw'
                                                             ).sampleset
            #print(f"    QPU time used: {embedded_answer.info['timing']['qpu_access_time']} microseconds")
            #print("QPU time used: ", self.qpu_time_used)
            #raise Exception("Not implemented")
        except (
                ConnectionError, ConnectionResetError, ConnectionAbortedError,
                ConnectionRefusedError):
            self.refresh_connection()
            embedded_answer = self.solver_backend.sample_bqm(embedded_bqm,
                                                             num_reads=sample_count,
                                                             answer_mode='raw'
                                                             ).sampleset
        answer = unembed_sampleset(target_sampleset=embedded_answer,
                                   embedding=this_embedding,
                                   source_bqm=source_bqm_unembedded)

        return answer


    def get_qa_samples_luna(self,Q):
        model = QuboTranslator.to_aq(Q, name="CDQBM QUBO", vtype=Vtype.Binary)
        solve_job = self.algorithm.run(model, name="test-qubo123")
        solution = solve_job.result()
        samples = solution.samples.tolist()

        return np.array(samples)


    def find_embedding_with_client(self, bqm, save, label = None):
        if bqm.quadratic == {}:
            qubo_graph = nx.Graph([(0, 0)])
            target_edges = qubo_graph.edges
        else:
            target_edges = bqm.quadratic
        embedding, embedding_found = minorminer.find_embedding(target_edges,
                                                               self.solver_backend.edges,
                                                               return_overlap=True,
                                                               random_seed=self.seed,
                                                               threads=4
                                                               )
        while not embedding_found:
            print("No embedding found. Trying again...")
            embedding, embedding_found = minorminer.find_embedding(
                target_edges, self.solver_backend.edges, return_overlap=True
            )
        # adapted from: https://support.dwavesys.com/hc/en-us/community
        # /posts/1500001417242-Solved-How-to-find-out-the-number-of-qubits
        # -used-to-solve-a-problem
        # print(f"Number of logical variables: {len(embedding.keys())}")
        # print(f"Number of physical qubits used in embedding: "
        #       f"{sum(len(chain) for chain in embedding.values())}"
        #         )


        if self.solver == 'Advantage_system4.1' or self.solver == 'Advantage_system7.1':
            dwave_networkx.draw_pegasus_embedding(dwave_networkx.pegasus_graph(16), emb=embedding, node_size=3,
                                                      width=.3)
        elif self.solver == 'Advantage2_system1.8':
            dwave_networkx.draw_zephyr_embedding(dwave_networkx.zephyr_graph(16, 4), emb=embedding,
                                                      node_size=3, width=.3)

        plt.show()
        #path = PurePath()
       ## path = Path(path / 'embeddings')
        #path.mkdir(mode=0o770, exist_ok=True)
        #plt.savefig("embeddings/cdqbm_embedding10f3k24168.png", transparent=True)

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
        solver_id = self.solver_backend.id
        self.client.close()
        # get new connection to client
        self.client = Client(token=self.TOKEN, solver=solver_id)
        # make sure to get the same solver_backend from this connection
        self.solver_backend = self.client.get_solver(name=solver_id)






# TODO: add embedding
