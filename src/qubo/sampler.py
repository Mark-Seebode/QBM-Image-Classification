# src/samplers.py
from __future__ import annotations
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

    def sample_Q(self, Q: np.ndarray) -> np.ndarray:
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
    def __init__(self, solver, api_token: str, groupQpuToken_name: str, num_reads: int, embedding=None, seed: int | None = None):
        self.solver = solver
        self.embedding = embedding
        self.num_reads = num_reads
        self.seed = seed
        try:
            self.backend = self.connect_to_luna(api_token, groupQpuToken_name)
            self.algorithm = self.prepare_algorithm()

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize D-Wave solver '{solver}'. "
                f"Use --solver SA or ensure TOKEN and solver name are valid. Error: {e}"
            )

    def sample_Q(self, Q: np.ndarray, num_reads: int) -> np.ndarray:
        bqm = _to_bqm(Q)
        if self.embedding is not None:
            from dwave.embedding import embed_bqm, EmbeddedStructure, unembed_sampleset
            embedded = embed_bqm(bqm, EmbeddedStructure(self.solver.edges, self.embedding))
            ss_e = self.solver.sample_bqm(embedded, num_reads=int(num_reads), answer_mode='raw').sampleset
            ss = unembed_sampleset(ss_e, self.embedding, bqm)
        else:
            ss = self.solver.sample_bqm(bqm, num_reads=int(num_reads), answer_mode='raw').sampleset
        return ss.record.sample.astype(np.float32)

    def connect_to_luna(self, api_token: str, groupQpuToken_name: str):
        LunaSolve.authenticate(api_token)

        group_qpu_token = GroupQpuToken(name=groupQpuToken_name)
        backend = DWaveQpu(token=group_qpu_token)

        return backend

    def prepare_algorithm(self):
        algorithm = QuantumAnnealing(
            backend=self.backend,
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

        return algorithm





# TODO: add embedding
