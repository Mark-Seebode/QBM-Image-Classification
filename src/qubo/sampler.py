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
    def __init__(self, num_reads: int, num_sweeps: int = 1000, seed: int | None = None):
        self.sa = SimulatedAnnealingSampler()
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.num_reads = num_reads

    def sample_Q(self, Q: np.ndarray) -> np.ndarray:
        bqm = _to_bqm(Q)
        if _is_linear_only(bqm):
            sample_set = _solve_linear_only(bqm, self.num_reads, self.seed)
        else:
            sample_set = self.sa.sample(bqm, num_reads=self.num_reads,
                                num_sweeps=self.num_sweeps, seed=self.seed)
        return sample_set.record.sample.astype(np.float32)

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
