# src/samplers.py
from __future__ import annotations
import numpy as np
import dimod as di
from neal import SimulatedAnnealingSampler

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
    def __init__(self, num_sweeps: int = 1000, seed: int | None = None):
        self.sa = SimulatedAnnealingSampler()
        self.num_sweeps = int(num_sweeps)
        self.seed = seed

    def sample_Q(self, Q: np.ndarray, num_reads: int) -> np.ndarray:
        bqm = _to_bqm(Q)
        if _is_linear_only(bqm):
            ss = _solve_linear_only(bqm, num_reads, self.seed)
        else:
            ss = self.sa.sample(bqm, num_reads=int(num_reads),
                                num_sweeps=self.num_sweeps, seed=self.seed)
        return ss.record.sample.astype(np.float32)

class DWaveAdapter:
    # TODO: PQA support
    def __init__(self, solver, embedding=None, seed: int | None = None):
        self.solver = solver
        self.embedding = embedding
        self.seed = seed

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


# TODO: add embedding
