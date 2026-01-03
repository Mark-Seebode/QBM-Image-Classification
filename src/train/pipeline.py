import numpy as np
from dataclasses import dataclass

from src.model.cdqbm_state import Conv_Deep_QBM
from src.model.inference import prepare_context, prepare_context_fully_connected
from src.qubo.builder import build_unclamped_qubo, build_clamped_qubo
from src.model.layers import last_hidden_slice as _last

class RunOutputs:
    samples: np.ndarray
    probs:   np.ndarray | None
    ctx:     any

    def __init__(self, samples: np.ndarray, probs: np.ndarray | None, ctx: any):
        self.samples = samples
        self.probs = probs
        self.ctx = ctx

def run_unclamped(model: Conv_Deep_QBM, x_img, beta_eff: float,
                  one_hot: bool, do_conv_label_bias=False, label_vec=None) -> RunOutputs:

    if model.kernel_size > 0:
        ctx = prepare_context(model, x_img)
    else:
        ctx = prepare_context_fully_connected(model, x_img)

    Q = build_unclamped_qubo(model, ctx, beta_eff, do_conv_label_bias, label_vec)
    samples = model.sampler.sample_Q(Q)

    out = samples[:, model.slices.out].mean(axis=0)
    if not one_hot:
        p1 = float(out[0]); p1 = min(max(p1, 1e-12), 1-1e-12)
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        s = float(out.sum())
        probs = (out / s).astype(np.float32) if s > 0 else np.full_like(out, 1/len(out))
    return RunOutputs(samples=samples, probs=probs, ctx=ctx)

def run_clamped(model, x_img, label_vec, beta_eff: float, do_conv_label_bias=False) -> RunOutputs:
    if model.kernel_size > 0:
        ctx = prepare_context(model, x_img)
    else:
        ctx = prepare_context_fully_connected(model, x_img)

    Q = build_clamped_qubo(model, ctx, np.asarray(label_vec, float), beta_eff, do_conv_label_bias)
    samples = model.sampler.sample_Q(Q, label_vec)
    return RunOutputs(samples=samples, probs=None, ctx=ctx)






