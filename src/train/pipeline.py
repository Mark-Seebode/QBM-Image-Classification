import numpy as np
from dataclasses import dataclass
from src.model.inference import prepare_context
from src.qubo.builder import build_unclamped_qubo, build_clamped_qubo
from src.model.layers import last_hidden_slice as _last

@dataclass(frozen=True)
class RunOutputs:
    samples: np.ndarray
    probs:   np.ndarray | None
    ctx:     any

def run_unclamped(model, x_img,  num_reads: int, beta_eff: float,
                  one_hot: bool) -> RunOutputs:
    ctx = prepare_context(model, x_img)
    # convenience so builder can use it
    object.__setattr__(ctx, "last_hidden_slice", _last(ctx.slices))

    Q = build_unclamped_qubo(model, ctx, beta_eff)
    samples = model.sampler.sample_Q(Q, num_reads)

    out = samples[:, ctx.slices.out].mean(axis=0)
    if not one_hot:
        p1 = float(out[0]); p1 = min(max(p1, 1e-12), 1-1e-12)
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        s = float(out.sum())
        probs = (out / s).astype(np.float32) if s > 0 else np.full_like(out, 1/len(out))
    return RunOutputs(samples=samples, probs=probs, ctx=ctx)

def run_clamped(model, x_img, label_vec, num_reads: int, beta_eff: float) -> RunOutputs:
    ctx = prepare_context(model, x_img)
    object.__setattr__(ctx, "last_hidden_slice", _last(ctx.slices))
    Q = build_clamped_qubo(model, ctx, np.asarray(label_vec, float), beta_eff)
    samples = model.sampler.sample_Q(Q, num_reads)
    return RunOutputs(samples=samples, probs=None, ctx=ctx)
