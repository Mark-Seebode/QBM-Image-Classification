from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class SeqSpec:
    sizes: Tuple[int, ...]

@dataclass(frozen=True)
class StackSpec:
    conv_active: int        # number of active conv units
    seq: SeqSpec            # sequential layer sizes
    n_out: int              # output nodes (1 or #classes)
    pooling_type: str       # "deterministic" | "probabilistic"
    n_pooled_units: int

    @property
    def n_hidden(self) -> int:
        if self.pooling_type == "deterministic":
            return self.conv_active + sum(self.seq.sizes)
        elif self.pooling_type == "probabilistic":
            return self.conv_active + self.n_pooled_units + sum(self.seq.sizes)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")


@dataclass(frozen=True)
class BlockSlices:
    conv: slice                 # [0 : conv_active)
    pool: slice                 # [conv_active : conv_active + n_pooled) only if probabilistic else = conv
    seq_layers: Tuple[slice, ...]
    hidden: slice               # [0 : n_hidden) n_hidden = everything beside out
    out: slice                  # [n_hidden : n_hidden + n_out)

def build_slices(spec: StackSpec) -> BlockSlices:
    cur = 0
    conv_sl = slice(cur, cur + spec.conv_active)

    if spec.pooling_type == "deterministic":
        pool_sl = conv_sl
        cur += spec.conv_active
    elif spec.pooling_type == "probabilistic":
        pool_sl = slice(spec.conv_active, spec.conv_active + spec.n_pooled_units)
        cur += spec.conv_active + spec.n_pooled_units
    else:
        raise ValueError(f"Unknown pooling_type: {spec.pooling_type}")

    seq_slices: List[slice] = []
    for s in spec.seq.sizes:
        seq_slices.append(slice(cur, cur + s))
        cur += s

    hidden_sl = slice(0, cur)
    out_sl = slice(cur, cur + spec.n_out)

    return BlockSlices(conv=conv_sl, pool=pool_sl, seq_layers=tuple(seq_slices), hidden=hidden_sl, out=out_sl)

def last_hidden_slice(slices: BlockSlices) -> slice:
    return slices.seq_layers[-1] if slices.seq_layers else slices.conv



def pooled_indices_for_input(
    fmap_flat: np.ndarray,
    num_conv_units: int,
    pooling_type: str,                 # "deterministic" | "probabilistic"
    pool_windows: List[np.ndarray] | None,
) -> np.ndarray:

    if pooling_type == "probabilistic":
        start = num_conv_units
        end = start + len(pool_windows)
        return np.arange(start, end, dtype=int)

    if not pool_windows:  # no windows configured -> keep all
        return np.arange(num_conv_units, dtype=int)

    picks: List[int] = []
    for win in pool_windows:
        ids = np.asarray(win, dtype=int)
        picks.append(int(ids[np.argmin(fmap_flat[ids])]))
    return np.asarray(picks, dtype=int)



