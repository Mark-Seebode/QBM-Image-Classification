from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class SeqSpec:
    sizes: Tuple[int, ...]

@dataclass(frozen=True)
class StackSpec:
    conv_active: list[int]        # number of active conv units
    seq: list[SeqSpec]            # sequential layer sizes
    n_out: int                    # output nodes (1 or #classes)
    pooling_type: str             # "deterministic" | "probabilistic"
    n_pooled_units: list[int]
    num_recurrent_layers: int

    @property
    def n_hidden(self) -> int:
        if self.pooling_type == "deterministic":
            sum = 0
            for recurrent_layer in range(self.num_recurrent_layers):
                sum += self.conv_active[recurrent_layer]
                for s in self.seq[recurrent_layer].sizes:
                    sum += s
            return sum
        elif self.pooling_type == "probabilistic":
            raise NotImplementedError("n_hidden property not implemented for probabilistic pooling")
            return self.conv_active + self.n_pooled_units + sum(self.seq.sizes)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")


@dataclass(frozen=True)
class BlockSlices:
    conv: list[slice]                 # [0 : conv_active)
    pool: list[slice]                 # [conv_active : conv_active + n_pooled) only if probabilistic else = conv
    seq_layers: list[Tuple[slice, ...]]
    hidden: slice                     # [0 : n_hidden) n_hidden = everything beside out
    out: slice                        # [n_hidden : n_hidden + n_out)

def build_slices(spec: StackSpec) -> BlockSlices:
    conv = []
    pool = []
    seq_layers = []
    cur = 0
    for recurrent_layer in range(spec.num_recurrent_layers):
        # convolutional layer slice
        conv_sl = slice(cur, cur + spec.conv_active[recurrent_layer])
        conv.append(conv_sl)

        if spec.pooling_type == "deterministic":
            pool_sl = conv_sl
            pool.append(pool_sl)
            cur += spec.conv_active[recurrent_layer]
        elif spec.pooling_type == "probabilistic":
            raise NotImplementedError ("build_slices not implemented for probabilistic pooling with multiple recurrent layers")
            pool_sl = slice(spec.conv_active, spec.conv_active + spec.n_pooled_units)
            cur += spec.conv_active + spec.n_pooled_units
        else:
            raise ValueError(f"Unknown pooling_type: {spec.pooling_type}")

        seq_slices: List[slice] = []
        for s in spec.seq[recurrent_layer].sizes:
            seq_slices.append(slice(cur, cur + s))
            cur += s
        seq_layers.append(tuple(seq_slices))

    hidden_sl = slice(0, cur)
    out_sl = slice(cur, cur + spec.n_out)

    return BlockSlices(conv=conv, pool=pool, seq_layers=seq_layers, hidden=hidden_sl, out=out_sl)



def last_hidden_slice(slices: BlockSlices) -> list[slice]:
    last_hidden_sl = []
    if slices.seq_layers:
        for seq_layer in slices.seq_layers:
            last_hidden_sl.append(seq_layer[-1])
    else:
       last_hidden_sl = slices.conv
    return last_hidden_sl



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



