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
    num_filter_kernels: int

    @property
    def n_hidden(self) -> int:
        if self.pooling_type == "deterministic":
            sum = 0
            for c in self.conv_active:
                sum += c
            for s in self.seq:
                for size in s.sizes:
                    sum += size
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
    last_hidden: list[slice]
    out: slice                          # [n_hidden : n_hidden + n_out)


def build_conv_slices(spec: StackSpec):
    conv = []
    idx = 0
    for c in spec.conv_active:
        # convolutional layer slice
        conv_sl = slice(idx, idx + c)
        conv.append(conv_sl)
        idx += c
    return conv, idx


def build_pool_slices(spec: StackSpec, conv: list[slice], idx: int):
    pool = []
    for i in range(spec.num_filter_kernels):
        if spec.pooling_type == "deterministic":
            pool_sl = conv[i]
            pool.append(pool_sl)
            idx += spec.conv_active[i]
        elif spec.pooling_type == "probabilistic":
            raise NotImplementedError(
                "build_slices not implemented for probabilistic pooling with multiple recurrent layers")
            pool_sl = slice(spec.conv_active, spec.conv_active + spec.n_pooled_units)
            idx += spec.conv_active + spec.n_pooled_units
        else:
            raise ValueError(f"Unknown pooling_type: {spec.pooling_type}")

    return pool, idx


def build_seq_slices(spec: StackSpec, idx: int):
    seq_layers = []
    for recurrent_layer in range(spec.num_recurrent_layers):
        seq_slices: List[slice] = []
        for s in spec.seq:
            for size in s.sizes:
                seq_slices.append(slice(idx, idx + size))
                idx += size
        seq_layers.append(tuple(seq_slices))

    return seq_layers, idx


def build_slices(spec: StackSpec) -> BlockSlices:
    conv, idx = build_conv_slices(spec)
    pool, idx = build_pool_slices(spec, conv, idx)
    seq_layers, idx = build_seq_slices(spec, idx)
    hidden_sl = slice(0, idx)
    last_hidden_sl = last_hidden_slice(seq_layers, pool, spec.num_recurrent_layers)
    out_sl = slice(idx, idx + spec.n_out)

    return BlockSlices(conv=conv, pool=pool, seq_layers=seq_layers, hidden=hidden_sl, last_hidden=last_hidden_sl, out=out_sl)


def last_hidden_slice(seq_layers, pool_slices, num_recurrent_layers) -> list[slice]:
    last_hidden_sl = []
    if len(seq_layers[0]) > 0:
        for seq_layer in seq_layers:
            last_hidden_sl.append(seq_layer[-1])
    else:
       last_hidden_sl = pool_slices
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



