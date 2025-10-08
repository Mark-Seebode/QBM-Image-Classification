from dataclasses import dataclass
import numpy as np
from src.model.geometry import conv2d_valid_stride
from src.model.layers import (
    SeqSpec, StackSpec, build_slices, pooled_indices_for_input
)

@dataclass(frozen=True)
class InferenceContext:
    fmap_2d: np.ndarray
    fmap_flat: np.ndarray
    pooled_idx: np.ndarray
    spec: StackSpec
    slices: any  # BlockSlices

def prepare_context(model, x_img) -> InferenceContext:
    fmap_2d = conv2d_valid_stride(x_img, model.kernel_weights, model.stride)
    fmap_flat = fmap_2d.ravel()

    pooled_idx = pooled_indices_for_input(
        fmap_flat=fmap_flat,
        num_conv_units=model.num_conv_units,
        pooling_type=model.pooling_type,
        pool_windows=getattr(model, "pool_windows", []),
    )

    conv_active = len(pooled_idx) if model.pooling_type == "deterministic" else model.num_conv_units
    spec = StackSpec(
        conv_active=conv_active,
        seq=SeqSpec(tuple(model.sequential_layer_sizes)),
        n_out=int(model.num_lable_nodes),
        pooling_type=model.pooling_type,
        n_pooled_units=len(pooled_idx)
    )

    slices = build_slices(spec)

    return InferenceContext(
        fmap_2d=fmap_2d,
        fmap_flat=fmap_flat,
        pooled_idx=pooled_idx,
        spec=spec,
        slices=slices,
    )
