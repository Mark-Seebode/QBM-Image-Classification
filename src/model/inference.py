from dataclasses import dataclass
import numpy as np
from src.model.geometry import conv2d_valid_stride
from src.model.layers import (
    SeqSpec, StackSpec, build_slices, pooled_indices_for_input
)

@dataclass(frozen=True)
class InferenceContext:
    fmap_2d: list[np.ndarray]
    fmap_flat: list[np.ndarray]
    pooled_idx: list[np.ndarray]
    spec: StackSpec
    slices: any  # BlockSlices

def prepare_context(model, x_img) -> InferenceContext:
    fmap_2d = []
    fmap_flat = []
    conv_active = []
    seq = []
    pooled_idexes = []
    num_pooled_units_per_recurrent_layer = []

    for recurrent_layer in range(model.num_recurrent_layers):
        currentlayer_fmap_2d = conv2d_valid_stride(x_img, model.kernel_weights, model.stride)
        currentlayer_fmap_flat = currentlayer_fmap_2d.ravel()
        fmap_2d.append(currentlayer_fmap_2d)
        fmap_flat.append(currentlayer_fmap_flat)

        pooled_idx = pooled_indices_for_input(
            fmap_flat=currentlayer_fmap_flat,
            num_conv_units=model.num_conv_units,
            pooling_type=model.pooling_type,
            pool_windows=getattr(model, "pool_windows", []),
        )

        conv_active.append(len(pooled_idx) if model.pooling_type == "deterministic" else model.num_conv_units)
        seq.append(SeqSpec(model.sequential_layer_sizes))
        pooled_idexes.append(pooled_idx)
        num_pooled_units_per_recurrent_layer.append(len(pooled_idx))

    spec = StackSpec(
        conv_active=conv_active,
        seq=seq,
        n_out=model.num_lable_nodes,
        pooling_type=model.pooling_type,
        n_pooled_units=num_pooled_units_per_recurrent_layer,
        num_recurrent_layers=model.num_recurrent_layers
    )

    slices = build_slices(spec)

    # TODO: do not build spec and slices every time

    return InferenceContext(
        fmap_2d=fmap_2d,
        fmap_flat=fmap_flat,
        pooled_idx=pooled_idexes,
        spec=spec,
        slices=slices,
    )
