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

def prepare_context(model, x_img) -> InferenceContext:
    fmap_2d = []
    fmap_flat = []
    pooled_idexes = []

    for recurrent_layer in range(model.num_recurrent_layers):
        currentlayer_fmap_2d = conv2d_valid_stride(x_img, model.kernel_weights[recurrent_layer], model.stride)
        currentlayer_fmap_flat = currentlayer_fmap_2d.ravel()
        fmap_2d.append(currentlayer_fmap_2d)
        fmap_flat.append(currentlayer_fmap_flat)

        pooled_idx = pooled_indices_for_input(
            fmap_flat=currentlayer_fmap_flat,
            num_conv_units=model.num_conv_units,
            pooling_type=model.pooling_type,
            pool_windows=getattr(model, "pool_windows", []),
        )

        pooled_idexes.append(pooled_idx)

    return InferenceContext(
        fmap_2d=fmap_2d,
        fmap_flat=fmap_flat,
        pooled_idx=pooled_idexes
    )
