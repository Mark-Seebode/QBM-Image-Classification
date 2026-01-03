from dataclasses import dataclass
import numpy as np
from src.model import geometry as geom
from src.model.cdqbm_state import Conv_Deep_QBM
from src.model.layers import (
    SeqSpec, StackSpec, build_slices, pooled_indices_for_input
)

@dataclass(frozen=True)
class InferenceContext:
    fmap_2d: list[np.ndarray]
    fmap_flat: list[np.ndarray]
    pooled_idx: list[np.ndarray]

def prepare_context(model: Conv_Deep_QBM, x_img) -> InferenceContext:
    fmap_2d = []
    fmap_flat = []
    pooled_idexes = []

    for fk in range(model.num_filter_kernels):
        currentlayer_fmap_2d = geom.conv2d_valid_stride(x_img, model.kernel_weights[fk], model.stride)
        currentlayer_fmap_flat = currentlayer_fmap_2d.ravel()
        fmap_2d.append(currentlayer_fmap_2d)
        fmap_flat.append(currentlayer_fmap_flat)

        if model.pooling_type == 'deterministic':
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


def prepare_context_fully_connected(model, x_img) -> InferenceContext:
    x_flat = x_img.flatten()
    fmap_2d = np.matmul(x_flat, model.kernel_weights)
    fmap_flat = fmap_2d.flatten()

    return InferenceContext(
        fmap_2d=fmap_2d,
        fmap_flat=fmap_flat,
        pooled_idx=None
    )


