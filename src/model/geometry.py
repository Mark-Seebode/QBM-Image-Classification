from __future__ import annotations
from typing import List, Tuple
import numpy as np



def conv_output_shape(
    image_shape: tuple[int, int],
    kernel_size: int,
    stride: int,
) -> tuple[int, int]:
    H, W = image_shape
    k, s = int(kernel_size), int(stride)
    oh = (H - k) // s + 1
    ow = (W - k) // s + 1

    return (oh, ow)


def get_input_groups_coords(
    image_shape: tuple[int, int],
    kernel_size: int,
    stride: int,
) -> List[tuple[np.ndarray, np.ndarray]]:

    H, W = image_shape
    k, s = int(kernel_size), int(stride)
    groups: List[tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, H - k + 1, s):
        rows = np.arange(i, i + k)
        for j in range(0, W - k + 1, s):
            cols = np.arange(j, j + k)
            groups.append((rows, cols))
    return groups


def conv2d_valid_stride(
    img2d: np.ndarray,
    kernel2d: np.ndarray,
    stride: int,
) -> np.ndarray:
    kh, kw = kernel2d.shape
    ih, iw = img2d.shape
    sh = sw = int(stride)
    out_h = (ih - kh) // sh + 1
    out_w = (iw - kw) // sw + 1
    out = np.empty((out_h, out_w), dtype=float)
    for i in range(out_h):
        ii = i * sh
        for j in range(out_w):
            jj = j * sw
            out[i, j] = np.sum(img2d[ii:ii + kh, jj:jj + kw] * kernel2d)
    return out


def build_pool_windows(
    conv_dim: tuple[int, int],
    pool_size: int,
) -> List[np.ndarray]:
    """
    Create non-overlapping p×p windows over the flattened conv fmap.
    Each window is returned as a 1D array of indices into the flattened fmap.
    If pool_size in (0, 1), returns an empty list.
    """
    if pool_size in (0, 1):
        return []
    H, W = conv_dim
    p = int(pool_size)
    wins: List[np.ndarray] = []
    for i in range(0, H - p + 1, p):
        for j in range(0, W - p + 1, p):
            idxs = []
            for di in range(p):
                for dj in range(p):
                    idxs.append((i + di) * W + (j + dj))
            wins.append(np.array(idxs, dtype=int))
    return wins



def num_conv_units_from_dim(conv_dim: tuple[int, int]) -> int:
    H, W = conv_dim
    return int(H * W)


def count_pooled_units(
    pooling_type: str,                 # "deterministic" | "probabilistic"
    pool_windows: List[np.ndarray],
    num_conv_units: int,
) -> int:
    # useful ?
    return len(pool_windows) if pool_windows else num_conv_units





