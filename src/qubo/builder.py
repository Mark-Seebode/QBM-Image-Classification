import numpy as np

def _conv_linear_terms(model, ctx) -> np.ndarray:
    """Return linear biases for the conv block"""
    if model.pooling_type == "deterministic":
        base = ctx.fmap_flat[ctx.pooled_idx]
        if model.hidden_bias_type == "shared":
            base = base + float(model.biases_conv_units[0])
        elif model.hidden_bias_type != "none":
            base = base
        return base
    # probabilistic -> all conv units active
    base = ctx.fmap_flat.copy()
    if model.hidden_bias_type == "shared":
        v = float(model.biases_conv_units[0]) if model.biases_conv_units.ndim else float(model.biases_conv_units)
        base = base + v
    elif model.hidden_bias_type != "none":
        base = base
    return base

def build_unclamped_qubo(model, ctx, beta_eff: float) -> np.ndarray:
    n = ctx.spec.n_hidden + ctx.spec.n_out
    Q = np.zeros((n, n), dtype=float)

    if model.pooling_type == "probabilistic":
        Q = add_at_most_one_penalty_upper(model, Q, 0.8225)
        Q = add_link_penalty_upper(model, Q, ctx, 0.8225)

    # Conv
    conv_bias = _conv_linear_terms(model, ctx)
    Q[ctx.slices.conv, ctx.slices.conv] += np.diag(conv_bias)

    # Sequential
    prev_sl = ctx.slices.pool
    for li, cur_sl in enumerate(ctx.slices.seq_layers):
        W = model.weights_sequential_layer[li]
        Q[prev_sl, cur_sl] += W
        prev_sl = cur_sl

    # within-layer
    if model.weights_interlayer_sequential is not None:
        for li, cur_sl in enumerate(ctx.slices.seq_layers):
            Q[cur_sl, cur_sl] += np.triu(model.weights_interlayer_sequential[li], k=1)

    # Hidden biases sequential
    if model.biases_sequential_units.size:
        num_units_before_seq = ctx.spec.conv_active
        if model.pooling_type == "probabilistic":
            num_units_before_seq = ctx.spec.conv_active + ctx.spec.n_pooled_units
        zeros_conv = np.zeros(num_units_before_seq, dtype=float)
        hid_bias = np.concatenate([zeros_conv, model.biases_sequential_units], axis=0)
        Q[ctx.slices.hidden, ctx.slices.hidden] += np.diag(hid_bias)

    # Hidden -> Output
    last_sl = ctx.last_hidden_slice
    W_hy = model.weights_hidden_to_output
    last_len = last_sl.stop - last_sl.start
    if W_hy.shape[0] != last_len:
        if model.pooling_type == "deterministic" and last_sl == ctx.slices.conv:
            W_hy = W_hy[np.asarray(ctx.pooled_idx, dtype=int), :]
        elif ctx.hidden_row_map is not None:
            W_hy = W_hy[np.asarray(ctx.hidden_row_map, dtype=int), :]
        else:
            raise ValueError()
    Q[last_sl, ctx.slices.out] += W_hy

    # output
    Q[ctx.slices.out, ctx.slices.out] += np.triu(model.weights_output_output, k=1)
    Q[ctx.slices.out, ctx.slices.out] += np.diag(model.biases_output)

    return Q / float(beta_eff)


def build_clamped_qubo(model, ctx, label_vec: np.ndarray, beta_eff: float) -> np.ndarray:
    n = ctx.spec.n_hidden
    Q = np.zeros((n, n), dtype=float)

    if model.pooling_type == "probabilistic":
        Q = add_at_most_one_penalty_upper(model, Q, 0.8225)
        Q = add_link_penalty_upper(model, Q,  ctx, 0.8225)

    # Conv
    conv_bias = _conv_linear_terms(model, ctx)
    Q[ctx.slices.conv, ctx.slices.conv] += np.diag(conv_bias)

    # Sequential connections
    prev_sl = ctx.slices.pool
    for li, cur_sl in enumerate(ctx.slices.seq_layers):
        Q[prev_sl, cur_sl] += model.weights_sequential_layer[li]
        prev_sl = cur_sl

    if model.weights_interlayer_sequential is not None:
        for li, cur_sl in enumerate(ctx.slices.seq_layers):
            Q[cur_sl, cur_sl] += np.triu(model.weights_interlayer_sequential[li], k=1)

    # Hidden biases for sequential
    if model.biases_sequential_units.size:
        num_units_before_seq = ctx.spec.conv_active
        if model.pooling_type == "probabilistic":
            num_units_before_seq = ctx.spec.conv_active + ctx.spec.n_pooled_units
        zeros_conv = np.zeros(num_units_before_seq, dtype=float)
        hid_bias = np.concatenate([zeros_conv, model.biases_sequential_units], axis=0)
        Q[ctx.slices.hidden, ctx.slices.hidden] += np.diag(hid_bias)

    # label bias
    last_sl = ctx.last_hidden_slice
    eff = (model.weights_hidden_to_output @ label_vec.reshape(-1, 1)).reshape(-1)
    Q[last_sl, last_sl] += np.diag(eff)

    return Q / float(beta_eff)


import numpy as np


def add_at_most_one_penalty_upper(model, qubo, penalty):
    # pairwise penalty for each group for at most one active per pool window
    for g in model.pool_windows:
        ids = np.asarray(g, dtype=int)
        m = ids.size
        if m <= 1:
            continue
        ii, jj = np.triu_indices(m, k=1)
        qubo[ids[ii], ids[jj]] += penalty
    return qubo


def add_link_penalty_upper(model, qubo: np.ndarray, ctx, penalty_B: float):
    # linking through logical OR
    p_start = ctx.pooled_idx[0] # first pooling var index
    for g_idx, g in enumerate(model.pool_windows):
        p = p_start + g_idx
        ids = np.asarray(g, dtype=int)
        if ids.size == 0:
            qubo[p, p] += penalty_B
            continue

        qubo[p, p] += penalty_B
        qubo[ids, ids] += penalty_B

        lo = np.minimum(ids, p)
        hi = np.maximum(ids, p)
        mask = lo != hi
        if np.any(mask):
            qubo[lo[mask], hi[mask]] += -2.0 * penalty_B

    return qubo
