import numpy as np

def _conv_linear_terms(model, ctx) -> np.ndarray:
    """Return linear biases for the conv block"""
    bases = []
    if model.pooling_type == "deterministic":
        for recurrent_layer in range(model.num_recurrent_layers):
            base = ctx.fmap_flat[recurrent_layer][ctx.pooled_idx]
            if model.hidden_bias_type == "shared":
                base = base + model.biases_conv_units[recurrent_layer][0]
            elif model.hidden_bias_type != "none":
                base = base
            bases.append(base)
        return np.concatenate(bases, axis=0)
    raise NotImplementedError("Not implemented for probabilistic pooling")
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
        raise NotImplementedError("Unclamped QUBO with probabilistic pooling not implemented")
        Q = add_at_most_one_penalty_upper(model, Q, 0.8225)
        Q = add_link_penalty_upper(model, Q, ctx, 0.8225)

    # Conv
    conv_biases = _conv_linear_terms(model, ctx)
    for recurrent_layer in range(model.num_recurrent_layers):
        conv_bias = conv_biases[recurrent_layer]
        Q[ctx.slices.conv[recurrent_layer], ctx.slices.conv[recurrent_layer]] += np.diag(conv_bias)

        # Sequential
        prev_sl = ctx.slices.pool[recurrent_layer]
        for li, cur_sl in enumerate(ctx.slices.seq_layers[recurrent_layer]):
            W = model.weights_sequential_layer[recurrent_layer][li]
            Q[prev_sl, cur_sl] += W
            prev_sl = cur_sl

        # within-layer
        if len(model.weights_interlayer_sequential) > 0:
            for li, cur_sl in enumerate(ctx.slices.seq_layers[recurrent_layer]):
                Q[cur_sl, cur_sl] += np.triu(model.weights_interlayer_sequential[recurrent_layer][li], k=1)

    # between-layer recurrent
    for recurrent_layer in range(model.num_recurrent_layers - 1):
        for seq_layer in range(len(ctx.slices.seq_layers[recurrent_layer])):
            cur_sl = ctx.slices.seq_layers[recurrent_layer][seq_layer]
            next_sl = ctx.slices.seq_layers[recurrent_layer + 1][seq_layer]
            W_rec = model.weights_seq_recurrent[recurrent_layer][seq_layer]
            Q[cur_sl, next_sl] += W_rec
    for seq_layer in range(len(ctx.slices.seq_layers[0])):
        cur_sl = ctx.slices.seq_layers[0][seq_layer]
        next_sl = ctx.slices.seq_layers[-1][seq_layer]
        W_rec = model.weights_seq_recurrent[-1][seq_layer]
        Q[cur_sl, next_sl] += W_rec

    # Hidden biases sequential
    if model.biases_sequential_units.size:
        num_units_before_seq = ctx.spec.conv_active[0] * model.num_recurrent_layers
        if model.pooling_type == "probabilistic":
            raise NotImplementedError("Probabilistic QUBO with probabilistic pooling not implemented")
            num_units_before_seq = ctx.spec.conv_active + ctx.spec.n_pooled_units
        zeros_conv = np.zeros(num_units_before_seq, dtype=float)
        hid_bias = np.concatenate([zeros_conv, model.biases_sequential_units], axis=0)
        Q[ctx.slices.hidden, ctx.slices.hidden] += np.diag(hid_bias)

    # Hidden -> Output
    last_sl = ctx.last_hidden_slice
    for recurrent_layer in range(model.num_recurrent_layers):
        W_hy = model.weights_hidden_to_output[recurrent_layer]
        last_len = last_sl[recurrent_layer].stop - last_sl[recurrent_layer].start
        if W_hy.shape[0] != last_len:
            if model.pooling_type == "deterministic" and last_sl[recurrent_layer] == ctx.slices.conv[recurrent_layer]:
                W_hy = W_hy[np.asarray(ctx.pooled_idx[recurrent_layer], dtype=int), :]
            elif ctx.hidden_row_map is not None:
                raise NotImplementedError("Unclamped QUBO with probabilistic pooling not implemented")
                W_hy = W_hy[np.asarray(ctx.hidden_row_map, dtype=int), :]
            else:
                raise ValueError()
        Q[last_sl[recurrent_layer], ctx.slices.out] += W_hy

    # output
    Q[ctx.slices.out, ctx.slices.out] += np.triu(model.weights_output_output, k=1)
    Q[ctx.slices.out, ctx.slices.out] += np.diag(model.biases_output)

    return Q / float(beta_eff)


def build_clamped_qubo(model, ctx, label_vec: np.ndarray, beta_eff: float) -> np.ndarray:
    n = ctx.spec.n_hidden
    Q = np.zeros((n, n), dtype=float)

    if model.pooling_type == "probabilistic":
        NotImplementedError("Probabilistic QUBO with probabilistic pooling not implemented")
        Q = add_at_most_one_penalty_upper(model, Q, 0.8225)
        Q = add_link_penalty_upper(model, Q,  ctx, 0.8225)

    # Conv
    conv_biases = _conv_linear_terms(model, ctx)
    for recurrent_layer in range(model.num_recurrent_layers):
        conv_bias = conv_biases[recurrent_layer]
        Q[ctx.slices.conv[recurrent_layer], ctx.slices.conv[recurrent_layer]] += np.diag(conv_bias)
        # Sequential
        prev_sl = ctx.slices.pool[recurrent_layer]
        for li, cur_sl in enumerate(ctx.slices.seq_layers[recurrent_layer]):
            W = model.weights_sequential_layer[recurrent_layer][li]
            Q[prev_sl, cur_sl] += W
            prev_sl = cur_sl

        # within-layer
        if len(model.weights_interlayer_sequential) > 0:
            for li, cur_sl in enumerate(ctx.slices.seq_layers[recurrent_layer]):
                Q[cur_sl, cur_sl] += np.triu(model.weights_interlayer_sequential[recurrent_layer][li], k=1)

    # between-layer recurrent
    for recurrent_layer in range(model.num_recurrent_layers - 1):
        for seq_layer in range(len(ctx.slices.seq_layers[recurrent_layer])):
            cur_sl = ctx.slices.seq_layers[recurrent_layer][seq_layer]
            next_sl = ctx.slices.seq_layers[recurrent_layer + 1][seq_layer]
            W_rec = model.weights_seq_recurrent[recurrent_layer][seq_layer]
            Q[cur_sl, next_sl] += W_rec
    for seq_layer in range(len(ctx.slices.seq_layers[0])):
        cur_sl = ctx.slices.seq_layers[0][seq_layer]
        next_sl = ctx.slices.seq_layers[-1][seq_layer]
        W_rec = model.weights_seq_recurrent[-1][seq_layer]
        Q[cur_sl, next_sl] += W_rec

    # Hidden biases sequential
    if model.biases_sequential_units.size:
        num_units_before_seq = ctx.spec.conv_active[0] * model.num_recurrent_layers
        if model.pooling_type == "probabilistic":
            raise NotImplementedError("Probabilistic QUBO with probabilistic pooling not implemented")
            num_units_before_seq = ctx.spec.conv_active + ctx.spec.n_pooled_units
        zeros_conv = np.zeros(num_units_before_seq, dtype=float)
        hid_bias = np.concatenate([zeros_conv, model.biases_sequential_units], axis=0)
        Q[ctx.slices.hidden, ctx.slices.hidden] += np.diag(hid_bias)

    # label bias
    for recurrent_layer in range(model.num_recurrent_layers):
        last_sl = ctx.last_hidden_slice[recurrent_layer]
        eff = (model.weights_hidden_to_output[recurrent_layer] @ label_vec.reshape(-1, 1)).reshape(-1)
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
