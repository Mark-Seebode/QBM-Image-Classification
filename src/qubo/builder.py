import numpy as np

from src.model.cdqbm_state import Conv_Deep_QBM


def _conv_linear_terms(model, ctx) -> np.ndarray:
    """Return linear biases for the conv block"""
    bases = []
    target_std = 0.5
    if model.pooling_type == "deterministic":
        for fk in range(model.num_filter_kernels):
            base = ctx.fmap_flat[fk][ctx.pooled_idx[fk]]

            # zero-mean, unit-variance
            # mu = base.mean()
            # sigma = base.std() + 1e-6
            # base = (base - mu) / sigma
            # base = base * target_std

            if model.hidden_bias_type == "shared":
                base = base + model.biases_conv_units[fk][0]
            bases.append(base)
            # elif model.hidden_bias_type != "none":
            #     base = base
            # bases.append(base)
        return np.array(bases)
    raise NotImplementedError("Not implemented for probabilistic pooling")
    # probabilistic -> all conv units active
    base = ctx.fmap_flat.copy()
    if model.hidden_bias_type == "shared":
        v = float(model.biases_conv_units[0]) if model.biases_conv_units.ndim else float(model.biases_conv_units)
        base = base + v
    elif model.hidden_bias_type != "none":
        base = base
    return base

def add_conv_biases(model, Q, ctx):
    conv_biases = _conv_linear_terms(model, ctx)
    for fk in range(model.num_filter_kernels):
        conv_bias = conv_biases[fk]
        conv_bias = np.diag(conv_bias)
        Q[model.slices.conv[fk], model.slices.conv[fk]] += conv_bias

    return Q

def add_seq_recurrent_weights(model, Q):
    for recurrent_layer in range(model.num_filter_kernels):
        prev_sl = model.slices.pool[recurrent_layer]
        for li, cur_sl in enumerate(model.slices.seq_layers[recurrent_layer]):
            W = model.weights_sequential_layer[recurrent_layer][li]
            Q[prev_sl, cur_sl] += W
            prev_sl = cur_sl

            # within-layer
            if not model.is_restricted:
                for li, cur_sl in enumerate(model.slices.seq_layers[recurrent_layer]):
                    Q[cur_sl, cur_sl] += np.triu(model.weights_intralayer_sequential[recurrent_layer][li], k=1)

    # between-layer recurrent
    for recurrent_layer in range(model.num_filter_kernels - 1):
        for seq_layer in range(len(model.slices.seq_layers[recurrent_layer])):
            cur_sl = model.slices.seq_layers[recurrent_layer][seq_layer]
            next_sl = model.slices.seq_layers[recurrent_layer + 1][seq_layer]
            W_rec = model.weights_seq_recurrent[recurrent_layer][seq_layer]
            Q[cur_sl, next_sl] += W_rec

    if model.num_filter_kernels > 2:
        for seq_layer in range(len(model.slices.seq_layers[0])):
            cur_sl = model.slices.seq_layers[0][seq_layer]
            next_sl = model.slices.seq_layers[-1][seq_layer]
            W_rec = model.weights_seq_recurrent[-1][seq_layer]
            Q[cur_sl, next_sl] += W_rec

    return Q


def add_seq_weights(model, Q):
    # Sequential
    if model.weights_seq_recurrent is not None:
        Q = add_seq_recurrent_weights(model, Q)
    else:
        first_seq_sl = model.slices.seq_layers[0][0]
        for fk in range(model.num_filter_kernels):
            pool_sl = model.slices.pool[fk]
            W = model.weights_sequential_layer[0][fk]
            Q[pool_sl, first_seq_sl] += W

        for i, cur_sl in enumerate(model.slices.seq_layers[0][1:]):
            prev_sl = model.slices.seq_layers[0][i]
            W = model.weights_sequential_layer[1][i]
            Q[prev_sl, cur_sl] += W

        # within-layer
        if not model.is_restricted:
            for li, cur_sl in enumerate(model.slices.seq_layers[0]):
                Q[cur_sl, cur_sl] += np.triu(model.weights_intralayer_sequential[0][li], k=1)

    return Q


def add_seq_biases(model, Q):
    if model.pooling_type == "probabilistic":
        raise NotImplementedError("Probabilistic QUBO with probabilistic pooling not implemented")
        num_units_before_seq = model.spec.conv_active + model.spec.n_pooled_units
    for l, seq_layer in enumerate(model.slices.seq_layers):
        for s, sl in enumerate(seq_layer):
            Q[sl, sl] += np.diag(model.biases_sequential_units[l][s])
    return Q


def add_weights_hidden_to_output(model: Conv_Deep_QBM, Q, ctx):
    for idx, last_sl in enumerate(model.slices.last_hidden):
        W_hy = model.weights_hidden_to_output[idx]
        Q[last_sl, model.slices.out] += W_hy
    return Q



def scale_qubo(Q: np.ndarray, model: Conv_Deep_QBM) -> np.ndarray:
    """scale weights of the sequential layers to fit max and min QUBO values"""
    max_val = np.max(Q)
    min_val = np.min(Q)
    abs_max = max(abs(max_val), abs(min_val))

    # for sl in model.slices.seq_layers[0]:
    #     Q[sl, sl] = Q[sl, sl] / abs_max  # scale to 80% of max abs value
    Q = Q / abs_max
    return Q



def build_unclamped_qubo(model: Conv_Deep_QBM, ctx, beta_eff: float, do_conv_label_bias=False, label_vec=None) -> np.ndarray:
    n = model.spec.n_hidden + model.spec.n_out
    Q = np.zeros((n, n), dtype=float)

    if model.pooling_type == "probabilistic":
        raise NotImplementedError("Unclamped QUBO with probabilistic pooling not implemented")
        Q = add_at_most_one_penalty_upper(model, Q, 0.8225)
        Q = add_link_penalty_upper(model, Q, ctx, 0.8225)

    # Conv
    Q = add_conv_biases(model, Q, ctx)
    # Sequential
    if len(model.sequential_layer_sizes) > 0:
        Q = add_seq_weights(model, Q)
        # Hidden biases sequential
        Q = add_seq_biases(model, Q)
    # Hidden -> Output
    Q = add_weights_hidden_to_output(model, Q, ctx)

    # output
    Q[model.slices.out, model.slices.out] += np.triu(model.weights_output_output, k=1)
    Q[model.slices.out, model.slices.out] += np.diag(model.biases_output)

    if do_conv_label_bias:
        for idx, conv_sl in enumerate(model.slices.conv):
            conv_label_bias = model.conv_label_bias[idx]
            Q[conv_sl, model.slices.out] += conv_label_bias

    #Q = scale_qubo(Q, model)

    return Q / float(beta_eff)


def build_clamped_qubo(model, ctx, label_vec: np.ndarray, beta_eff: float, do_conv_label_bias=False) -> np.ndarray:
    n = model.spec.n_hidden
    Q = np.zeros((n, n), dtype=float)

    if model.pooling_type == "probabilistic":
        NotImplementedError("Probabilistic QUBO with probabilistic pooling not implemented")
        Q = add_at_most_one_penalty_upper(model, Q, 0.8225)
        Q = add_link_penalty_upper(model, Q,  ctx, 0.8225)

    # Conv
    Q = add_conv_biases(model, Q, ctx)

    # Sequential
    if len(model.sequential_layer_sizes) > 0:
        Q = add_seq_weights(model, Q)
        # Hidden biases sequential
        Q = add_seq_biases(model, Q)

    # label bias
    for idx, last_sl in enumerate(model.slices.last_hidden):
        k = label_vec.reshape(-1, 1)
        eff = (model.weights_hidden_to_output[idx] @ label_vec.reshape(-1, 1)).reshape(-1)
        Q[last_sl, last_sl] += np.diag(eff)

    if do_conv_label_bias:
        for idx, conv_sl in enumerate(model.slices.conv):
            conv_label_bias = (model.conv_label_bias[idx] @ label_vec.reshape(-1, 1)).reshape(-1)
            Q[conv_sl, conv_sl] += np.diag(conv_label_bias)



    #Q = scale_qubo(Q, model)

    return Q / float(beta_eff)



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
