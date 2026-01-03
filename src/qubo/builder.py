import numpy as np

from src.model.cdqbm_state import Conv_Deep_QBM


def _conv_linear_terms(model, ctx) -> np.ndarray:
    """Return linear biases for the conv_sl block"""
    bases = []
    if model.pooling_type == "deterministic":
        for fk in range(model.num_filter_kernels):
            base = ctx.fmap_flat[fk][ctx.pooled_idx[fk]]

            if model.hidden_bias_type == "shared":
                base = base + model.biases_conv_units[fk][0]
            else: # case model.hidden_bias_type == "individual":
                base = base + model.biases_conv_units[ctx.pooled_idx[fk]]

            bases.append(base)

    else: # case probabilistic pooling
        for fk in range(model.num_filter_kernels):
            base = ctx.fmap_flat[fk]
            base += model.biases_conv_units[fk][0]
            bases.append(base)

    return np.array(bases)



def add_conv_biases(model, Q, ctx):
    if model.kernel_size > 0:
        conv_biases = _conv_linear_terms(model, ctx)
        for fk in range(model.num_filter_kernels):
            conv_bias = conv_biases[fk]
            conv_bias = np.diag(conv_bias)
            Q[model.slices.conv[fk], model.slices.conv[fk]] += conv_bias
    else:
        Q[model.slices.seq_layers[0][0], model.slices.seq_layers[0][0]] += np.diag(ctx.fmap_flat)

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
        if model.kernel_size > 0:
            first_seq_sl = model.slices.seq_layers[0][0]
            for fk in range(model.num_filter_kernels):
                pool_sl = model.slices.pool[fk]
                W = model.weights_sequential_layer[0][fk]
                plpl = Q[pool_sl, first_seq_sl]
                Q[pool_sl, first_seq_sl] += W

            for i, cur_sl in enumerate(model.slices.seq_layers[0][1:]):
                prev_sl = model.slices.seq_layers[0][i]
                W = model.weights_sequential_layer[1][i]
                Q[prev_sl, cur_sl] += W
        else:
            for i, cur_sl in enumerate(model.slices.seq_layers[0][:-1]):
                next_sl = model.slices.seq_layers[0][i+1]
                W = model.weights_sequential_layer[0][i]
                Q[cur_sl, next_sl] += W

        # within-layer
        if not model.is_restricted:
            for li, cur_sl in enumerate(model.slices.seq_layers[0]):
                Q[cur_sl, cur_sl] += np.triu(model.weights_intralayer_sequential[0][li], k=1)

    return Q


def add_seq_biases(model, Q):
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


def add_probabilistic_pooling_terms(model: Conv_Deep_QBM, Q: np.ndarray, ctx):
    Q = add_at_most_one_penalty_upper(model, Q, 0.8225)
    Q = add_link_penalty_upper(model, Q, ctx, 0.8225)
    return Q



def build_unclamped_qubo(model: Conv_Deep_QBM, ctx, beta_eff: float, do_conv_label_bias=False, label_vec=None) -> np.ndarray:
    n = model.spec.n_hidden + model.spec.n_out
    Q = np.zeros((n, n), dtype=float)

    # Conv
    Q = add_conv_biases(model, Q, ctx)

    if model.pooling_type == "probabilistic":
        Q = add_probabilistic_pooling_terms(model, Q, ctx)
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

        # if len(model.sequential_layer_sizes) > 1:
        #     for idx, seq_sl in enumerate(model.slices.seq_layers[0][:-1]):
        #         seq_label_bias = model.sequential_label_bias[idx]
        #         Q[seq_sl, model.slices.out] += seq_label_bias


    #Q = scale_qubo(Q, model)

    return Q / float(beta_eff)


def build_clamped_qubo(model, ctx, label_vec: np.ndarray, beta_eff: float, do_conv_label_bias=False) -> np.ndarray:
    n = model.spec.n_hidden
    Q = np.zeros((n, n), dtype=float)

    # Conv
    Q = add_conv_biases(model, Q, ctx)

    if model.pooling_type == "probabilistic":
        Q = add_probabilistic_pooling_terms(model, Q, ctx)

    # Sequential
    if len(model.sequential_layer_sizes) > 0:
        Q = add_seq_weights(model, Q)
        # Hidden biases sequential
        Q = add_seq_biases(model, Q)

    # label bias
    for idx, last_sl in enumerate(model.slices.last_hidden):
        eff = (model.weights_hidden_to_output[idx] @ label_vec.reshape(-1, 1)).reshape(-1)
        Q[last_sl, last_sl] += np.diag(eff)

    # conncet label

    if do_conv_label_bias:
        for idx, conv_sl in enumerate(model.slices.conv):
            conv_label_bias = (model.conv_label_bias[idx] @ label_vec.reshape(-1, 1)).reshape(-1)
            Q[conv_sl, conv_sl] += np.diag(conv_label_bias)

        # if len(model.sequential_layer_sizes) > 1:
        #     for idx, seq_sl in enumerate(model.slices.seq_layers[0][:-1]):
        #         seq_label_bias = (model.sequential_label_bias[idx] @ label_vec.reshape(-1, 1)).reshape(-1)
        #         Q[seq_sl, seq_sl] += np.diag(seq_label_bias)


    #Q = scale_qubo(Q, model)

    return Q / float(beta_eff)



def add_at_most_one_penalty_upper(model:Conv_Deep_QBM, qubo, penalty):
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
    """
    Enforce for each pooling group:
        y = OR(x_1, ..., x_n)
    where:
        conv_sl contains the x indices (many),
        pool_sl contains exactly one y index.
    Updates QUBO in "upper-triangular" form (i <= j).
    """

    for pool_sl, conv_sl in zip(model.slices.pool, model.slices.conv):

        # --- y index (pool_sl contains only one variable) ---
        if isinstance(pool_sl, slice):
            y_idx = pool_sl.start
        else:
            # could be int or 1-length list/array
            y_idx = int(pool_sl[0]) if hasattr(pool_sl, "__len__") else int(pool_sl)

        # --- x indices (conv_sl can be slice or list/array) ---
        if isinstance(conv_sl, slice):
            x_ids = np.arange(conv_sl.start, conv_sl.stop, dtype=np.int64)
        else:
            x_ids = np.asarray(conv_sl, dtype=np.int64)

        n = x_ids.size
        if n == 0:
            continue
        if n == 1:
            # OR with one input is just equality y = x
            x = x_ids[0]
            # B*(x + y - 2xy)
            qubo[x, x] += penalty_B
            qubo[y_idx, y_idx] += penalty_B
            lo, hi = (x, y_idx) if x < y_idx else (y_idx, x)
            if lo != hi:
                qubo[lo, hi] += -2.0 * penalty_B
            continue

        B = penalty_B

        # Diagonals:
        # +B * sum_i x_i
        qubo[x_ids, x_ids] += B
        # +B * (n-1) * y
        qubo[y_idx, y_idx] += B * (n - 1)

        # Cross terms x_i * y:  -2B * sum_i x_i y
        lo = np.minimum(x_ids, y_idx)
        hi = np.maximum(x_ids, y_idx)
        mask = lo != hi
        qubo[lo[mask], hi[mask]] += -2.0 * B

        # Pairwise x_i * x_j:  +2B * sum_{i<j} x_i x_j
        # Add +2B to all pairs among x_ids (upper triangle only)
        ii, jj = np.triu_indices(n, k=1)
        a = x_ids[ii]
        b = x_ids[jj]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        qubo[lo, hi] += 2.0 * B

    return qubo

