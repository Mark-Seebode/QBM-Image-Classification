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

    #Q = add_seq_biases_with_virtual_carry(model, Q, lam=1.0, agg_kernels="mean", use_abs=False)
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
        #Q = add_residual_skip_connections(model, Q, lam=1.0, mode="tied", normalize=False, use_abs=False)
        # Hidden biases sequential
        Q = add_seq_biases(model, Q)
    # Hidden -> Output
    #Q = add_skip_penultimate_to_output(model, Q, lam=1.0, normalize=False, use_abs=False)
    #Q = add_residual_skips_from_output(model, Q, lam=1.0, mode="tied", normalize=False, use_abs=False)
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
        #Q = add_residual_skip_connections(model, Q, lam=1.0, mode="tied", normalize=False, use_abs=False)
        # Hidden biases sequential
        Q = add_seq_biases(model, Q)

    #Q = add_skip_penultimate_to_output(model, Q, lam=1.0, normalize=False, use_abs=False)
    #Q = add_residual_skips_from_output(model, Q, lam=1.0, mode="tied", normalize=False, use_abs=False)

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


import numpy as np

def zero_structure_like(obj):
    """Recursively mirror `obj` structure, replacing ndarrays with zeros of the same shape,
    sequences with sequences of zeros, and scalars with 0 of the same type."""
    if isinstance(obj, np.ndarray):
        return np.zeros_like(obj)
    if isinstance(obj, list):
        return [zero_structure_like(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(zero_structure_like(x) for x in obj)
    # fallback for scalars / other objects
    try:
        return type(obj)(0)
    except Exception:
        return 0

def compute_seq_biases_with_virtual_carry(
    model,
    lam: float = 1.0,
    agg_kernels: str = "sum",
    use_abs: bool = False,
    include_original: bool = True,
):
    """
    Virtual-bias carry for kernel_size>0, non-recurrent, len(seq_sizes)>1.

    Rule:
      bias(seq_{s+1}) += lam * sum_incoming( W_in→seq_s )

    where:
      - for s=0 (seq1), W_in is pool→seq1 (per filter kernel)
      - for s>=1, W_in is seq_s→seq_{s+1}

    Parameters
    ----------
    lam : float
        Strength of the added virtual bias.
    agg_kernels : {"sum", "mean", "max"}
        How to aggregate across filter kernels for pool→seq1 weights.
    use_abs : bool
        If True, use absolute weights before summing (helps if signs cancel undesirably).
    include_original : bool
        If True, start from existing biases and add carry; if False, return only carry term.

    Returns
    -------
    b_eff : same nested structure as model.biases_sequential_units
        Effective sequential biases with the carry added.
    """
    assert model.kernel_size > 0, "This function is for kernel_size>0."
    assert model.weights_seq_recurrent is None, "This function assumes non-recurrent."
    assert len(model.sequential_layer_sizes) > 1, "Needs at least 2 sequential layers."

    # Copy structure
    b_eff = zero_structure_like(model.biases_sequential_units)

    # We only operate on the first (and only) seq stack in your non-recurrent setup
    l = 0
    n_seq_layers = len(model.biases_sequential_units[l])

    for s in range(n_seq_layers):
        if include_original:
            b_eff[l][s] = np.array(model.biases_sequential_units[l][s], copy=True)
        else:
            b_eff[l][s] = np.zeros_like(model.biases_sequential_units[l][s])

    # ---- Helper: sum incoming dimension -> vector over target units
    def incoming_sum(W):
        # W shape: (n_in, n_out)
        if use_abs:
            W = np.abs(W)
        return W.sum(axis=0)  # -> (n_out,)

    # ---- 1) pool -> seq1 weights, aggregate across kernels, add to bias of seq2
    # weights_sequential_layer[0] exists only if len(seq_sizes)>0 (true here)
    # shape: (num_kernels, n_pool, n_seq1)
    W_pool_seq1 = model.weights_sequential_layer[0]
    if isinstance(W_pool_seq1, list):
        W_pool_seq1 = np.array(W_pool_seq1)

    if W_pool_seq1 is None or len(W_pool_seq1) == 0:
        raise ValueError("Expected pool->seq1 weights in weights_sequential_layer[0].")

    # compute per-kernel incoming sums for seq1 units: shape (num_kernels, n_seq1)
    per_kernel = np.stack([incoming_sum(W_pool_seq1[fk]) for fk in range(W_pool_seq1.shape[0])], axis=0)

    if agg_kernels == "sum":
        carry_seq1 = per_kernel.sum(axis=0)
    elif agg_kernels == "mean":
        carry_seq1 = per_kernel.mean(axis=0)
    elif agg_kernels == "max":
        carry_seq1 = per_kernel.max(axis=0)
    else:
        raise ValueError("agg_kernels must be one of: 'sum', 'mean', 'max'.")

    # add to bias of seq2 (index 1), but only if seq2 exists
    # NOTE: carry_seq1 has size n_seq1; but you're asking to add it to bias of seq2.
    # That requires a mapping from seq1-units to seq2-units. The simplest consistent
    # interpretation of your request is: use the *next* weight matrix to map it forward:
    #   carry_to_seq2 = (W_seq1_seq2.T @ 1_vector_weighted_by_seq1_incoming)
    #
    # However you asked explicitly: "weights connecting conv units with first seq layer
    # should be summed up and added to the bias of the second seq layer."
    #
    # That means you want a vector of length n_seq2. So we *project* the seq1-carry
    # through W_seq1_seq2 to get a n_seq2 vector:
    W_seq1_seq2 = model.weights_sequential_layer[1][0]  # shape (n_seq1, n_seq2)
    if use_abs:
        W_seq1_seq2_used = np.abs(W_seq1_seq2)
    else:
        W_seq1_seq2_used = W_seq1_seq2

    carry_to_seq2 = W_seq1_seq2_used.T @ carry_seq1  # -> (n_seq2,)

    b_eff[l][1] += lam * carry_to_seq2

    # ---- 2) For each seq_s -> seq_{s+1}, add sum_incoming(W_seq_s->seq_{s+1}) to bias of seq_{s+2}
    # i indexes the interlayer weights: i=0 is seq1->seq2, i=1 is seq2->seq3, ...
    # For bias of seq_{i+2}, we use summed incoming of W_{i}: (n_seq_{i+1},) then project through next W
    # to match dimension of seq_{i+2}.
    for i in range(0, n_seq_layers - 2):
        W_cur = model.weights_sequential_layer[1][i]      # seq_{i+1} -> seq_{i+2}? careful:
        # In your add_seq_weights:
        #   for i, cur_sl in enumerate(seq_layers[0][1:]):
        #       prev_sl = seq_layers[0][i]
        #       W = weights_sequential_layer[1][i]
        # so weights_sequential_layer[1][i] is seq_i -> seq_{i+1} (0-based)
        # Therefore:
        #   i=0: seq1 -> seq2
        #   i=1: seq2 -> seq3
        # and we want to add sum(W_cur) (a vector over target layer = seq_{i+1})
        # into the bias of seq_{i+2}, so we must project through W_next.

        W_i = model.weights_sequential_layer[1][i]        # seq_{i} -> seq_{i+1}
        vec_target = incoming_sum(W_i)                    # length = n_seq_{i+1}

        W_next = model.weights_sequential_layer[1][i + 1] # seq_{i+1} -> seq_{i+2}
        W_next_used = np.abs(W_next) if use_abs else W_next

        carry_to_nextnext = W_next_used.T @ vec_target    # length = n_seq_{i+2}

        b_eff[l][i + 2] += lam * carry_to_nextnext

    return b_eff


def add_seq_biases_with_virtual_carry(
    model,
    Q,
    lam: float = 1.0,
    agg_kernels: str = "sum",
    use_abs: bool = False,
):
    """
    Writes sequential biases into Q, after applying virtual carry.
    """
    b_eff = compute_seq_biases_with_virtual_carry(
        model,
        lam=lam,
        agg_kernels=agg_kernels,
        use_abs=use_abs,
        include_original=True,
    )

    for l, seq_layer in enumerate(model.slices.seq_layers):
        for s, sl in enumerate(seq_layer):
            Q[sl, sl] += np.diag(b_eff[l][s])

    return Q


import numpy as np

def add_residual_skip_connections(
    model,
    Q: np.ndarray,
    lam: float = 0.1,
    mode: str = "tied",          # "tied" | "identity"
    normalize: bool = True,
    use_abs: bool = False,
):
    """
    Add residual (skip) couplings directly into the QUBO energy.

    Supports: kernel_size>0, non-recurrent (weights_seq_recurrent is None),
    and at least 2 sequential layers.

    Skips added:
      - pool -> seq2  (skip seq1)
      - seq_i -> seq_{i+2} for i=0..L-3

    Parameters
    ----------
    lam : float
        Global skip strength multiplier.
    mode : str
        "identity": add lam * I (diagonal-to-diagonal) skips when dims match.
                    If dims don't match, we use a rectangular identity on min(dim).
        "tied":     skip weights derived from existing weights, no new trainables:
                    pool->seq2 uses W(pool->seq1) @ W(seq1->seq2)
                    seq_i->seq_{i+2} uses W(i->i+1) @ W(i+1->i+2)
    normalize : bool
        If True, scale skip weights by sqrt(width_of_middle_layer) to keep magnitudes sane.
    use_abs : bool
        If True, use abs() of weights when forming products (reduces sign cancellation).

    Returns
    -------
    Q : np.ndarray
        Updated QUBO (upper-triangular fill style consistent with your code).
    """
    if model.weights_seq_recurrent is not None:
        # Not handled here (recurrent case needs different wiring)
        return Q
    if model.kernel_size <= 0:
        # You can extend similarly for FC-only case if you want
        return Q
    if len(model.slices.seq_layers) == 0 or len(model.slices.seq_layers[0]) < 2:
        return Q

    seq_slices = model.slices.seq_layers[0]
    L = len(seq_slices)

    def _maybe_abs(W):
        return np.abs(W) if use_abs else W

    def _rect_identity(n_in: int, n_out: int):
        """Rectangular identity-like matrix."""
        m = min(n_in, n_out)
        M = np.zeros((n_in, n_out), dtype=float)
        M[np.arange(m), np.arange(m)] = 1.0
        return M

    def _norm_factor(width_mid: int):
        return np.sqrt(width_mid) if (normalize and width_mid > 0) else 1.0

    # --------- Skip 1: pool -> seq2 (skip seq1) ----------
    # Only if we have at least 2 seq layers: seq1 (idx 0), seq2 (idx 1)
    pool_to_seq2_blocks = []
    if L >= 2:
        seq1_sl = seq_slices[0]
        seq2_sl = seq_slices[1]

        if mode == "tied":
            # W_pool->seq1 is per filter kernel: weights_sequential_layer[0][fk]
            # W_seq1->seq2 is weights_sequential_layer[1][0]
            W_seq1_seq2 = _maybe_abs(model.weights_sequential_layer[1][0])  # (n_seq1, n_seq2)

            for fk in range(model.num_filter_kernels):
                pool_sl = model.slices.pool[fk]
                W_pool_seq1 = _maybe_abs(model.weights_sequential_layer[0][fk])  # (n_pool, n_seq1)

                # product gives (n_pool, n_seq2)
                W_skip = (W_pool_seq1 @ W_seq1_seq2) / _norm_factor(W_seq1_seq2.shape[0])
                pool_to_seq2_blocks.append((pool_sl, W_skip))

            # add each kernel's block into Q
            for pool_sl, W_skip in pool_to_seq2_blocks:
                Q[pool_sl, seq2_sl] += lam * W_skip

        elif mode == "identity":
            # identity-like skip: pool dim -> seq2 dim (rectangular identity)
            for fk in range(model.num_filter_kernels):
                pool_sl = model.slices.pool[fk]
                n_pool = pool_sl.stop - pool_sl.start
                n_seq2 = seq2_sl.stop - seq2_sl.start
                W_skip = _rect_identity(n_pool, n_seq2)
                Q[pool_sl, seq2_sl] += lam * W_skip

        else:
            raise ValueError("mode must be 'tied' or 'identity'.")

    # --------- Skip 2: seq_i -> seq_{i+2} for i=0..L-3 ----------
    if L >= 3:
        if mode == "tied":
            # weights_sequential_layer[1][i] is seq_i -> seq_{i+1}
            for i in range(L - 2):
                src_sl = seq_slices[i]
                mid_sl = seq_slices[i + 1]
                dst_sl = seq_slices[i + 2]

                W_i   = _maybe_abs(model.weights_sequential_layer[1][i])     # (n_i, n_{i+1})
                W_ip1 = _maybe_abs(model.weights_sequential_layer[1][i + 1]) # (n_{i+1}, n_{i+2})

                # product gives (n_i, n_{i+2})
                W_skip = (W_i @ W_ip1) / _norm_factor(W_i.shape[1])
                Q[src_sl, dst_sl] += lam * W_skip

        elif mode == "identity":
            for i in range(L - 2):
                src_sl = seq_slices[i]
                dst_sl = seq_slices[i + 2]
                n_src = src_sl.stop - src_sl.start
                n_dst = dst_sl.stop - dst_sl.start
                W_skip = _rect_identity(n_src, n_dst)
                Q[src_sl, dst_sl] += lam * W_skip

        else:
            raise ValueError("mode must be 'tied' or 'identity'.")

    return Q


import numpy as np
from src.model.cdqbm_state import Conv_Deep_QBM

def add_skip_penultimate_to_output(
    model: Conv_Deep_QBM,
    Q: np.ndarray,
    lam: float = 0.1,
    normalize: bool = True,
    use_abs: bool = False,
):
    """
    Add a skip connection from the penultimate seq layer to output, derived from:
      W_skip = W_(L-2 -> L-1) @ W_(L-1 -> out)

    Works for kernel_size>0 or 0, as long as there are >=2 sequential layers.
    Non-recurrent case (weights_seq_recurrent is None).

    Adds to Q[seq_{L-2}, out].
    """
    # only for non-recurrent in this helper
    if model.weights_seq_recurrent is not None:
        return Q

    # Need at least two sequential layers
    if len(model.slices.seq_layers) == 0:
        return Q
    seq = model.slices.seq_layers[0]
    if len(seq) < 2:
        return Q

    # Identify slices
    penult_sl = seq[-2]
    last_sl   = seq[-1]

    # Existing weights:
    # seq_{L-2} -> seq_{L-1} is weights_sequential_layer[1][L-2]
    # because weights_sequential_layer[1][i] maps seq_i -> seq_{i+1} (0-based)
    i = len(seq) - 2  # index for edge from penultimate to last
    W_penult_last = model.weights_sequential_layer[1][i]  # (n_penult, n_last)

    # last -> out weight: in your code it's weights_hidden_to_output[idx] with idx over last_hidden slices.
    # In non-recurrent + seq case, last_hidden is [last_sl], so idx=0.
    W_last_out = model.weights_hidden_to_output[0]        # (n_last, n_out)

    if use_abs:
        W_penult_last = np.abs(W_penult_last)
        W_last_out    = np.abs(W_last_out)

    # Compose
    W_skip = W_penult_last @ W_last_out  # (n_penult, n_out)

    # Optional normalization to control magnitude growth
    if normalize:
        W_skip = W_skip / np.sqrt(W_penult_last.shape[1] + 1e-12)

    # Add to Q
    Q[penult_sl, model.slices.out] += lam * W_skip
    return Q

import numpy as np

def add_residual_skips_from_output(
    model,
    Q: np.ndarray,
    lam: float = 0.1,
    mode: str = "tied",          # "tied" | "identity"
    normalize: bool = True,
    use_abs: bool = False,
    max_hops: int = 2,           # how far back to skip: 2 means connect L-2 and L-3 to out
):
    """
    Add residual skip connections *from the output side* into the QUBO energy.

    Non-recurrent (weights_seq_recurrent is None) and requires sequential layers.

    Adds skip couplings:
      - seq_{L-2} -> out  (skips last layer)
      - seq_{L-3} -> out  (skips last two layers)   if max_hops>=3
      - ...
      - seq_{L-k} -> out  for k = 2..max_hops  (if available)

    TIED mode:
      W_skip(seq_{L-k} -> out) = W_{L-k -> L-k+1} @ ... @ W_{L-1 -> out}

    IDENTITY mode:
      If n_src == n_out: W_skip = I
      else: rectangular identity-like matrix on min(n_src, n_out)

    Notes:
      - This modifies the energy directly (adds off-diagonal blocks to Q).
      - For clamped QUBO you must fold these into diagonal biases using label_vec,
        analogous to what you already do for last_hidden.

    Parameters
    ----------
    max_hops:
        2 means only connect seq_{L-2} -> out.
        3 means connect seq_{L-2} and seq_{L-3} -> out, etc.

    Returns
    -------
    Q : np.ndarray
    """
    if model.weights_seq_recurrent is not None:
        return Q

    if len(model.slices.seq_layers) == 0:
        return Q
    seq_slices = model.slices.seq_layers[0]
    L = len(seq_slices)
    if L < 2:
        return Q

    def _maybe_abs(W):
        return np.abs(W) if use_abs else W

    def _rect_identity(n_in: int, n_out: int):
        m = min(n_in, n_out)
        M = np.zeros((n_in, n_out), dtype=float)
        M[np.arange(m), np.arange(m)] = 1.0
        return M

    def _norm_factor(width_mid: int):
        return np.sqrt(width_mid) if (normalize and width_mid > 0) else 1.0

    out_sl = model.slices.out
    n_out = out_sl.stop - out_sl.start

    # last -> out weights (non-recurrent seq case: idx=0)
    W_last_out = _maybe_abs(model.weights_hidden_to_output[0])  # (n_last, n_out)

    # How many layers back can we connect?
    # k=2 corresponds to source layer = L-2 (penultimate)
    max_k = min(max_hops, L)  # can't go back beyond seq1
    for k in range(2, max_k + 1):
        src_idx = L - k
        src_sl = seq_slices[src_idx]
        n_src = src_sl.stop - src_sl.start

        if mode == "identity":
            if n_src == n_out:
                W_skip = np.eye(n_src, dtype=float)
            else:
                W_skip = _rect_identity(n_src, n_out)
            Q[src_sl, out_sl] += lam * W_skip
            continue

        if mode != "tied":
            raise ValueError("mode must be 'tied' or 'identity'.")

        # Build product from seq_{src_idx} up to seq_{L-1}, then to out.
        # weights_sequential_layer[1][i] maps seq_i -> seq_{i+1}
        # so multiply W_src @ W_{src+1} @ ... @ W_{L-2} @ W_last_out
        W_prod = None
        for i in range(src_idx, L - 1):
            W_i = _maybe_abs(model.weights_sequential_layer[1][i])  # (n_i, n_{i+1})
            if W_prod is None:
                W_prod = W_i
            else:
                W_prod = W_prod @ W_i

            # optional normalization each hop (keeps magnitudes stable)
            if normalize:
                W_prod = W_prod / _norm_factor(W_i.shape[1])

        # now to out
        W_skip = W_prod @ W_last_out  # (n_src, n_out)

        # optional normalization for final projection width
        if normalize:
            W_skip = W_skip / _norm_factor(W_last_out.shape[0])

        Q[src_sl, out_sl] += lam * W_skip

    return Q





