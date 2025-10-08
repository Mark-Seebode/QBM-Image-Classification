# src/train.py
from __future__ import annotations
import numpy as np
from src.train.pipeline import run_unclamped, run_clamped
from tqdm import tqdm

def nll_from_probs_binary(probs: np.ndarray, y: int, eps=1e-12) -> float:
    # probs = [p0, p1]
    p = probs[int(y)]
    return float(-np.log(max(p, eps)))

def train_one_iteration(
    model,
    X, Y,
    num_reads: int,
    beta_eff: float,
    lr: float,
    one_hot: bool = False,
    print_every: int = 0,
):
    errors_biases_conv = 0
    errors_biases_seq = 0
    errors_biases_out = 0
    errors_weights_kernels = 0
    if not model.is_restricted:
        errors_weights_interlayer_sequential = [0 for _ in model.weights_interlayer_sequential]
    errors_weights_sequential = [0 for _ in model.sequential_layer_sizes]
    errors_weights_hidden_to_output = 0
    errors_weights_output_output = 0

    n = len(X)
    tot_loss, tot_err = 0.0, 0.0

    for i, (x, y) in enumerate(zip(X, Y), 1):
        if one_hot:
            lab = np.zeros(model.num_lable_nodes, dtype=float)
            lab[int(y)] = 1.0
        else:
            lab = np.array([int(y)], dtype=float)

        out_c = run_clamped(model, x, lab, num_reads, beta_eff)
        out_u = run_unclamped(model, x, num_reads, beta_eff, one_hot)


        if not one_hot:
            loss = nll_from_probs_binary(out_u.probs, int(y))
        else:
            p = max(out_u.probs[int(y)], 1e-12)
            loss = float(-np.log(p))
        tot_loss += loss



        (
            avgs_biases_conv_units_c,
            avgs_biases_sequential_c,
            avgs_biases_output_c,
            avgs_kernel_weights_c,
            avgs_weights_interlayer_sequential_c,
            avgs_weights_sequential_layers_c,
            avgs_weights_hidden_to_output_c,
            avgs_weights_output_output_c
        ) = get_average_configuration_single(model, out_c, x, y=lab)

        (
            avgs_biases_conv_units_u,
            avgs_biases_sequential_u,
            avgs_biases_output_u,
            avgs_kernel_weights_u,
            avgs_weights_interlayer_sequential_u,
            avgs_weights_sequential_layers_u,
            avgs_weights_hidden_to_output_u,
            avgs_weights_output_output_u
        ) = get_average_configuration_single(model, out_u, x)

        errors_biases_conv += (avgs_biases_conv_units_c - avgs_biases_conv_units_u)

        errors_biases_seq += (avgs_biases_sequential_c - avgs_biases_sequential_u)

        errors_biases_out += (avgs_biases_output_c - avgs_biases_output_u)

        errors_weights_kernels += (avgs_kernel_weights_c - avgs_kernel_weights_u)

        # errors_weights_hidden_interlayer += (
        #           avgs_clamped_weights_hidden_interlayer - avgs_unclamped_weights_hidden_interlayer)
        if not model.is_restricted:
            errors_weights_interlayer_sequential = \
            [errors_weights_interlayer_sequential[i] +
                (avgs_weights_interlayer_sequential_c[i] - avgs_weights_interlayer_sequential_u[i])
                for i in range(len(model.weights_interlayer_sequential))]

        errors_weights_sequential = \
            [errors_weights_sequential[i] +
             (avgs_weights_sequential_layers_c[i] - avgs_weights_sequential_layers_u[i])
             for i in range(len(model.sequential_layer_sizes))]

        errors_weights_hidden_to_output += (avgs_weights_hidden_to_output_c - avgs_weights_hidden_to_output_u)

        errors_weights_output_output += (avgs_weights_output_output_c - avgs_weights_output_output_u)

    errors_biases_conv /= X.shape[0]
    errors_biases_seq /= X.shape[0]
    errors_biases_out /= X.shape[0]
    errors_weights_kernels /= X.shape[0]

    # errors_weights_hidden_interlayer /= x_batch.shape[0]
    if not model.is_restricted:
        errors_weights_interlayer_sequential = [error / X.shape[0] for error in errors_weights_interlayer_sequential]

    errors_weights_sequential = [error / X.shape[0] for error in errors_weights_sequential]
    errors_weights_hidden_to_output /= X.shape[0]
    errors_weights_output_output /= X.shape[0]

    model.biases_conv_units -= lr * errors_biases_conv
    model.biases_sequential_units -= lr * errors_biases_seq
    model.biases_output -= lr * errors_biases_out

    model.kernel_weights -= lr * errors_weights_kernels

    # self.weights_hidden_interlayer -= learning_rate * errors_weights_hidden_interlayer
    if not model.is_restricted:
        model.weights_interlayer_sequential = [weights - lr * errors_weights
                                              for weights, errors_weights in zip(model.weights_interlayer_sequential,
                                                                                 errors_weights_interlayer_sequential)]

    model.weights_sequential_layer = [weights - lr * errors_weights
                                     for weights, errors_weights in
                                     zip(model.weights_sequential_layer, errors_weights_sequential)]
    model.weights_hidden_to_output -= lr * errors_weights_hidden_to_output
    model.weights_output_output -= lr * errors_weights_output_output

    return tot_loss / max(1, n)


def get_average_configuration_single(model, samples, x_input: np.ndarray, y: np.ndarray = None):
    unclamped = y is None
    label = None if unclamped else np.array(y).flatten()

    n_hidden = sum(model.num_active_units_per_layer[1:])
    num_pooled_units = model.num_active_units_per_layer[1]
    sizes_active = model.num_active_units_per_layer[1:]
    starts = np.cumsum([0] + sizes_active[:-1])  # local starts per active layer

    avgs_biases_conv_units = np.zeros_like(model.biases_conv_units)
    avgs_biases_sequential = np.zeros_like(model.biases_sequential_units)
    avgs_biases_output = np.zeros_like(model.biases_output)
    avgs_kernel_weights = np.zeros_like(model.kernel_weights)

    avgs_weights_interlayer_sequential = [np.zeros_like(w) for w in model.weights_interlayer_sequential] if not model.is_restricted else 0
    avgs_weights_sequential_layers = [np.zeros_like(w) for w in model.weights_sequential_layer]
    avgs_weights_hidden_to_output = np.zeros_like(model.weights_hidden_to_output)
    avgs_weights_output_output = np.zeros_like(model.weights_output_output)

    sample_matrix = np.vstack([samples.samples])

    if model.pooling_type == "probabilistic":
        #remove the conv units from the sampel matrix
        sample_matrix = sample_matrix[:, model.num_conv_units:]
    n_reads = sample_matrix.shape[0]

    avg_biases = sample_matrix.mean(axis=0)

    if model.hidden_bias_type == "shared":
            # deprecated only worked with old pyramid structure
            # sum per layer into a shared bias slot
            # start = 0
            # for li in range(model.sequential_layer_sizes):
            #     cnt = model.num_hidden_units_per_layer[li]
            #     avgs_biases_conv_units[li] += np.sum(avg_biases[start:start + cnt])
            #     start += cnt
         avgs_biases_conv_units[0] += np.sum(avg_biases[:num_pooled_units]) #/ num_active_conv_units
    elif model.hidden_bias_type == "none":
        pass  # keep zeros
    else:
        # conv biases
        pooled_idx = np.asarray(model.pooled_units, dtype=int)
        pooled_marginals = avg_biases[:num_pooled_units]
        avgs_biases_conv_units[pooled_idx] += pooled_marginals.astype(avgs_biases_conv_units.dtype)

    avgs_biases_sequential += avg_biases[len(model.pool_windows):n_hidden]

    if unclamped:
        avgs_biases_output += avg_biases[n_hidden:]
    else:
        avgs_biases_output += label

    # Input units -> conv units
    for i, h in enumerate(samples.ctx.pooled_idx): # TODO: not working with probabilistic pooling
        rows, cols = model.input_groups[h]
        patch = x_input[np.ix_(rows, cols)]
        Eh = float(sample_matrix[:, i].mean())
        avgs_kernel_weights += patch * Eh

    #avgs_kernel_weights /= len(model.pooled_units)

    if not model.is_restricted:
        #within layer connections in the sequential layers:
        for li, W in enumerate(model.weights_interlayer_sequential):
            # indices of the li-th sequential layer block inside the hidden slice
            cur_size = sizes_active[li + 1]  # sequential layer size
            cur_s = int(starts[li + 1])  # start index of that layer in sample_matrix
            cur_e = cur_s + cur_size

            # samples over this layer: shape (num_reads, cur_size)
            cur_block = sample_matrix[:, cur_s:cur_e]

            # E[h_i h_j] as an average outer product (upper triangle only)
            avg_outer = (cur_block.T @ cur_block) / n_reads
            triu = np.triu_indices(cur_size, k=1)
            avgs_weights_interlayer_sequential[li][triu] = avg_outer[triu]

    # Sequential Layer
    for li, _ in enumerate(model.weights_sequential_layer):
        prev_size = sizes_active[li]
        cur_size = sizes_active[li + 1]
        prev_s = int(starts[li])
        cur_s = int(starts[li + 1])
        prev_block = sample_matrix[:, prev_s:prev_s + prev_size]  # (N, prev)
        cur_block = sample_matrix[:, cur_s:cur_s + cur_size]  # (N, cur)
        avgs_weights_sequential_layers[li][:] = (prev_block.T @ cur_block) / n_reads

    last_hidden = sum(model.num_active_units_per_layer[1:-1]) + np.arange(model.num_active_units_per_layer[-1])
    if unclamped:
        for o in range(model.num_lable_nodes):
            x_out = sample_matrix[:, n_hidden + o]  # E[y_o]
            y_last = sample_matrix[:, last_hidden]  # all last hidden
            avgs_weights_hidden_to_output[:, o] += (y_last.T @ x_out) / n_reads
    else:
        for o in range(model.num_lable_nodes):
            for i_h, h in enumerate(last_hidden):
                y_h = sample_matrix[:, h]
                avgs_weights_hidden_to_output[i_h, o] += np.average(y_h * label[o])

    if unclamped:
        yvars = sample_matrix[:, n_hidden:]
        avg_outer = np.einsum('ni,nj->ij', yvars, yvars) / n_reads
        triu = np.triu_indices(model.num_lable_nodes, k=1)
        avgs_weights_output_output[triu] += avg_outer[triu]
    else:
        outer = np.outer(label, label)
        triu = np.triu_indices(model.num_lable_nodes, k=1)
        avgs_weights_output_output[triu] += outer[triu]

    return (
            avgs_biases_conv_units,
            avgs_biases_sequential,
            avgs_biases_output,
            avgs_kernel_weights,
            avgs_weights_interlayer_sequential,
            avgs_weights_sequential_layers,
            avgs_weights_hidden_to_output,
            avgs_weights_output_output
        )


def train_model(model, train_x, train_y, batch_size, epochs, lr, sample_count, beta_eff, one_hot: bool = False):
    n = len(train_x)
    epoch_loss_list = []
    for epoch in tqdm(range(1, epochs + 1),
                      desc="Epochs",
                      ncols=100, leave=False):

        epoch_loss = 0.0

        with tqdm(range(0, n, batch_size),
                  desc=f"Epoch {epoch}/{epochs} batches",
                  ncols=100,
                  leave=False) as batch_bar:

            for idx, b in enumerate(batch_bar):
                xb = train_x[b:b + batch_size]
                yb = train_y[b:b + batch_size]

                loss = train_one_iteration(
                    model, xb, yb,
                    num_reads=sample_count,
                    beta_eff=beta_eff,
                    lr=lr,
                    one_hot=one_hot,
                    print_every=0
                )
                epoch_loss += loss
                avg_loss = epoch_loss / (idx + 1)
                epoch_loss_list.append(avg_loss)
                batch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        tqdm.write(f"Epoch {epoch}/{epochs} finished - avg loss: {avg_loss:.4f}")

    return epoch_loss_list

