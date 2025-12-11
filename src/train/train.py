# src/train.py
from __future__ import annotations
import numpy as np

from src import data_loader
from src.model.cdqbm_state import Conv_Deep_QBM
from src.model.geometry import conv2d_valid_stride
from src.model.layers import pooled_indices_for_input, SeqSpec, StackSpec, build_slices
from src.train.pipeline import run_unclamped, run_clamped
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, ConfusionMatrixDisplay
)

def nll_from_probs_binary(probs: np.ndarray, y: int, eps=1e-12) -> float:
    # probs = [p0, p1]
    p = probs[int(y)]
    return float(-np.log(max(p, eps)))

def train_one_iteration(
    model,
    X, Y,
    beta_eff: float,
    lr: float,
    kernel_change_history:list[int],
    sample_change_history:list[int],
    one_hot: bool = False,
    convLabel_bias: bool = False
):
    (
        errors_biases_conv,
        errors_biases_seq,
        errors_biases_out
    ) = initialize_zero_errors_biases(model)

    (
        errors_weights_kernels,
        errors_weights_interlayer_sequential,
        errors_weights_sequential,
        error_weights_seq_recurrent,
        errors_weights_hidden_to_output,
        errors_weights_output_output
    ) = initialize_zero_errors_weights(model)

    errors_conv_label_bias = np.zeros_like(model.conv_label_bias)

    n = len(X)
    tot_loss, tot_err = 0.0, 0.0
    prev_kernel_weights = model.kernel_weights.copy()
    for i, (x, y) in enumerate(zip(X, Y), 1):
        if one_hot:
            lab = np.zeros(model.num_label_nodes, dtype=float)
            lab[int(y)] = 1.0
        else:
            lab = np.array([int(y)], dtype=float)



        out_c = run_clamped(model, x, lab, beta_eff, convLabel_bias)
        if convLabel_bias:
            out_u = run_unclamped(model, x, beta_eff, one_hot, convLabel_bias, lab)
        else:
            out_u = run_unclamped(model, x, beta_eff, one_hot)

        # track change in samples
        samples_clamped = out_c.samples
        # remove last two entries in unclamped samples if output nodes are included
        samples_unclamped = out_u.samples
        if model.num_label_nodes > 0:
            samples_unclamped = samples_unclamped[:, :-model.num_label_nodes]


        sample_diff = np.mean(np.abs(samples_clamped - samples_unclamped))
        sample_change_history.append(sample_diff)


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
            avgs_weights_seq_recurrent_c,
            avgs_weights_hidden_to_output_c,
            avgs_weights_output_output_c,
            avgs_conv_label_bias_c
        ) = get_average_configuration_single(model, out_c, x, y=lab, convLabel_bias=convLabel_bias, conv_label=lab)

        (
            avgs_biases_conv_units_u,
            avgs_biases_sequential_u,
            avgs_biases_output_u,
            avgs_kernel_weights_u,
            avgs_weights_interlayer_sequential_u,
            avgs_weights_sequential_layers_u,
            avgs_weights_seq_recurrent_u,
            avgs_weights_hidden_to_output_u,
            avgs_weights_output_output_u,
            avgs_conv_label_bias_u
        ) = get_average_configuration_single(model, out_u, x, convLabel_bias=convLabel_bias, conv_label=lab)

        errors_biases_conv += (avgs_biases_conv_units_c - avgs_biases_conv_units_u)
        errors_weights_kernels += (avgs_kernel_weights_c - avgs_kernel_weights_u)
        if len(model.slices.seq_layers) > 0:
            for i, (ebs, ebsc, ebsu) in enumerate(zip(errors_biases_seq,
                                          avgs_biases_sequential_c,
                                          avgs_biases_sequential_u)):
                # element-wise update for nested list of arrays
                ebs = [ebs[j] + (ebsc[j] - ebsu[j]) for j in range(len(ebs))]
                errors_biases_seq[i] = ebs


            # errors_weights_hidden_interlayer += (
            #           avgs_clamped_weights_hidden_interlayer - avgs_unclamped_weights_hidden_interlayer)
            # remains zero if restricted
        if not model.is_restricted:
            for i, (ewis, ewisc, ewisu) in enumerate(zip(errors_weights_interlayer_sequential,
                                            avgs_weights_interlayer_sequential_c,
                                            avgs_weights_interlayer_sequential_u)):
                # element-wise update for nested list of arrays
                ewis = [ewis[j] + (ewisc[j] - ewisu[j]) for j in range(len(ewis))]
                errors_weights_interlayer_sequential[i] = ewis

            # errors_weights_interlayer_sequential[recurrent_layer] = \
            # [errors_weights_interlayer_sequential[recurrent_layer][i] +
            #     (avgs_weights_interlayer_sequential_c[recurrent_layer][i] - avgs_weights_interlayer_sequential_u[recurrent_layer][i])
            #     for i in range(len(model.weights_intralayer_sequential[recurrent_layer]))]
        for i, (ews, ewsc, ewsu) in enumerate(zip(errors_weights_sequential, avgs_weights_sequential_layers_c, avgs_weights_sequential_layers_u)):
            # element-wise update for nested list of arrays
            ews = [ews[j] + (ewsc[j] - ewsu[j]) for j in range(len(ews))]
            errors_weights_sequential[i] = ews


            #
        # errors_weights_sequential[recurrent_layer] = \
        #         [errors_weights_sequential[recurrent_layer][i] +
        #          (avgs_weights_sequential_layers_c[recurrent_layer][i] - avgs_weights_sequential_layers_u[recurrent_layer][i])
        #          for i in range(len(model.sequential_layer_sizes))]
        if model.is_recurrent_weights:
            for recurrent_layer in range(model.num_filter_kernels):
                for s in range(len(model.slices.seq_layers[recurrent_layer])):
                    error_weights_seq_recurrent[recurrent_layer][s] += (
                                avgs_weights_seq_recurrent_c[recurrent_layer][s]
                                - avgs_weights_seq_recurrent_u[recurrent_layer][s]
                    )

                errors_weights_hidden_to_output[recurrent_layer] += (avgs_weights_hidden_to_output_c[recurrent_layer] - avgs_weights_hidden_to_output_u[recurrent_layer])
        else:
            errors_weights_hidden_to_output[0] += (avgs_weights_hidden_to_output_c[0] - avgs_weights_hidden_to_output_u[0])

        if convLabel_bias:
            errors_conv_label_bias += (avgs_conv_label_bias_c - avgs_conv_label_bias_u)

        errors_biases_out += (avgs_biases_output_c - avgs_biases_output_u)
        errors_weights_output_output += (avgs_weights_output_output_c - avgs_weights_output_output_u)

    for recurrent_layer in range(model.num_filter_kernels):
        errors_biases_conv[recurrent_layer] /= X.shape[0]
        errors_weights_kernels[recurrent_layer] /= X.shape[0]

    if len(model.slices.seq_layers) > 0:
        for i in range(len(errors_biases_seq)):
            errors_biases_seq[i] = [err / X.shape[0] for err in errors_biases_seq[i]]

    # errors_weights_hidden_interlayer /= x_batch.shape[0]
    if not model.is_restricted:
        for i in range(len(errors_weights_interlayer_sequential)):
            errors_weights_interlayer_sequential[i] = [error / X.shape[0] for error in errors_weights_interlayer_sequential[i]]

    for i in range(len(errors_weights_sequential)):
        errors_weights_sequential[i] = [error / X.shape[0] for error in errors_weights_sequential[i]]

    if model.is_recurrent_weights:
        for recurrent_layer in range(model.num_filter_kernels):
            error_weights_seq_recurrent[recurrent_layer] = [error / X.shape[0] for error in error_weights_seq_recurrent[recurrent_layer]]

    errors_weights_hidden_to_output /= X.shape[0]

    model.biases_conv_units -= lr * errors_biases_conv
    if len(model.slices.seq_layers) > 0:
        for i, _ in enumerate(zip(model.biases_sequential_units,
                                          errors_biases_seq)):
            model.biases_sequential_units[i] = [
                b - lr * e
                for b, e in zip(model.biases_sequential_units[i],
                                        errors_biases_seq[i])
            ]

    model.kernel_weights -= lr * errors_weights_kernels

    if len(model.slices.seq_layers) > 0:
        # self.weights_hidden_interlayer -= learning_rate * errors_weights_hidden_interlayer
        if not model.is_restricted:
            for i, _ in enumerate(zip(model.weights_intralayer_sequential,
                                        errors_weights_interlayer_sequential)):
                model.weights_intralayer_sequential[i] = [weights - lr * errors_weights
                                                        for weights, errors_weights in zip(model.weights_intralayer_sequential[i],
                                                                                              errors_weights_interlayer_sequential[i])]

    for i, _ in enumerate(zip(model.weights_sequential_layer,
                                    errors_weights_sequential)):
        model.weights_sequential_layer[i] = [weights - lr * errors_weights
                                                for weights, errors_weights in
                                                zip(model.weights_sequential_layer[i], errors_weights_sequential[i])]
    if model.is_recurrent_weights:
        for recurrent_layer in range(model.num_filter_kernels):
            model.weights_seq_recurrent[recurrent_layer] = [weights - lr * errors_weights
                                            for weights, errors_weights in
                                            zip(model.weights_seq_recurrent[recurrent_layer], error_weights_seq_recurrent[recurrent_layer])]

    for i, _ in enumerate(zip(model.weights_hidden_to_output,errors_weights_hidden_to_output)):
        model.weights_hidden_to_output[i] -= lr * errors_weights_hidden_to_output[i]

    errors_weights_output_output /= X.shape[0]
    errors_biases_out /= X.shape[0]
    model.biases_output -= lr * errors_biases_out
    model.weights_output_output -= lr * errors_weights_output_output

    if convLabel_bias:
        model.conv_label_bias -= lr * errors_conv_label_bias / X.shape[0]

    delta = model.kernel_weights - prev_kernel_weights
    kernel_change = np.linalg.norm(delta)

    kernel_change_history.append(kernel_change)


    return tot_loss / max(1, n)


def initialize_zero_errors_biases(model: Conv_Deep_QBM):
    errors_biases_conv = np.zeros_like(model.biases_conv_units)
    errors_biases_seq = zero_structure_like(model.biases_sequential_units)
    errors_biases_out = np.zeros_like(model.biases_output)

    return (
        errors_biases_conv,
        errors_biases_seq,
        errors_biases_out
    )

def initialize_zero_errors_weights(model: Conv_Deep_QBM):
    errors_weights_kernels = np.zeros_like(model.kernel_weights)
    errors_weights_interlayer_sequential = (zero_structure_like(model.weights_intralayer_sequential)
        if not model.is_restricted else 0
    )
    errors_weights_sequential = zero_structure_like(model.weights_sequential_layer)
    error_weights_seq_recurrent = zero_structure_like(model.weights_seq_recurrent)
    errors_weights_hidden_to_output = np.zeros_like(model.weights_hidden_to_output)
    errors_weights_output_output = np.zeros_like(model.weights_output_output)

    return (
        errors_weights_kernels,
        errors_weights_interlayer_sequential,
        errors_weights_sequential,
        error_weights_seq_recurrent,
        errors_weights_hidden_to_output,
        errors_weights_output_output
    )


def calculate_avg_biases(sample_matrix, model, is_unclamped: bool, label):
    (avgs_biases_conv_units,
     avgs_biases_sequential,
     avgs_biases_output) = initialize_zero_errors_biases(model)
    avg_biases = sample_matrix.mean(axis=0)
    for fk in range(model.num_filter_kernels):
        if model.hidden_bias_type == "shared":
            conv_biases = avg_biases[model.slices.conv[fk]]
            avgs_biases_conv_units[fk] += np.sum(conv_biases)
        elif model.hidden_bias_type == "none":
            pass  # keep zeros
        else:  # case "individual"
            # conv biases
            raise NotImplementedError("Individual hidden biases buggy")
            pooled_idx = np.asarray(model.pooled_units, dtype=int)
            pooled_marginals = avg_biases[:num_pooled_units]
            avgs_biases_conv_units[pooled_idx] += pooled_marginals.astype(avgs_biases_conv_units.dtype)
    if len(model.slices.seq_layers) > 0:
        # sequential biases
        for l, seq_layer in enumerate(model.slices.seq_layers):
            for s, sl in enumerate(seq_layer):
                avg_biases_for_slice = avg_biases[sl]
                avgs_biases_sequential[l][s] += avg_biases_for_slice

    if is_unclamped:
        avgs_biases_output += avg_biases[model.slices.out]
    else:
        avgs_biases_output += label

    return (
        avgs_biases_conv_units,
        avgs_biases_sequential,
        avgs_biases_output
    )


def get_average_configuration_single(model: Conv_Deep_QBM, samples, x_input: np.ndarray, y: np.ndarray = None, convLabel_bias=False, conv_label:np.ndarray=None):
    unclamped = y is None
    label = None if unclamped else np.array(y).flatten() # TODO: check shape

    (
     avgs_kernel_weights,
     avgs_weights_interlayer_sequential,
     avgs_weights_sequential_layers,
     avgs_weights_seq_recurrent,
     avgs_weights_hidden_to_output,
     avgs_weights_output_output) = initialize_zero_errors_weights(model)
    avgs_conv_label_bias = np.zeros_like(model.conv_label_bias)

    #sample_matrix = np.vstack([samples.samples])
    sample_matrix = samples.samples

    if model.pooling_type == "probabilistic":
        #remove the conv units from the sampel matrix
        sample_matrix = sample_matrix[:, model.num_conv_units:]
    n_reads = sample_matrix.shape[0]

    (
        avgs_biases_conv_units,
        avgs_biases_sequential,
        avgs_biases_output
    ) = calculate_avg_biases(sample_matrix, model, unclamped, label)

    for fk in range(model.num_filter_kernels):
        # Input units -> conv units
        conv_sl = model.slices.pool[fk]  # same as conv for deterministic pooling
        for local_i, pool_id in enumerate(samples.ctx.pooled_idx[fk]):
            rows, cols = model.input_groups[pool_id]
            patch = x_input[np.ix_(rows, cols)]
            global_i = conv_sl.start + local_i
            Eh = float(sample_matrix[:, global_i].mean())
            avgs_kernel_weights[fk] += patch * Eh

        #avgs_kernel_weights[recurrent_layer] /= len(samples.ctx.pooled_idx[recurrent_layer])

    if len(model.sequential_layer_sizes) > 0:
        # TODO: weights pooled to first seq layer are stored here at index 0 -> have to start at 1
        #  at weights_sequential_layers. Use separate variable?

        # pooled units -> first sequential layer
        if model.is_recurrent_weights:
            for recurrent_layer in range(model.num_filter_kernels):
                pooled_block = sample_matrix[:, model.slices.pool[recurrent_layer]]
                first_seq_block = sample_matrix[:, model.slices.seq_layers[recurrent_layer][0]]
                avgs_weights_sequential_layers[recurrent_layer][0][:] = (pooled_block.T @ first_seq_block) / n_reads
        else:
            for pool_sl in model.slices.pool:
                pooled_block = sample_matrix[:, pool_sl]
                first_seq_block = sample_matrix[:, model.slices.seq_layers[0][0]]
                avgs_weights_sequential_layers[0][0][:] = (pooled_block.T @ first_seq_block) / n_reads


        # sequential layers -> sequential layers
        if model.is_recurrent_weights:
            for recurrent_layer in range(model.num_filter_kernels):
                for li in range(len(model.weights_sequential_layer[recurrent_layer]) - 1):
                    prev_slice = model.slices.seq_layers[recurrent_layer][li]
                    next_slice = model.slices.seq_layers[recurrent_layer][li + 1]
                    prev_block = sample_matrix[:, prev_slice]
                    next_block = sample_matrix[:, next_slice]
                    avgs_weights_sequential_layers[recurrent_layer][li + 1][:] = (prev_block.T @ next_block) / n_reads
        else:
            for li in range(len(model.weights_sequential_layer[1:]) - 1):
                prev_slice = model.slices.seq_layers[0][li]
                next_slice = model.slices.seq_layers[0][li + 1]
                prev_block = sample_matrix[:, prev_slice]
                next_block = sample_matrix[:, next_slice]
                avgs_weights_sequential_layers[1][li][:] = (prev_block.T @ next_block) / n_reads



        if not model.is_restricted:
            #within layer connections in the sequential layers:
            for r, seq_layer in enumerate(model.slices.seq_layers):
                for s, seq_slice in enumerate(seq_layer):
                    cur_block = sample_matrix[:, seq_slice]
                    avg_outer = (cur_block.T @ cur_block) / n_reads
                    size = seq_slice.stop - seq_slice.start
                    triu = np.triu_indices(size, k=1)
                    avgs_weights_interlayer_sequential[r][s][triu] = avg_outer[triu]

        if model.is_recurrent_weights:
            # recurrent connections between layers
            for recurrent_layer in range(model.num_filter_kernels - 1):
                for seq_layer in range(len(model.slices.seq_layers[recurrent_layer])):
                    cur_sl = model.slices.seq_layers[recurrent_layer][seq_layer]
                    next_sl = model.slices.seq_layers[recurrent_layer + 1][seq_layer]
                    cur_block = sample_matrix[:, cur_sl]
                    next_block = sample_matrix[:, next_sl]
                    avgs_weights_seq_recurrent[recurrent_layer][seq_layer] = (cur_block.T @ next_block) / n_reads
            # first and last recurrent layer
            if model.num_filter_kernels > 2:
                for seq_layer in range(len(model.slices.seq_layers[0])):
                    cur_sl = model.slices.seq_layers[0][seq_layer]
                    next_sl = model.slices.seq_layers[-1][seq_layer]
                    cur_block = sample_matrix[:, cur_sl]
                    next_block = sample_matrix[:, next_sl]
                    avgs_weights_seq_recurrent[-1][seq_layer][:] = (cur_block.T @ next_block) / n_reads

    # last hidden -> output
    last_hidden_slices = model.slices.last_hidden
    if unclamped:
        for l, lh_slice in enumerate(last_hidden_slices):
            x_out = sample_matrix[:, model.slices.out]  # E[y_o]
            y_last = sample_matrix[:, lh_slice]  # all last hidden
            avgs_weights_hidden_to_output[l] += (y_last.T @ x_out) / n_reads
    else:
        for o in range(model.num_label_nodes):
            for l, lh_slice in enumerate(last_hidden_slices):
                y_last = sample_matrix[:, lh_slice]  # (n_reads, n_hidden_last)
                avgs_weights_hidden_to_output[l][:, o] += label[o] * y_last.mean(axis=0)

    # output -> output
    if unclamped:
        yvars = sample_matrix[:, model.slices.out]
        avg_outer = np.einsum('ni,nj->ij', yvars, yvars) / n_reads
        triu = np.triu_indices(model.num_label_nodes, k=1)
        avgs_weights_output_output[triu] += avg_outer[triu]
    else:
        outer = np.outer(label, label)
        triu = np.triu_indices(model.num_label_nodes, k=1)
        avgs_weights_output_output[triu] += outer[triu]

    # conv_label_biases
    if convLabel_bias:
        if not unclamped:
            c_label = np.array(conv_label).flatten()
            for o in range(model.num_label_nodes):
                for l, conv_sl in enumerate(model.slices.conv):
                    y_last = sample_matrix[:, conv_sl]  # (n_reads, n_hidden_last)
                    avgs_conv_label_bias[l][:, o] += c_label[o] * y_last.mean(axis=0)
        else:
            for l, conv_sl in enumerate(model.slices.conv):
                x_out = sample_matrix[:, model.slices.out]  # E[y_o]
                y_last = sample_matrix[:, conv_sl]  # all last hidden
                avgs_conv_label_bias[l] += (y_last.T @ x_out) / n_reads



    return (
            avgs_biases_conv_units,
            avgs_biases_sequential,
            avgs_biases_output,
            avgs_kernel_weights,
            avgs_weights_interlayer_sequential,
            avgs_weights_sequential_layers,
            avgs_weights_seq_recurrent,
            avgs_weights_hidden_to_output,
            avgs_weights_output_output,
            avgs_conv_label_bias
        )


def train_model(model, train_x, train_y, batch_size, epochs, lr, sample_count, beta_eff, one_hot: bool = False, test_x=None, test_y=None):
    n = len(train_x)
    epoch_loss_list = []
    auc_list = []
    acc_list = []

    kernel_change_history = []
    sample_change_history = []
    conv_label = True
    for epoch in tqdm(range(1, epochs + 1),
                      desc="Epochs",
                      ncols=100, leave=False):

        epoch_loss = 0.0

        train_x, train_y = data_loader.shuffle_images(train_x, train_y, model.seed + epoch)

        with tqdm(range(0, n, batch_size),
                  desc=f"Epoch {epoch}/{epochs} batches",
                  ncols=100,
                  leave=False) as batch_bar:

            for idx, b in enumerate(batch_bar):
                if (b + batch_size) <= len(train_x):
                    x_batch = train_x[b:b + batch_size]  # [X_train[i] for i in range(b, b + batch_size)]
                    y_batch = train_y[b:b + batch_size]
                else:
                    x_batch = train_x[b:]  # [X_train[i] for i in range(b, len(X_train))]
                    y_batch = train_y[b:]

                if len(x_batch) == 0:
                    raise ValueError("Empty batch encountered during training")

                if epoch == 100:
                    conv_label = False

                try:
                    loss = train_one_iteration(
                            model, x_batch, y_batch,
                            beta_eff=beta_eff,
                            lr=lr,
                            kernel_change_history=kernel_change_history,
                            sample_change_history=sample_change_history,
                            one_hot=one_hot,
                            convLabel_bias=conv_label
                        )

                except Exception as e:
                    tqdm.write(f"Error during training at epoch {epoch}, batch {idx}: {e}")
                    model.save_weights(title=f"e{epoch}_b{idx}_error_backup")
                    raise e
                epoch_loss += loss
                avg_loss = epoch_loss / (idx + 1)
                epoch_loss_list.append(avg_loss)
                batch_bar.set_postfix(loss=f"{avg_loss:.4f}")
        #train_x, train_y = data_loader.shuffle_images(train_x, train_y, dataset_shuffle_seeds[epoch-1])
        tqdm.write(f"Epoch {epoch}/{epochs} finished - avg loss: {avg_loss:.4f}")

        predictions = []
        probs_all = []
        for i in tqdm(range(len(test_x)), desc="Predicting on test data", ncols=80, leave=False):
            run = run_unclamped(
                model, test_x[i],
                beta_eff=float(beta_eff),
                one_hot=bool(one_hot),
                do_conv_label_bias=conv_label
            )
            pred = int(np.argmax(run.probs))
            predictions.append(pred)
            probs_all.append(run.probs)

        acc = accuracy_score(test_y, predictions)

        if model.num_label_nodes == 1 or model.num_label_nodes == 2:
            pos_scores = np.array([p[1] for p in probs_all])
            auc = roc_auc_score(test_y, predictions)
        else:
            # macro-average AUC with one-vs-rest
            from sklearn.preprocessing import label_binarize
            Y_true = label_binarize(test_y, classes=list(range(2)))
            auc = roc_auc_score(Y_true, np.stack(probs_all, axis=0), average="macro", multi_class="ovr")
        auc_list.append(auc)
        acc_list.append(acc)


    import matplotlib.pyplot as plt
    # plot samples change history
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(sample_change_history) + 1), sample_change_history, marker='o')
    plt.title('Sample Change History')
    plt.xlabel('Iteration')
    plt.ylabel('Average Sample Change')
    plt.grid()
    plt.tight_layout()
    plt.show()


    # #plot auc_list and acc_list
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, epochs + 1), auc_list, marker='o')
    # plt.title('AUC over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('AUC')
    # plt.grid()
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, epochs + 1), acc_list, marker='o',
    #             color='orange')
    # plt.title('Accuracy over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    return epoch_loss_list, acc_list, auc_list, kernel_change_history


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

# Usage examples:
# errors_weights_sequential = zero_structure_like(model.weights_sequential_layer)
# errors_weights_interlayer_sequential = zero_structure_like(model.weights_intralayer_sequential)




