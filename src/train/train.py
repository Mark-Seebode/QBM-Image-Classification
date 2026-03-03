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
from typing import Any
import time
import pickle

class AvgConfig:
    biases_conv_units: list
    biases_sequential: list
    biases_output: list
    kernel_weights: list
    weights_intralayer_sequential: list
    weights_sequential_layers: list
    weights_seq_recurrent: list
    weights_hidden_to_output: list
    weights_output_output: list
    conv_label_bias: list
    seq_label_bias: list

    def __init__(self):
        self.biases_conv_units = []
        self.biases_sequential = []
        self.biases_output = []
        self.kernel_weights = []
        self.weights_intralayer_sequential = []
        self.weights_sequential_layers = []
        self.weights_seq_recurrent = []
        self.weights_hidden_to_output = []
        self.weights_output_output = []
        self.conv_label_bias = []
        self.seq_label_bias = []

class Weight_Erros():
    errors_biases_conv: Any
    errors_biases_seq: Any
    errors_biases_out: Any
    errors_weights_kernels: Any
    errors_weights_intralayer_sequential: Any
    errors_weights_sequential: Any
    error_weights_seq_recurrent: Any
    errors_weights_hidden_to_output: Any
    errors_weights_output_output: Any
    errors_seq_label_bias: Any
    errors_conv_label_bias: Any

    def __init__(self, model):
        (
            self.errors_biases_conv,
            self.errors_biases_seq,
            self.errors_biases_out
        ) = initialize_zero_errors_biases(model)

        (
            self.errors_weights_kernels,
            self.errors_weights_intralayer_sequential,
            self.errors_weights_sequential,
            self.error_weights_seq_recurrent,
            self.errors_weights_hidden_to_output,
            self.errors_weights_output_output
        ) = initialize_zero_errors_weights(model)

        self.errors_seq_label_bias = zero_structure_like(model.sequential_label_bias)
        self.errors_conv_label_bias = np.zeros_like(model.conv_label_bias)


def nll_from_probs_binary(probs: np.ndarray, y: int, eps=1e-12, ising_or_qubo="qubo") -> float:
    # probs = [p0, p1]
    if ising_or_qubo == "ising":
        index = 0 if y == -1 else 1
    else:
        index = y
    p = probs[index]
    return float(-np.log(max(p, eps)))


def getting_samples_batch(model, X, Y, beta_eff, one_hot, convLabel_bias):
    all_samples_c = []
    all_samples_u = []
    batch_loss = 0.0
    for i, (x, y) in enumerate(tqdm(zip(X, Y), total=len(X),
                                    desc=f"Getting Samples", ncols=100, leave=False)):

        out_c = run_clamped(model, x, y, beta_eff, convLabel_bias)
        if convLabel_bias:
            out_u = run_unclamped(model, x, beta_eff, one_hot, convLabel_bias, y)
        else:
            out_u = run_unclamped(model, x, beta_eff, one_hot)

        all_samples_c.append(out_c)
        all_samples_u.append(out_u)

        if not one_hot:
            loss = nll_from_probs_binary(out_u.probs, int(y), ising_or_qubo=model.sampler.ising_or_qubo)
        else:
            p = max(out_u.probs[int(y)], 1e-12)
            loss = float(-np.log(p))
        batch_loss += loss

    return np.array(all_samples_c), np.array(all_samples_u), batch_loss / len(X)


def track_sample_change(model, all_samples_c, all_samples_u, sample_change_history):
    # remove last two entries in unclamped samples if output nodes are included
    if model.num_label_nodes > 0:
        all_samples_u = all_samples_u[:, :-model.num_label_nodes]

    sample_diff = np.mean(np.abs(all_samples_c - all_samples_u))
    sample_change_history.append(sample_diff)

    return sample_change_history


def get_average_configuration_batch(model, all_samples_c, all_samples_u, X, Y, convLabel_bias):
    avgs_c = AvgConfig()
    avgs_u = AvgConfig()

    for i, (c, u, x, y) in enumerate(tqdm(zip(all_samples_c, all_samples_u, X, Y), total=len(X),
                                    desc=f"Calculating avg. configuration for batch", ncols=100, leave=False)):

        (
            avgs_biases_conv_units_c,
            avgs_biases_sequential_c,
            avgs_biases_output_c,
            avgs_kernel_weights_c,
            avgs_weights_intralayer_sequential_c,
            avgs_weights_sequential_layers_c,
            avgs_weights_seq_recurrent_c,
            avgs_weights_hidden_to_output_c,
            avgs_weights_output_output_c,
            avgs_conv_label_bias_c,
            avgs_seq_label_bias_c,
        ) = get_average_configuration_single(model, c, x, y=y, convLabel_bias=convLabel_bias, conv_label=y)

        avgs_c.biases_conv_units.append(avgs_biases_conv_units_c)
        avgs_c.biases_sequential.append(avgs_biases_sequential_c)
        avgs_c.biases_output.append(avgs_biases_output_c)
        avgs_c.kernel_weights.append(avgs_kernel_weights_c)
        avgs_c.weights_intralayer_sequential.append(avgs_weights_intralayer_sequential_c)
        avgs_c.weights_sequential_layers.append(avgs_weights_sequential_layers_c)
        avgs_c.weights_seq_recurrent.append(avgs_weights_seq_recurrent_c)
        avgs_c.weights_hidden_to_output.append(avgs_weights_hidden_to_output_c)
        avgs_c.weights_output_output.append(avgs_weights_output_output_c)
        avgs_c.conv_label_bias.append(avgs_conv_label_bias_c)
        avgs_c.seq_label_bias.append(avgs_seq_label_bias_c)

        (
            avgs_biases_conv_units_u,
            avgs_biases_sequential_u,
            avgs_biases_output_u,
            avgs_kernel_weights_u,
            avgs_weights_intralayer_sequential_u,
            avgs_weights_sequential_layers_u,
            avgs_weights_seq_recurrent_u,
            avgs_weights_hidden_to_output_u,
            avgs_weights_output_output_u,
            avgs_conv_label_bias_u,
            avgs_seq_label_bias_u,
        ) = get_average_configuration_single(model, u, x, convLabel_bias=convLabel_bias, conv_label=y)

        avgs_u.biases_conv_units.append(avgs_biases_conv_units_u)
        avgs_u.biases_sequential.append(avgs_biases_sequential_u)
        avgs_u.biases_output.append(avgs_biases_output_u)
        avgs_u.kernel_weights.append(avgs_kernel_weights_u)
        avgs_u.weights_intralayer_sequential.append(avgs_weights_intralayer_sequential_u)
        avgs_u.weights_sequential_layers.append(avgs_weights_sequential_layers_u)
        avgs_u.weights_seq_recurrent.append(avgs_weights_seq_recurrent_u)
        avgs_u.weights_hidden_to_output.append(avgs_weights_hidden_to_output_u)
        avgs_u.weights_output_output.append(avgs_weights_output_output_u)
        avgs_u.conv_label_bias.append(avgs_conv_label_bias_u)
        avgs_u.seq_label_bias.append(avgs_seq_label_bias_u)


    return avgs_c, avgs_u


def get_error(model, avgs_c: AvgConfig, avgs_u: AvgConfig, convLabel_bias=False):
    error = Weight_Erros(model)

    for item in range(len(avgs_c.biases_conv_units)):
        error.errors_biases_conv += (avgs_c.biases_conv_units[item] - avgs_u.biases_conv_units[item])
        error.errors_weights_kernels += (avgs_c.kernel_weights[item] - avgs_u.kernel_weights[item])
        if len(model.slices.seq_layers) > 0:
            for i, (ebs, ebsc, ebsu) in enumerate(zip(error.errors_biases_seq,
                                                      avgs_c.biases_sequential[item],
                                                      avgs_u.biases_sequential[item])):
                ebs = [ebs[j] + (ebsc[j] - ebsu[j]) for j in range(len(ebs))]
                error.errors_biases_seq[i] = ebs

        if not model.is_restricted:
            for i, (ewis, ewisc, ewisu) in enumerate(zip(error.errors_weights_intralayer_sequential,
                                                         avgs_c.weights_intralayer_sequential[item],
                                                         avgs_u.weights_intralayer_sequential[item],)):
                ewis = [ewis[j] + (ewisc[j] - ewisu[j]) for j in range(len(ewis))]
                error.errors_weights_intralayer_sequential[i] = ewis

        for i, (ews, ewsc, ewsu) in enumerate(
                zip(error.errors_weights_sequential, avgs_c.weights_sequential_layers[item], avgs_u.weights_sequential_layers[item])):
            ews = [ews[j] + (ewsc[j] - ewsu[j]) for j in range(len(ews))]
            error.errors_weights_sequential[i] = ews

        if model.is_recurrent_weights:
            for recurrent_layer in range(model.num_filter_kernels):
                for s in range(len(model.slices.seq_layers[recurrent_layer])):
                    error.error_weights_seq_recurrent[recurrent_layer][s] += (
                            avgs_c.weights_seq_recurrent[item][recurrent_layer][s]
                            - avgs_u.weights_seq_recurrent[item][recurrent_layer][s]
                    )

                error.errors_weights_hidden_to_output[recurrent_layer] += (
                            avgs_c.weights_hidden_to_output[item][recurrent_layer] - avgs_u.weights_hidden_to_output[item][
                        recurrent_layer])
        else:
            error.errors_weights_hidden_to_output[0] += (avgs_c.weights_hidden_to_output[item][0] - avgs_u.weights_hidden_to_output[item][0])

        if convLabel_bias and model.kernel_size > 0:
            error.errors_conv_label_bias += (avgs_c.conv_label_bias[item] - avgs_u.conv_label_bias[item] )
            for i in range(len(error.errors_seq_label_bias)):
                error.errors_seq_label_bias[i] += (avgs_c.seq_label_bias[item][i] - avgs_u.seq_label_bias[item][i])

        error.errors_biases_out += (avgs_c.biases_output[item] - avgs_u.biases_output[item])
        error.errors_weights_output_output += (avgs_c.weights_output_output[item] - avgs_u.weights_output_output[item])

    return error


def normalize_error(model, error: Weight_Erros, norm_factor, convLabel_bias):
    if model.kernel_size > 0:
        for recurrent_layer in range(model.num_filter_kernels):
            error.errors_biases_conv[recurrent_layer] /= norm_factor
            error.errors_weights_kernels[recurrent_layer] /= norm_factor

    if len(model.slices.seq_layers) > 0:
        for i in range(len(error.errors_biases_seq)):
            error.errors_biases_seq[i] = [err / norm_factor for err in error.errors_biases_seq[i]]

    # errors_weights_hidden_interlayer /= x_batch.shape[0]
    if not model.is_restricted:
        for i in range(len(error.errors_weights_intralayer_sequential)):
            error.errors_weights_intralayer_sequential[i] = [error / norm_factor for error in error.errors_weights_intralayer_sequential[i]]

    for i in range(len(error.errors_weights_sequential)):
        error.errors_weights_sequential[i] = [error / norm_factor for error in error.errors_weights_sequential[i]]

    if model.is_recurrent_weights:
        for recurrent_layer in range(model.num_filter_kernels):
            error.error_weights_seq_recurrent[recurrent_layer] = [error / norm_factor for error in error.error_weights_seq_recurrent[recurrent_layer]]

    error.errors_weights_hidden_to_output /= norm_factor

    error.errors_weights_output_output /= norm_factor
    error.errors_biases_out /= norm_factor

    if convLabel_bias and model.kernel_size > 0:
        error.errors_conv_label_bias /= norm_factor
        error.errors_seq_label_bias = [error.errors_seq_label_bias[i] / norm_factor for i in range(len(error.errors_seq_label_bias))]

    return error


def apply_gradient_step(model, errors, lr, convLabel_bias):

    model.kernel_weights = model.kernel_weights - errors.errors_weights_kernels * lr
    model.biases_conv_units = model.biases_conv_units - errors.errors_biases_conv * lr

    if len(model.slices.seq_layers) > 0:
        for i, _ in enumerate(zip(model.biases_sequential_units,
                                          errors.errors_biases_seq)):
            model.biases_sequential_units[i] = [
                b - lr * e
                for b, e in zip(model.biases_sequential_units[i],
                                        errors.errors_biases_seq[i])
            ]

    if len(model.slices.seq_layers) > 0:
        if not model.is_restricted:
            for i, _ in enumerate(zip(model.weights_intralayer_sequential,
                                        errors.errors_weights_intralayer_sequential)):
                model.weights_intralayer_sequential[i] = [weights - lr * errors_weights
                                                        for weights, errors_weights in zip(model.weights_intralayer_sequential[i],
                                                                                              errors.errors_weights_intralayer_sequential[i])]

    for i, _ in enumerate(zip(model.weights_sequential_layer,
                                    errors.errors_weights_sequential)):
        model.weights_sequential_layer[i] = [weights - lr * errors_weights
                                                for weights, errors_weights in
                                                zip(model.weights_sequential_layer[i], errors.errors_weights_sequential[i])]
    if model.is_recurrent_weights:
        for recurrent_layer in range(model.num_filter_kernels):
            model.weights_seq_recurrent[recurrent_layer] = [weights - lr * errors_weights
                                            for weights, errors_weights in
                                            zip(model.weights_seq_recurrent[recurrent_layer], errors.error_weights_seq_recurrent[recurrent_layer])]

    for i in range(len(model.weights_hidden_to_output)):
        model.weights_hidden_to_output[i] = model.weights_hidden_to_output[i] - errors.errors_weights_hidden_to_output[i] *lr

    model.weights_output_output = model.weights_output_output - errors.errors_weights_output_output * lr
    model.biases_output = model.biases_output - errors.errors_biases_out * lr

    if convLabel_bias and model.kernel_size > 0:
        model.conv_label_bias = model.conv_label_bias - lr * errors.errors_conv_label_bias
        model.sequential_label_bias = [model.sequential_label_bias[i] - lr * errors.errors_seq_label_bias[i] for i in range(len(model.sequential_label_bias))]


def train_one_iteration(
    model: Conv_Deep_QBM,
    X, Y,
    beta_eff: float,
    lr: float,
    conv_learning_rate: float,
    kernel_change_history:list[int],
    sample_change_history:list[int],
    one_hot: bool = False,
    convLabel_bias: bool = False,
):


    n = len(X)
    prev_kernel_weights = model.kernel_weights.copy()

    # prepare label batch for the whole Y at once
    if one_hot:
        lab_batch = np.zeros((len(Y), model.num_label_nodes), dtype=float)
        for idx, yy in enumerate(Y):
            lab_batch[idx, int(yy)] = 1.0
    else:
        lab_batch = np.array([[int(yy)] for yy in Y], dtype=float)

    all_samples_c, all_samples_u, batch_loss =  getting_samples_batch(model, X, lab_batch, beta_eff, one_hot, convLabel_bias)
    #sample_change_history = track_sample_change(model, all_samples_c, all_samples_u, sample_change_history)
    avgs_c, avgs_u = get_average_configuration_batch(model, all_samples_c, all_samples_u, X, Y, convLabel_bias)

    errors = get_error(model, avgs_c, avgs_u, convLabel_bias)

    errors = normalize_error(model, errors, X.shape[0], convLabel_bias)

    apply_gradient_step(model, errors, lr, convLabel_bias)

    delta = model.kernel_weights - prev_kernel_weights
    kernel_change = np.linalg.norm(delta)

    kernel_change_history.append(kernel_change)

    if model.centerize:
        update_biases_with_centers(model, all_samples_c, one_hot, Y)

        update_centers(model, all_samples_c, Y, one_hot)

    return batch_loss


def orthonormalize_all_weights(model:Conv_Deep_QBM):
    for fk in range(model.num_filter_kernels):
        reorthogonalize_qr(model.kernel_weights[fk])

    if len(model.slices.seq_layers) > 0:
        for recurrent_layer in range(model.num_filter_kernels):
            for i in range(len(model.weights_sequential_layer[recurrent_layer])):
                reorthogonalize_qr(model.weights_sequential_layer[recurrent_layer][i])

    if model.is_recurrent_weights:
        for recurrent_layer in range(model.num_filter_kernels):
            for i in range(len(model.weights_seq_recurrent[recurrent_layer])):
                reorthogonalize_qr(model.weights_seq_recurrent[recurrent_layer][i])

    # hidden to output weights
    for i in range(len(model.weights_hidden_to_output)):
        reorthogonalize_qr(model.weights_hidden_to_output[i])

def update_biases_with_centers(model:Conv_Deep_QBM, all_samples_c, uses_one_hot: bool, label_batch):
    """
       a= a + ν·W (yd −β)
       b=b + ν·W(xd−α)+ν·V (zd−γ)
       c=c+ ν·V(yd −β)
       """

    avg_samples_clamped = np.mean(np.vstack(all_samples_c), axis=0)
    # one hot encode labels if model uses one hot encoding
    if uses_one_hot:
        label_batch_one_hot = []
        for lbl in label_batch:
            lab = np.zeros(model.num_label_nodes, dtype=float)
            lab[lbl] = 1.0
            label_batch_one_hot.append(lab)
        label_batch = np.array(label_batch_one_hot)
    mean_out = np.mean(label_batch, axis=0)

    v = 0.9
    #update conv_sl biases
    for fk in range(model.num_filter_kernels):
        if len(model.sequential_layer_sizes) > 0:
            first_seq_sl = model.slices.seq_layers[0][0]
            W = model.weights_sequential_layer[0][fk]
            avg_first_sq_sl = avg_samples_clamped[first_seq_sl]
            model.biases_conv_units[fk] += np.sum(v * W @ (avg_first_sq_sl - model.center_seq[0][0]))
        else:
            for idx, last_sl in enumerate(model.slices.last_hidden):
                W = model.weights_hidden_to_output[idx]
                model.biases_conv_units[fk] += np.sum(v * W @ (mean_out - model.center_out))

    # update sequential biases
    for idx, seq_layer in enumerate(model.slices.seq_layers[0]):
        if idx == 0:
            for fk in range(model.num_filter_kernels):
                for c, conv_sl in enumerate(model.slices.conv):
                    W_prev = model.weights_sequential_layer[0][fk]
                    avg_conv_sl = avg_samples_clamped[conv_sl]
                    model.biases_sequential_units[0][idx] += v * W_prev.T @ (avg_conv_sl - model.center_conv[fk])
            if len(model.slices.seq_layers[0]) > 1: # not only one layer
                next_slice = model.slices.seq_layers[0][idx + 1]
                W_next = model.weights_sequential_layer[1][idx]
                avg_next_sl = avg_samples_clamped[next_slice]
                model.biases_sequential_units[0][idx] += v * W_next @ (avg_next_sl - model.center_seq[0][idx + 1])
            else:
                for i, last_sl in enumerate(model.slices.last_hidden):
                    W_next = model.weights_hidden_to_output[i]
                    model.biases_sequential_units[0][idx] += v * W_next @ (mean_out - model.center_out)
        else:
            if idx >= len(model.slices.seq_layers[0]) - 1:
                prev_slice = model.slices.seq_layers[0][idx - 1]
                W_prev = model.weights_sequential_layer[1][idx - 1]
                avg_prev_sl = avg_samples_clamped[prev_slice]
                model.biases_sequential_units[0][idx] += v * W_prev.T @ (avg_prev_sl - model.center_seq[0][idx - 1])

                # if idx < len(model.slices.seq_layers[0]) - 2: # not last layer
                #     next_slice = model.slices.seq_layers[0][idx + 1]
                #     W_next = model.weights_sequential_layer[1][idx + 1]
                #     avg_next_sl = avg_samples_clamped[next_slice]
                #     model.biases_sequential_units[0][idx] += v * W_next @ (avg_next_sl - model.center_seq[0][idx + 1])
                # else:
                for i, last_sl in enumerate(model.slices.last_hidden):
                    W_next = model.weights_hidden_to_output[i]
                    model.biases_sequential_units[0][idx] += v * W_next @ (mean_out - model.center_out)



    #update out biases

    if len(model.slices.seq_layers[0]) >= 1:
        for i, last_sl in enumerate(model.slices.last_hidden):
            W = model.weights_hidden_to_output[i]
            avg_last_sl = avg_samples_clamped[last_sl]
            model.biases_output += v * W.T @ (avg_last_sl - model.center_seq[0][-1])
    else:
      pass



def reorthogonalize_qr(W):
    """
    Enforces orthogonality on W in-place.
    If W is (m, n):
      - m >= n → columns orthonormal
      - m < n  → rows orthonormal
    """
    m, n = W.shape

    if m >= n:
        Q, R = np.linalg.qr(W)
        # fix sign ambiguity (important!)
        D = np.sign(np.diag(R))
        Q *= D
        W[:] = Q
    else:
        Q, R = np.linalg.qr(W.T)
        D = np.sign(np.diag(R))
        Q *= D
        W[:] = Q.T






def update_centers(model:Conv_Deep_QBM, all_samples_c, label_batch, uses_one_hot: bool):
    # average samples clamped
    v = 0.9
    #β = (1 − ν) · β + ν · yd
    avg_samples_clamped = np.mean(np.vstack(all_samples_c), axis=0)

    # update hidden beta
    for fk in range(model.num_filter_kernels):
        mean_h = avg_samples_clamped[model.slices.conv[fk]]
        model.center_conv[fk] = (1.0 - v) *model.center_conv[fk] + v * mean_h

    for sl_idx, seq_layer in enumerate(model.slices.seq_layers):
        for s, sl in enumerate(seq_layer):
            mean_h = avg_samples_clamped[sl]
            model.center_seq[sl_idx][s] = (1.0 - v) * model.center_seq[sl_idx][s] + v * mean_h

    # one hot encode labels if model uses one hot encoding
    if uses_one_hot:
        label_batch_one_hot = []
        for lbl in label_batch:
            lab = np.zeros(model.num_label_nodes, dtype=float)
            lab[lbl] = 1.0
            label_batch_one_hot.append(lab)
        label_batch = np.array(label_batch_one_hot)
    mean_out = np.mean(label_batch, axis=0)
    model.center_out = (1.0 - v) * model.center_out + v * mean_out

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

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

    if model.kernel_size > 0:
        for fk in range(model.num_filter_kernels):
            if model.hidden_bias_type == "shared":
                conv_biases = avg_biases[model.slices.conv[fk]]
                alpha_conv = model.center_conv[fk]
                centered_bias = conv_biases #- alpha_conv
                avgs_biases_conv_units[fk] += np.sum(centered_bias)
            else: # case "individual"
                conv_biases = avg_biases[model.slices.conv[fk]]
                alpha_conv = model.center_conv[fk]  # same shape
                centered_bias = conv_biases #- alpha_conv
                avgs_biases_conv_units[fk] += centered_bias

    if len(model.slices.seq_layers) > 0:
        # sequential biases
        for l, seq_layer in enumerate(model.slices.seq_layers):
            for s, sl in enumerate(seq_layer):
                avg_biases_for_slice = avg_biases[sl]
                alpha_seq = model.center_seq[l][s]  # same shape
                centered_bias = avg_biases_for_slice #- alpha_seq
                avgs_biases_sequential[l][s] += centered_bias

    if is_unclamped:
        out_biases = avg_biases[model.slices.out]
        centered_out = out_biases #- model.center_out
        avgs_biases_output += centered_out
    else:
        centered_label = label #- model.center_out
        avgs_biases_output += centered_label

    return (
        avgs_biases_conv_units,
        avgs_biases_sequential,
        avgs_biases_output
    )


def get_average_configuration_single(model: Conv_Deep_QBM, samples, x_input: np.ndarray, y: np.ndarray = None, convLabel_bias=False, conv_label:np.ndarray=None):
    unclamped = y is None
    label = None if unclamped else np.array(y).flatten()
    # turn label 0 into -1 if binary classification without one-hot encoding
    # if not unclamped and model.num_label_nodes == 1:
    #     label = np.where(label == 0, -1, label)

    (
     avgs_kernel_weights,
     avgs_weights_intralayer_sequential,
     avgs_weights_sequential_layers,
     avgs_weights_seq_recurrent,
     avgs_weights_hidden_to_output,
     avgs_weights_output_output) = initialize_zero_errors_weights(model)
    avgs_conv_label_bias = np.zeros_like(model.conv_label_bias)
    avgs_seq_label_bias = zero_structure_like(model.sequential_label_bias)

    #sample_matrix = np.vstack([samples.samples])
    sample_matrix = samples.samples
    n_reads = sample_matrix.shape[0]

    (
        avgs_biases_conv_units,
        avgs_biases_sequential,
        avgs_biases_output
    ) = calculate_avg_biases(sample_matrix, model, unclamped, label)

    if model.kernel_size > 0:
        for fk in range(model.num_filter_kernels):
            if model.pooling_type == "deterministic":
                # Input units -> conv_sl units
                conv_sl = model.slices.pool[fk]  # same as conv_sl for deterministic pooling
                for local_i, pool_id in enumerate(samples.ctx.pooled_idx[fk]):
                    rows, cols = model.input_groups[pool_id]
                    patch = x_input[np.ix_(rows, cols)]
                    global_i = conv_sl.start + local_i
                    Eh = float(sample_matrix[:, global_i].mean())
                    Eh = Eh if not model.centerize else Eh - model.center_conv[fk][local_i]
                    avgs_kernel_weights[fk] += patch * Eh
                #avgs_kernel_weights[fk] /= len(samples.ctx.pooled_idx[fk])
            else: # case probabilistic pooling
                # Input units -> conv_sl units
                conv_sl = model.slices.conv[fk]
                for local_i in range(conv_sl.stop - conv_sl.start):
                    global_i = conv_sl.start + local_i
                    rows, cols = model.input_groups[local_i]
                    patch = x_input[np.ix_(rows, cols)]
                    global_i = conv_sl.start + local_i
                    Eh = float(sample_matrix[:, global_i].mean())
                    Eh = Eh if not model.centerize else Eh - model.center_conv[fk][local_i]
                    avgs_kernel_weights[fk] += patch * Eh

    else:
        first_seq_block = sample_matrix[:, model.slices.seq_layers[0][0]]
        flat_input = x_input.flatten().reshape(-1, 1)

        avgs_kernel_weights += flat_input @ first_seq_block.mean(axis=0, keepdims=True)



    if len(model.sequential_layer_sizes) > 0:
        # TODO: weights pooled to first seq layer are stored here at index 0 -> have to start at 1
        #  at weights_sequential_layers. Use separate variable?

        # pooled units -> first sequential layer
        if model.kernel_size > 0:
            if model.is_recurrent_weights:
                for recurrent_layer in range(model.num_filter_kernels):
                    pooled_block = sample_matrix[:, model.slices.pool[recurrent_layer]]
                    first_seq_block = sample_matrix[:, model.slices.seq_layers[recurrent_layer][0]]
                    avgs_weights_sequential_layers[recurrent_layer][0][:] = (pooled_block.T @ first_seq_block) / n_reads
            else:
                for i, pool_sl in enumerate(model.slices.pool):
                    pooled_block = sample_matrix[:, pool_sl]
                    first_seq_block = sample_matrix[:, model.slices.seq_layers[0][0]]

                    pooled = pooled_block if not model.centerize else pooled_block - model.center_conv[i][0]

                    first_seq = first_seq_block if not model.centerize else first_seq_block - model.center_seq[0][0]

                    avgs_weights_sequential_layers[0][i] = (pooled.T @ first_seq) / n_reads

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

                prev = prev_block if not model.centerize else prev_block - model.center_seq[0][li]
                next = next_block if not model.centerize else next_block - model.center_seq[0][li + 1]
                avgs_weights_sequential_layers[1][li][:] = (prev.T @ next) / n_reads



        if not model.is_restricted:
            #within layer connections in the sequential layers:
            for r, seq_layer in enumerate(model.slices.seq_layers):
                for s, seq_slice in enumerate(seq_layer):
                    cur_block = sample_matrix[:, seq_slice]
                    block = cur_block if not model.centerize else cur_block - model.center_seq[r]
                    avg_outer = (block.T @ block) / n_reads
                    size = seq_slice.stop - seq_slice.start
                    triu = np.triu_indices(size, k=1)
                    avgs_weights_intralayer_sequential[r][s][triu] = avg_outer[triu]

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

            if model.centerize:
                alpha_last = model.center_seq[0][-1] if len(model.sequential_layer_sizes) > 0 \
                             else float(model.center_conv[l][0])
                alpha_out = model.center_out
                y_last = y_last - alpha_last
                x_out = x_out - alpha_out
            else:
                y_last = y_last
                x_out = x_out

            avgs_weights_hidden_to_output[l] += (y_last.T @ x_out) / n_reads
    else:
        for o in range(model.num_label_nodes):
            for l, lh_slice in enumerate(last_hidden_slices):
                y_last = sample_matrix[:, lh_slice]  # (n_reads, n_hidden_last)
                avgs_weights_hidden_to_output[l][:, o] += label[o] * y_last.mean(axis=0)

    # output -> output
    if unclamped:
        yvars = sample_matrix[:, model.slices.out]
        yvars = yvars if not model.centerize else yvars -model.center_out
        avg_outer = np.einsum('ni,nj->ij', yvars, yvars) / n_reads
        triu = np.triu_indices(model.num_label_nodes, k=1)
        avgs_weights_output_output[triu] += avg_outer[triu]
    else:
        label = label if not model.centerize else label - model.center_out
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
            if len(model.sequential_layer_sizes) > 1:
                for o in range(model.num_label_nodes):
                    for l, seq_sl in enumerate(model.slices.seq_layers[0][:-1]):
                        y_last = sample_matrix[:, seq_sl]  # (n_reads, n_hidden_last)
                        avgs_seq_label_bias[l][:, o] += c_label[o] * y_last.mean(axis=0)
        else:
            for l, conv_sl in enumerate(model.slices.conv):
                x_out = sample_matrix[:, model.slices.out]  # E[y_o]
                y_last = sample_matrix[:, conv_sl]  # all last hidden
                avgs_conv_label_bias[l] += (y_last.T @ x_out) / n_reads

            if len(model.sequential_layer_sizes) > 1:
                for l, seq_sl in enumerate(model.slices.seq_layers[0][:-1]):
                    x_out = sample_matrix[:, model.slices.out]  # E[y_o]
                    y_last = sample_matrix[:, seq_sl]  # all last hidden
                    avgs_seq_label_bias[l] += (y_last.T @ x_out) / n_reads

    return (
            avgs_biases_conv_units,
            avgs_biases_sequential,
            avgs_biases_output,
            avgs_kernel_weights,
            avgs_weights_intralayer_sequential,
            avgs_weights_sequential_layers,
            avgs_weights_seq_recurrent,
            avgs_weights_hidden_to_output,
            avgs_weights_output_output,
            avgs_conv_label_bias,
            avgs_seq_label_bias
        )


def train_model(model:Conv_Deep_QBM, train_x, train_y, batch_size, epochs, lr, sample_count, beta_eff, conv_learning_rate=None, one_hot: bool = False, test_x=None, test_y=None, restart_from_epoch=1,restart_from_batch_n=1, save_path="out/", loaded_acc_list=None, loaded_auc_list=None):
    n = len(train_x)
    epoch_loss_list = []

    auc_list = loaded_auc_list if loaded_auc_list is not None else []
    acc_list = loaded_acc_list if loaded_acc_list is not None else []

    kernel_change_history = []
    sample_change_history = []
    conv_label = True

    num_batches_total = (len(train_x) + batch_size - 1) // batch_size
    if restart_from_batch_n < 1:
        raise ValueError("restart_from_batch_n must be >= 1")
    if restart_from_batch_n > num_batches_total:
        raise ValueError(
            f"restart_from_batch_n ({restart_from_batch_n}) exceeds total batches "
            f"({num_batches_total}) for batch_size={batch_size}"
        )

    restart_active = restart_from_batch_n > 1

    if conv_learning_rate is None:
        conv_learning_rate = lr
    for epoch in tqdm(range(1, epochs + 1),
                      desc="Epochs",
                      ncols=100, leave=False):
        if restart_from_epoch > 1:
            true_epoch = epoch + restart_from_epoch
        else:
            true_epoch = epoch
        epoch_loss = 0.0

        with tqdm(range(0, n, batch_size),
                  desc=f"Epoch {epoch}/{epochs} batches",
                  ncols=100,
                  leave=False) as batch_bar:

            for batchnum, b in enumerate(batch_bar):
                if restart_active and epoch == 1 and batchnum < restart_from_batch_n:
                    batchnum += 1
                    continue
                if (b + batch_size) <= len(train_x):
                    x_batch = train_x[b:b + batch_size]  # [X_train[i] for i in range(b, b + batch_size)]
                    y_batch = train_y[b:b + batch_size]
                else:
                    x_batch = train_x[b:]  # [X_train[i] for i in range(b, len(X_train))]
                    y_batch = train_y[b:]

                if len(x_batch) == 0:
                    raise ValueError("Empty batch encountered during training")



                try:
                    loss = train_one_iteration(
                            model, x_batch, y_batch,
                            beta_eff=beta_eff,
                            lr=lr,
                            conv_learning_rate=conv_learning_rate,
                            kernel_change_history=kernel_change_history,
                            sample_change_history=sample_change_history,
                            one_hot=one_hot,
                            convLabel_bias=conv_label,
                        )
                except Exception as e:
                    tqdm.write(f"Error during training at epoch {true_epoch}, batch {batchnum}: {e}")
                    model.save_weights(title=f"e{true_epoch}_b{batchnum}_s{model.seed}_error_backup")
                    # save acc and auc lists up to this point as pickle files
                    with open(f"{save_path}/acc_list_backup{model.seed}.pkl", "wb") as f:
                        pickle.dump(acc_list, f)
                    with open(f"{save_path}/auc_list_backup{model.seed}.pkl", "wb") as f:
                        pickle.dump(auc_list, f)
                    raise e
                epoch_loss += loss
                avg_loss = epoch_loss / (batchnum + 1)
                epoch_loss_list.append(avg_loss)
                batch_bar.set_postfix(loss=f"{avg_loss:.4f}")
                batchnum += 1
        #train_x, train_y = data_loader.shuffle_images(train_x, train_y, dataset_shuffle_seeds[epoch-1])
        tqdm.write(f"Epoch {true_epoch}/{epochs} finished - avg loss: {avg_loss:.4f}")
        model.save_weights(title=f"e{true_epoch}_seed{model.seed}", path=save_path)

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
            auc = roc_auc_score(test_y, pos_scores)
        else:
            # macro-average AUC with one-vs-rest
            from sklearn.preprocessing import label_binarize
            Y_true = label_binarize(test_y, classes=list(range(model.num_label_nodes)))
            auc = roc_auc_score(Y_true, np.stack(probs_all, axis=0), average="macro", multi_class="ovr")
        auc_list.append(auc)
        acc_list.append(acc)

        #model.sampler.refresh_connection()


        #print("\nacc:", acc )
        #print("auc:", auc)

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


def compute_fisher_preconditioner(grad: np.ndarray, eps: float = 1e-6):
    fisher_diag = np.mean(grad ** 2, axis=0) if grad.ndim > 1 else grad ** 2
    return 1.0 / np.sqrt(fisher_diag + eps)


def apply_natural_gradient(param: np.ndarray, grad: np.ndarray, lr: float, eps: float = 1e-6):

    scale = compute_fisher_preconditioner(grad, eps)
    return param - lr * grad #* scale





