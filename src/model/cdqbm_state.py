import random
from pathlib import Path
import pickle
import numpy as np

from src.model.layers import pooled_indices_for_input, SeqSpec, StackSpec, build_slices, BlockSlices
from src.model.model_ab import MODEL
from src.model.geometry import (
    conv_output_shape, get_input_groups_coords, build_pool_windows,
    num_conv_units_from_dim, count_pooled_units, conv2d_valid_stride
)
from src.qubo.sampler import LocalSASampler, DWaveAdapter
import pickle

class Conv_Deep_QBM(MODEL):
    def __init__(self, num_visible_nodes, num_lable_nodes, example_image, image_shape=(28,28), seed=77, kernel_size=3, pooling_size=0,
                 pooling_type="deterministic", stride=1, sequential_layer_sizes=None, num_filter_kernels=0, is_recurrent_weights=False,
                 param_string="", load_path="", speicherort=None, is_restricted=False, parallelize=False,
                 hidden_bias_type="none", solver="SA", num_reads=100, anneal=1000, api_token="", groupQpuToken_name=""):

        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.stride = stride
        self.image_shape = image_shape

        if sequential_layer_sizes is None:
            sequential_layer_sizes = []
        self.sequential_layer_sizes = sequential_layer_sizes
        self.is_recurrent_weights = is_recurrent_weights

        (num_hidden_nodes,
         self.num_active_units,
         self.num_hidden_units_per_layer,
         self.num_active_units_per_layer,
         self.input_groups,
         self.conv_layer_dim,
         self.num_conv_units) = self.build_model_structure()

        self.num_filter_kernels = num_filter_kernels

        super().__init__(seed, num_hidden_nodes, num_visible_nodes, num_lable_nodes, is_restricted)

        self.hidden_bias_type = hidden_bias_type

        self.weight_objects = [self.kernel_weights,
                            self.weights_sequential_layer,
                            self.weights_hidden_to_output,
                            self.weights_output_output,
                            self.weights_intralayer_sequential,
                            self.weights_seq_recurrent,
                            self.biases_conv_units,
                            self.biases_sequential_units,
                            self.biases_output] = self.init_params(is_recurrent_weights)

        self.param_string = param_string

        self.load_path = load_path
        self.speicherort = speicherort

        self.spec: StackSpec = self.build_layer_indexing(example_image, is_recurrent_weights)
        self.slices: BlockSlices = build_slices(self.spec)

        self.sampler = self.init_sampler(solver, num_reads, anneal, parallelize, seed, api_token, groupQpuToken_name)


    def init_sampler(self, solver="SA", num_reads=100, anneal=1000, parallelize=False, seed=77, api_token="", groupQpuToken_name=""):
        # -------------------
        # Sampler
        # -------------------
        if solver.upper() == "SA":
            sampler = LocalSASampler(num_reads=num_reads, num_sweeps=anneal, parallelize=parallelize, seed=seed)
        else:
            sampler = DWaveAdapter(
                solver=solver,
                api_token=api_token,
                groupQpuToken_name=groupQpuToken_name,
                num_reads=num_reads,
                embedding=None, # TODO: do embedding here?
                seed=seed)
            print(f"Using D-Wave solver: {solver}")

        return sampler

    def load_params(self, file_path):
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, "rb") as file:
                loaded_params = pickle.load(file)
        else:
            raise FileNotFoundError("params file not found")

        (self.kernel_weights,
         self.weights_hidden_interlayer,
         self.weights_hidden_to_output, self.weights_output_output, self.biases_conv_units,
         self.biases_output) = loaded_params

        self.weight_objects = [self.kernel_weights,
                               self.weights_hidden_interlayer,
                               self.weights_hidden_to_output, self.weights_output_output, self.biases_conv_units,
                               self.biases_output]


    def build_model_structure(self):
        num_hidden_nodes = 0

        # conv geometry
        conv_dim = conv_output_shape(self.image_shape, self.kernel_size, self.stride)
        input_groups = get_input_groups_coords(self.image_shape, self.kernel_size, self.stride)
        num_conv_units = num_conv_units_from_dim(conv_dim)

        # pooling windows (static tiling)
        self.pool_windows = build_pool_windows(conv_dim, self.pooling_size)

        # per-layer counts
        hidden_units_per_layer = [num_conv_units]
        active_units_per_layer = [num_conv_units]
        active_units_per_layer.append(count_pooled_units(self.pooling_type, self.pool_windows, num_conv_units))

        num_hidden_nodes += num_conv_units

        # sequential layers
        for s in self.sequential_layer_sizes:
            hidden_units_per_layer.append(s)
            active_units_per_layer.append(s)
            num_hidden_nodes += s

        num_active_units = sum(active_units_per_layer)
        return (
            num_hidden_nodes,
            num_active_units,
            hidden_units_per_layer,
            active_units_per_layer,
            input_groups,
            conv_dim,
            num_conv_units,
        )


    def init_weights_hidden_to_output(self, last_hidden_layer_dim: int, num_output_units: int):
        weights = np.random.uniform(-1, 1, (last_hidden_layer_dim, num_output_units))
        return weights


    def init_weights(self, is_recurrent_weights: bool):
        random.seed(self.seed)
        np.random.seed(self.seed)

        if is_recurrent_weights:
            (kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_intralayer_sequential,
            weights_seq_recurrent) = self.init_recurrent_weights()
        else:
            (kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_intralayer_sequential) = self.init_single_sequential_weights()
            weights_seq_recurrent = None


        # output -> output
        weights_output_output = np.triu(
            np.random.uniform(-1, 1, (self.num_label_nodes, self.num_label_nodes)), k=1
        )

        return (
            kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_output_output,
            weights_intralayer_sequential,
            weights_seq_recurrent
        )

    def init_single_sequential_weights(self):
        kernel_weights = []
        weights_sequential_layer = []
        weights_intralayer_sequential = []
        weights_hidden_to_output = []

        for _ in range(self.num_filter_kernels):
            kernel_weights.append(np.random.uniform(-1, 1, (self.kernel_size, self.kernel_size)))

        # pooled -> first seq
        weights_pool_to_first_seq = []
        for fk in range(self.num_filter_kernels):
            num_seq_units = self.sequential_layer_sizes[0]
            weights_pool_to_first_seq.append(
                np.random.uniform(-1, 1, (self.num_active_units_per_layer[1], num_seq_units)))
        weights_sequential_layer.append(weights_pool_to_first_seq)

        # hidden -> hidden (interlayer)
        weights_sequential_current = []
        for i, num_units in enumerate(self.sequential_layer_sizes[1:]):
            weights_sequential_current.append(
                np.random.uniform(-1, 1, (self.num_active_units_per_layer[2 + i], num_units)))
        weights_sequential_layer.append(weights_sequential_current)

        # hidden -> hidden (intralayer)
        weights_intralayer_sequential_current = []
        for size in self.sequential_layer_sizes:
            weights = np.triu(np.random.uniform(-1, 1, size))
            weights_intralayer_sequential_current.append(weights)
        weights_intralayer_sequential.append(weights_intralayer_sequential_current)

        # Last hidden -> output
        weights_hidden_to_output.append(
            self.init_weights_hidden_to_output(self.num_active_units_per_layer[-1], self.num_label_nodes)
        )

        return (
            kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_intralayer_sequential
        )

    def init_recurrent_weights(self):
        kernel_weights = []
        weights_sequential_layer = []
        weights_intralayer_sequential = []
        weights_hidden_to_output = []
        weights_seq_recurrent = []

        for layer in range(self.num_filter_kernels):
            kernel_weights.append(np.random.uniform(-1, 1, (self.kernel_size, self.kernel_size)))

            # hidden -> hidden (interlayer)
            weights_sequential_current_recurrent = []
            for i, num_units in enumerate(self.sequential_layer_sizes):
                weights_sequential_current_recurrent.append(
                    np.random.uniform(-1, 1, (self.num_active_units_per_layer[1 + i], num_units)))
            weights_sequential_layer.append(weights_sequential_current_recurrent)

            # hidden -> hidden (intralayer)
            weights_intralayer_sequential_current_recurrent = []
            for size in self.sequential_layer_sizes:
                weights = np.triu(np.random.uniform(-1, 1, size))
                weights_intralayer_sequential_current_recurrent.append(weights)
            weights_intralayer_sequential.append(weights_intralayer_sequential_current_recurrent)

            weights_recurrent_current = []
            for size in self.sequential_layer_sizes:
                weights = np.random.uniform(-1, 1, (size, size))
                weights_recurrent_current.append(weights)
            weights_seq_recurrent.append(weights_recurrent_current)

            if layer == 1 and self.num_filter_kernels == 2:
                # first to last
                weights_recurrent_current = []
                for size in self.sequential_layer_sizes:
                    weights = np.zeros((size, size))
                    weights_recurrent_current.append(weights)
                weights_seq_recurrent.append(weights_recurrent_current)

            # Last hidden -> output
            weights_hidden_to_output.append(
                self.init_weights_hidden_to_output(self.num_active_units_per_layer[-1], self.num_label_nodes)
            )

        return (
            kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_intralayer_sequential,
            weights_seq_recurrent
        )

    def init_biases(self, is_recurrent_weights: bool):
        biases_conv_units = []
        biases_sequential_units = []
        for recurrent_layer in range(self.num_filter_kernels):
            if self.hidden_bias_type == "shared":
                biases_conv_units.append(np.random.uniform(-1, 1, 1))
            elif self.hidden_bias_type == "none":
                biases_conv_units.append(np.zeros(self.sequential_layer_sizes))  # TODO: not working
            else:  # self.hidden_bias_type == "individual"
                biases_conv_units.append(np.random.uniform(-1, 1, self.num_conv_units))

        if is_recurrent_weights:
            for recurrent_layer in range(self.num_filter_kernels):
                sequential_biases_current_recurrent = []
                for size in self.sequential_layer_sizes:
                    sequential_biases_current_recurrent.append(np.random.uniform(-1, 1, size))
                biases_sequential_units.append(sequential_biases_current_recurrent)
        else:
            sequential_biases_current_recurrent = []
            for size in self.sequential_layer_sizes:
                sequential_biases_current_recurrent.append(np.random.uniform(-1, 1, size))
            biases_sequential_units.append(sequential_biases_current_recurrent)

        biases_output = np.random.uniform(-1, 1, self.num_label_nodes)

        return biases_conv_units, biases_sequential_units, biases_output


    def init_params(self, is_recurrent_weights=False):
        (
        kernel_weights,
        weights_sequential_layer,
        weights_hidden_to_output,
        weights_output_output,
        weights_interlayer_sequential,
        weights_seq_recurrent
        ) = self.init_weights(is_recurrent_weights)

        (
        biases_conv_units,
        biases_sequential_units,
        biases_output
        ) = self.init_biases(is_recurrent_weights)

        return (
            kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_output_output,
            weights_interlayer_sequential,
            weights_seq_recurrent,
            biases_conv_units,
            biases_sequential_units,
            biases_output
        )

    def build_layer_indexing(self, example_image, is_recurrent_weights=False) ->  StackSpec:
        currentlayer_fmap_2d = conv2d_valid_stride(example_image, self.kernel_weights[0], self.stride)
        currentlayer_fmap_flat = currentlayer_fmap_2d.ravel()

        mock_pooled_idx = pooled_indices_for_input(
            fmap_flat=currentlayer_fmap_flat,
            num_conv_units=self.num_conv_units,
            pooling_type=self.pooling_type,
            pool_windows=getattr(self, "pool_windows", []),
        )

        conv_active = []
        seq = []
        num_pooled_units_per_recurrent_layer = []
        for _ in range(self.num_filter_kernels):
            conv_active.append(len(mock_pooled_idx) if self.pooling_type == "deterministic" else self.num_conv_units)
            num_pooled_units_per_recurrent_layer.append(len(mock_pooled_idx))
        if is_recurrent_weights:
            for _ in range(self.num_filter_kernels):
                seq.append(SeqSpec(self.sequential_layer_sizes))
        else:
            seq.append(SeqSpec(self.sequential_layer_sizes))

        spec = StackSpec(
            conv_active=conv_active,
            seq=seq,
            n_out=self.num_label_nodes,
            pooling_type=self.pooling_type,
            n_pooled_units=num_pooled_units_per_recurrent_layer,
            num_recurrent_layers=self.num_filter_kernels if is_recurrent_weights else 1,
            num_filter_kernels=self.num_filter_kernels
        )


        return spec


    def save_weights(self, title="", path=""):
        if path == "":
            path = self.speicherort
        if title == "":
            # take current date and time as title
            title = "cdqbm_weights_" + str(np.datetime64('now')).replace(":", "-").replace(" ", "_")
        file_path = Path(path) / f"{title}.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(self.weight_objects, file)

    def count_parameters(self) -> int:
        total_params = 0
        for weight_matrix in self.weight_objects:
            if isinstance(weight_matrix, list):
                for item in weight_matrix:
                    if isinstance(item, list):
                        for sub_item in item:
                            total_params += sub_item.size
                    else:
                        total_params += item.size
            else:
                total_params += weight_matrix.size if weight_matrix is not None else 0
        return total_params


