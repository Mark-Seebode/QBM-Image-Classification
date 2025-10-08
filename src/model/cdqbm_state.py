import random
from pathlib import Path
import pickle
import numpy as np
from src.model.model_ab import MODEL
from src.model.geometry import (
    conv_output_shape, get_input_groups_coords, build_pool_windows,
    num_conv_units_from_dim, count_pooled_units
)
from src.qubo.sampler import LocalSASampler, DWaveAdapter


class Conv_Deep_QBM(MODEL):
    def __init__(self, num_visible_nodes, num_lable_nodes, image_shape=(28,28), seed=77, kernel_size=3, pooling_size=0,
                 pooling_type="deterministic", stride=1, sequential_layer_sizes=None,
                 param_string="", load_path="", speicherort=None, is_restricted=False,
                hidden_bias_type="none", solver="SA", anneal=1000, token=""):

        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.stride = stride
        self.image_shape = image_shape

        if sequential_layer_sizes is None:
            sequential_layer_sizes = []
        self.sequential_layer_sizes = sequential_layer_sizes

        (num_hidden_nodes,
         self.num_active_units,
         self.num_hidden_units_per_layer,
         self.num_active_units_per_layer,
         self.input_groups,
         self.conv_layer_dim,
         self.num_conv_units) = self.build_model_structure()

        super().__init__(seed, num_hidden_nodes, num_visible_nodes, num_lable_nodes, is_restricted)

        self.hidden_bias_type = hidden_bias_type

        self.weight_objects = [self.kernel_weights,
                            self.weights_sequential_layer,
                            self.weights_hidden_to_output,
                            self.weights_output_output,
                            self.weights_interlayer_sequential,
                            self.biases_conv_units,
                            self.biases_sequential_units,
                            self.biases_output] = self.init_params()

        self.param_string = param_string

        self.load_path = load_path
        self.speicherort = speicherort

        self.sampler = self.init_sampler(solver, anneal, seed, token)


    def init_sampler(self, solver="SA", anneal=1000, seed=77, token=""):
        # -------------------
        # Sampler
        # -------------------
        if solver.upper() == "SA":
            sampler = LocalSASampler(num_sweeps=anneal, seed=seed)
        else:
            # Optional: D-Wave path if you want it
            try:
                from dwave.cloud import Client
                client = Client(token=token, solver=solver)
                solver_obj = client.get_solver(name=solver)
                sampler = DWaveAdapter(solver=solver_obj, embedding=None, seed=seed)
                print(f"Using D-Wave solver: {solver}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize D-Wave solver '{solver}'. "
                    f"Use --solver SA or ensure TOKEN and solver name are valid. Error: {e}"
                )

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


    def init_weights(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

        kernel_weights = np.random.uniform(-1, 1, (self.kernel_size, self.kernel_size))

        weights_sequential_layer = []
        for i, num_units in enumerate(self.sequential_layer_sizes):
            weights_sequential_layer.append(
                np.random.uniform(-1, 1, (self.num_active_units_per_layer[1+i], num_units)))

        if not self.is_restricted:
            weights_interlayer_sequential = []
            for size in self.sequential_layer_sizes:
                weights = np.triu(np.random.uniform(-1, 1, size))
                weights_interlayer_sequential.append(weights)
        else:
            weights_interlayer_sequential = None

        # Last hidden -> output
        weights_hidden_to_output = self.init_weights_hidden_to_output(
            self.num_active_units_per_layer[-1], self.num_lable_nodes
        )
        # output -> output
        weights_output_output = np.triu(
            np.random.uniform(-1, 1, (self.num_lable_nodes, self.num_lable_nodes)), k=1
        )

        return (
            kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_output_output,
            weights_interlayer_sequential
        )

    def init_biases(self):
        # Biases
        if self.hidden_bias_type == "shared":
            biases_conv_units = np.random.uniform(-1, 1, 1)  # TODO: currently only one Conv filter supported
        elif self.hidden_bias_type == "none":
            biases_conv_units = np.zeros(self.sequential_layer_sizes)  # TODO: not working
        else:
            biases_conv_units = np.random.uniform(-1, 1, self.num_conv_units)

        biases_sequential_units = np.random.uniform(-1, 1, sum(self.sequential_layer_sizes))

        biases_output = np.random.uniform(-1, 1, self.num_lable_nodes)
        return biases_conv_units, biases_sequential_units, biases_output


    def init_params(self):
        (
        kernel_weights,
        weights_sequential_layer,
        weights_hidden_to_output,
        weights_output_output,
        weights_interlayer_sequential
        ) = self.init_weights()

        (
        biases_conv_units,
        biases_sequential_units,
        biases_output
        ) = self.init_biases()

        return (
            kernel_weights,
            weights_sequential_layer,
            weights_hidden_to_output,
            weights_output_output,
            weights_interlayer_sequential,
            biases_conv_units,
            biases_sequential_units,
            biases_output
        )

