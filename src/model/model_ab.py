import abc
import pickle
import numpy as np

class MODEL(metaclass=abc.ABCMeta):

    def __init__(self, seed, num_hidden_nodes, num_visible_nodes, num_lable_nodes, is_restricted) -> None:
        np.random.seed(seed)

        self.num_hidden_nodes: int = num_hidden_nodes
        self.num_visible : int = num_visible_nodes

        self.num_lable_nodes = num_lable_nodes
        self.seed: int = seed

        self.weight_objects = None

        self.is_restricted: bool = is_restricted

    @abc.abstractmethod
    def load_params(self, file_path):
        """
        Load the model parameters from file
        """

    @abc.abstractmethod
    def init_params(self):
        """
        Initialize the model parameters
        """


    def save_weights(self, title, path=""):
        with open(f"{path}/{title}.pkl", "wb") as f:
            pickle.dump(self.weight_objects, f)

