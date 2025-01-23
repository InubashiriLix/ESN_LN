import numpy as np


class EsnModel:

    def __init__(self, n_reservoirs, lr: float = 0.8, sr: float = 1e-7):
        """
        initialize the model with the num of reservoirs, learning rate and spectral radius
        the lr will be 0.5 if not specified
        and the sr will be 1e-7 if not specified

        :param n_reservoirs: specify how many reservoirs in the model
        :param lr: specify the learning rate
        :param sr: specify the spectral radius
        """

        self.n_reservoirs = n_reservoirs
        self.lr = lr
        self.sr = sr

        self.W_in = np.random.rand(n_reservoirs, 1) - 0.5
        self.W_res = np.random.rand(n_reservoirs, n_reservoirs) - 0.5
        self.W_res *= sr / np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_out = None

        self.reservoirs_states = None

    def train(self, input_data: np.ndarray, target_data: np.ndarray):
        """
        train the model with the input data and output data
        :param input_data: the input data
        :param target_data: the output data
        """
        self.W_out = np.dot(np.linalg.pinv(self.run_reservoir(input_data)), target_data)

    def predict(self, input_data):
        return np.dot(self.run_reservoir(input_data), self.W_out)

    def run_reservoir(self, input_data: np.ndarray) -> np.ndarray:
        """
        run the reservoir with the input data
        :return: np.ndarray to contain the result
        """
        # initialize the reservoir
        self.reservoirs_states = np.zeros((len(input_data), self.n_reservoirs))
        # run the reservoir
        for i in range(1, len(input_data)):
            self.reservoirs_states[i, :] = np.tanh(
                np.dot(self.W_res, self.reservoirs_states[i - 1, :]) + np.dot(self.W_in, input_data[i]))

        return self.reservoirs_states
