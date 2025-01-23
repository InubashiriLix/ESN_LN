import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model):
        self.model = model
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def load_data(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def train(self):
        if self.X_train is None or self.Y_train is None:
            raise ValueError("data not set")

        self.model.train(self.X_train, self.Y_train)
        print("training finished")

    def predict(self, drawing, shift_fb, X_test, Y_test):

        if X_test is not None and Y_test is not None:
            pass
        elif X_test is None and Y_test is None:
            X_test = self.X_test
            Y_test = self.Y_test
        else:
            raise ValueError("X_test and Y_test must be both None or both not None")

        if len(X_test) != len(Y_test):
            raise ValueError("X_test and Y_test must have the same length")

        predictions = np.zeros(len(X_test))
        last_prediction = None
        last_train_data = None
        for i in range(len(X_test)):
            last_prediction = self.model.predict(X_test[i])
            predictions[i] = last_prediction
            last_train_data = X_test[i]
            if shift_fb and last_prediction is not None:
                self.model.train(last_train_data, last_prediction)

        if drawing:
            plt.plot(Y_test, label="real")
            plt.plot(predictions, label="prediction")
            plt.legend()
            plt.show()

        return predictions


