import numpy as np


class BatchGradientDescentRegressor:
    def __init__(self, learning_rate: float = 0.1, random_state: int = 42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None
        self.__z = None
        self.__af = None
        self.x = None
        self.y = None
        self.__errors = None
        self.loss = 1
        self.epoch = 0
        self.abort = 0

    def __initialise_weights(self):
        _, n_col = self.x.shape
        np.random.seed(self.random_state)
        self.weights = np.random.randn(n_col)

    def __sum_function(self):
        self.__z = np.dot(self.x, self.weights.reshape(-1, 1))
        self.__af = self.__z.ravel()

    def __weight_update(self):
        n_row = len(self.__af)
        self.__errors = self.__af - (self.y).ravel()
        self.weights = self.weights - (self.learning_rate * (1 / n_row) * np.dot(self.__errors, self.x))

    def __loss_update(self):
        self.loss = np.mean(self.__errors ** 2)

    def fit(self, x, y):
        self.x = x
        self.y = y
        n_row, _ = self.x.shape
        ones = np.ones((n_row, 1))
        self.x = np.hstack((ones, self.x))
        self.__initialise_weights()
        while np.sqrt(self.loss) > 0.009:
            self.__sum_function()
            self.__weight_update()
            self.__loss_update()
            self.epoch += 1
            print(f'Epoch {self.epoch}\nTraining MSE: {self.loss}')
            if self.epoch > 100000:
                self.abort = 1
                break

    def predict(self, x):
        n_row, _ = x.shape
        ones = np.ones((n_row, 1))
        x = np.hstack((ones, x))
        return np.dot(x, self.weights.reshape(-1, 1))
