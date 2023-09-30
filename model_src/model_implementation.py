import numpy as np


class LogisticRegression:
    def __init__(self, n_iters=1000, lr=0.001, threshold=0.5):
        self.n_iters = n_iters
        self.lr = lr
        self.threshold = threshold
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x_train, y_train):
        n_rows, n_cols = x_train.shape
        np.random.seed(40)
        self.weights = np.random.randn(n_cols)
        self.bias = 0

        for i in range(self.n_iters):
            lin_comb = np.matmul(x_train, self.weights) + self.bias
            predicted_probas = self._sigmoid(lin_comb)
            predicted_classes = np.where(predicted_probas >= self.threshold, 1, 0)

            self.weights -= self.lr * (2 / n_rows) * np.matmul(predicted_classes - y_train, x_train)
            self.bias -= self.lr * (2 / n_rows) * np.sum(predicted_classes - y_train)

    def predict(self, x_test):
        lin_comb = np.matmul(self.weights, x_test.T) + self.bias
        pred_proba = self._sigmoid(lin_comb)
        y_pred = np.where(pred_proba >= self.threshold, 1, 0)
        return y_pred
