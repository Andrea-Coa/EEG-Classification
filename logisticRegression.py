import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate, n_iter, d, threshold):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = np.random.randn(d, 1)
        self.b = np.random.randn()
        self.threshold = threshold

    def error(self, x, y):
        s = sigmoid(np.matmul(x, self.w) + self.b) # "feedforward"
        return y - s                               # y - s = error

    def gradient_w(self, x, err):
        return np.matmul(x.T, err) / len(x)

    def fit(self, x, y):
        for _ in range(self.n_iter):
            alpha = self.learning_rate
            err = self.error(x, y)
            db = -  np.mean(err)
            dw = - self.gradient_w(x, err)

            self.w = self.w - alpha * dw
            self.b = self.b - alpha * db

            # if _ % 10 == 0:
            #     acc = self.accuracy(x, y)
            #     print("accuracy: ", acc)

    def predict(self, x):
        return (sigmoid(np.matmul(x, self.w) + self.b) >= self.threshold).astype(int)

    def accuracy(self, x, y):
        predictions = self.predict(x)
        correct_predictions = np.sum(predictions == y)
        return correct_predictions / len(y)

    def f1(self, actual_y, pred_y):
        tp = np.sum((actual_y + pred_y) == 2)
        fp = np.sum((actual_y - pred_y) == -1)
        fn = np.sum((actual_y - pred_y) == 1)

        print(tp)
        print(fp)
        print(fn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return 2 * precision * recall / (precision + recall)

    def set_parameters(self, w, b):
        self.w = w
        self.b = b