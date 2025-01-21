import numpy as np
from rtree import index

class kNNClassifier:
    def __init__(self, d=2):
        p = index.Property()
        p.dimension = d
        self.idx = index.Index(properties=p)

    def insert_data(self, X, y):
        self.labels = y
        for i, row in enumerate(X):
            elem = tuple(np.concatenate((row, row)))
            self.idx.insert(i, elem)

    def find_label(self, nearest, k):
        label0 = np.sum(self.labels[nearest] == 0)
        label1 = np.sum(self.labels[nearest] == 1)
        return int(label1 > label0)

    def predict(self, X, k=5):
        y_pred = []
        for x in X:
            nearest = list(self.idx.nearest(tuple(np.concatenate((x, x))), k))
            y_pred.append(self.find_label(nearest, k))
        return np.array(y_pred)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy