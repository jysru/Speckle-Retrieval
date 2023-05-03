import numpy as np


def mse(x, y):
    return np.mean(np.square(np.abs(x) - np.abs(y)))


def quality(x, y):
    return np.power(np.abs(np.sum(x * np.conjugate(y))) / np.sum(np.abs(x) * np.abs(y)), 2)


def pearson(x, y):
        x = np.abs(x)
        y = np.abs(y)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)
        n = x.size

        s = np.sum((x - mean_x) * (y - mean_y) / n)
        r = s / (std_x * std_y)
        return r