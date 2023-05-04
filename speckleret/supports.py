import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def pixels_meshgrids(size: int, center: bool = True, return_polar: bool = True):
    grid = np.arange(start=0, stop=size)
    if center:
        grid = grid - float(size) / 2 + 0.5
    X, Y = np.meshgrid(grid, grid)

    if return_polar:
        R = np.sqrt(np.square(X) + np.square(Y))
        T = np.arctan2(Y, X)
        return (X, Y, R, T)
    else:
        return (X, Y)
    

def threshold_support(array: np.ndarray, threshold: float = 0.01):
    support = np.zeros(array.shape, dtype=bool)
    support[np.square(np.abs(array) / np.max(np.abs(array))) >= threshold] = True
    return support
    

def square_support(array: np.ndarray, size: int, offsets: tuple[float, float] = None):
    support = np.zeros(array.shape, dtype=bool)
    X, Y = pixels_meshgrids(array.shape[0], center=True, return_polar=False)
    if offsets is None:
        support[np.logical_and(np.abs(X) <= size/2, np.abs(Y) <= size/2) ] = True
    else:
        support[np.logical_and(np.abs(X - offsets[0]) <= size/2, np.abs(Y - offsets[1]) <= size/2) ] = True
    return support
    

def disk_support(array: np.ndarray, radius: int, offsets: tuple[float, float] = None):
    support = np.zeros(array.shape, dtype=bool)
    X, Y, R, _ = pixels_meshgrids(array.shape[0], center=True)
    if offsets is None:
        support[R <= radius] = True
    else:
        support[np.sqrt(np.square(X - offsets[0]) + np.square(Y - offsets[1])) <= radius] = True
    return support


def shrinkwrap(field, filter_sigma: float = 1, threshold: float = 0.03):
    # https://www.youtube.com/watch?v=8rJYhRMVvQw&t=1742s (0:40)
    density = np.square(np.abs(field) / np.max(np.abs(field)))
    smeared = gaussian_filter(density, sigma=filter_sigma)

    support = np.zeros(field.shape, dtype=bool)
    support[smeared >= threshold] = True
    return support


if __name__ == "__main__":
    arr = np.zeros((14,14))
    sup1 = disk_support(arr, radius=4, offsets=(1,-2))
    sup2 = square_support(arr, size=4, offsets=(-1,1))
    
    plt.imshow(sup1)
    plt.show()