import cupy as cp


def pixels_meshgrids(size: int, center: bool = True, return_polar: bool = True):
    grid = cp.arange(start=0, stop=size)
    if center:
        grid = grid - float(size) / 2 + 0.5
    X, Y = cp.meshgrid(grid, grid)

    if return_polar:
        R = cp.sqrt(cp.square(X) + cp.square(Y))
        T = cp.arctan2(Y, X)
        return (X, Y, R, T)
    else:
        return (X, Y)
    

def threshold_support(array: cp.ndarray, threshold: float = 0.01):
    support = cp.zeros(array.shape, dtype=bool)
    support[cp.square(cp.abs(array) / cp.max(cp.abs(array))) >= threshold] = True
    return support
    

def square_support(array: cp.ndarray, size: int, offsets: tuple[float, float] = None):
    support = cp.zeros(array.shape, dtype=bool)
    X, Y = pixels_meshgrids(array.shape[0], center=True, return_polar=False)
    if offsets is None:
        support[cp.logical_and(cp.abs(X) <= size/2, cp.abs(Y) <= size/2) ] = True
    else:
        support[cp.logical_and(cp.abs(X - offsets[0]) <= size/2, cp.abs(Y - offsets[1]) <= size/2) ] = True
    return support
    

def disk_support(array: cp.ndarray, radius: int, offsets: tuple[float, float] = None):
    support = cp.zeros(array.shape, dtype=bool)
    X, Y, R, _ = pixels_meshgrids(array.shape[0], center=True)
    if offsets is None:
        support[R <= radius] = True
    else:
        support[cp.sqrt(cp.square(X - offsets[0]) + cp.square(Y - offsets[1])) <= radius] = True
    return support

