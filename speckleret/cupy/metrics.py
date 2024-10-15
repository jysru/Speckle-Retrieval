import cupy as cp


def mae(x: cp.ndarray, y: cp.ndarray):
    """Return the mean absolute error between the two arrays."""
    if cp.iscomplexobj(x):
        x = cp.abs(x)
    if cp.iscomplexobj(y):
        y = cp.abs(y)
    return cp.mean(cp.abs(x - y))


def mse(x: cp.ndarray, y: cp.ndarray):
    """Return the mean square error between the two arrays."""
    if cp.iscomplexobj(x):
        x = cp.abs(x)
    if cp.iscomplexobj(y):
        y = cp.abs(y)
    return cp.mean(cp.square(x - y))


def dot_product(x: cp.ndarray, y: cp.ndarray, normalized: bool = True):
    """Return the scalar product between the two complex arrays."""
    prod = cp.sum(x * cp.conjugate(y))
    norm = cp.sum(cp.abs(x) * cp.abs(y))
    return prod / norm if normalized else prod


def quality(x: cp.ndarray, y: cp.ndarray, squared: bool = True, inversed: bool = False):
    """Return the magnitude of the normalized dot product between the two complex arrays."""
    q = cp.abs(dot_product(x, y, normalized=True))
    if squared:
        q = cp.square(q)
    if inversed:
        q = 1 - q
    return q


def pearson(x, y, inversed: bool = False):
        if cp.iscomplexobj(x):
            x = cp.abs(x)
        if cp.iscomplexobj(y):
            y = cp.abs(y)

        s = cp.sum((x - cp.mean(x)) * (y - cp.mean(y)) / x.size)
        p = s / (cp.std(x) * cp.std(y))
        if inversed:
            p = 1 - p
        return p