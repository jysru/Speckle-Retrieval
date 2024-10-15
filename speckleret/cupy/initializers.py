import cupy as cp


def flat_phases(magnitude: cp.ndarray):
    phi = cp.zeros(shape=magnitude.shape)
    if cp.iscomplexobj(magnitude):
        return cp.abs(magnitude) * cp.exp(1j * phi)
    else:
        return magnitude * cp.exp(1j * phi)
    

def random_phases(magnitude: cp.ndarray):
    phi = 2 * cp.pi * cp.random.rand(*magnitude.shape)
    if cp.iscomplexobj(magnitude):
        return cp.abs(magnitude) * cp.exp(1j * phi)
    else:
        return magnitude * cp.exp(1j * phi)
    

def power_method(*args, **kwargs):
    raise NotImplementedError


def wirtinger_flow(*args, **kwargs):
    raise NotImplementedError


def gao_xu(*args, **kwargs):
    raise NotImplementedError
