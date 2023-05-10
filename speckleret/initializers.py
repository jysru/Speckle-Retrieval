import numpy as np


def flat_phases(magnitude: np.ndarray):
    phi = np.zeros(*magnitude.shape)
    if np.iscomplexobj(magnitude):
        return np.abs(magnitude) * np.exp(1j * phi)
    else:
        return magnitude * np.exp(1j * phi)
    

def random_phases(magnitude: np.ndarray):
    phi = 2 * np.pi * np.random.rand(*magnitude.shape)
    if np.iscomplexobj(magnitude):
        return np.abs(magnitude) * np.exp(1j * phi)
    else:
        return magnitude * np.exp(1j * phi)
    

def power_method(*args, **kwargs):
    raise NotImplementedError


def wirtinger_flow(*args, **kwargs):
    raise NotImplementedError


def gao_xu(*args, **kwargs):
    raise NotImplementedError
