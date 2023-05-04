import numpy as np
import matplotlib.pyplot as plt
import speckleret.transforms as transforms


def energy(field: np.ndarray):
    return np.sum(np.square(np.abs(field)))


def lowpass_fourier_filter(field: np.ndarray, radius: float, pixel_size: float = 5.04e6, pad: float = None):
    print(energy(field))
    
    ft = transforms.fourier_transform(field, pad=pad)
    print(energy(ft))

    x, y, _, _ = transforms.fourier_grids(field, pixel_size=pixel_size)
    r = np.sqrt(np.square(x) + np.square(y))

    filter = np.zeros(ft.shape, dtype=bool)
    filter[r >= radius] = True

    filtered_ft = ft.copy()
    filtered_ft[filter] = 0

    print(energy(filtered_ft))

    ift = transforms.inverse_fourier_transform(filtered_ft, pad=pad)
    print(energy(ift))
    return ift


def highpass_fourier_filter(field: np.ndarray, radius: float, pixel_size: float = 5.04e6, pad: float = None):
    print(energy(field))
    ft = transforms.fourier_transform(field, pad=pad)

    x, y, _, _ = transforms.fourier_grids(field, pixel_size=pixel_size)
    r = np.sqrt(np.square(x) + np.square(y))

    filter = np.zeros(ft.shape, dtype=bool)
    filter[r <= radius] = True

    filtered_ft = ft.copy()
    filtered_ft[filter] = 0

    ift = transforms.inverse_fourier_transform(filtered_ft, pad=pad)
    print(energy(ift))
    return ift

