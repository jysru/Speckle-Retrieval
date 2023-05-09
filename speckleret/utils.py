import numpy as np
from scipy.io import loadmat
import scipy.ndimage
import skimage
import speckleret


def nested_arrays_to_ndarray(nested_array, abs_if_complex: bool = True, sqrt_if_real: bool = True):
    ndarray = np.zeros(shape=(nested_array.shape + nested_array[0, 0].shape))

    for i in range(ndarray.shape[0]):
        for j in range(ndarray.shape[1]):
            if np.iscomplexobj(nested_array[i, j]):
                ndarray[i, j, ...] = np.abs(nested_array[i, j]) if abs_if_complex else nested_array[i, j]
            else:
                ndarray[i, j, ...] = np.sqrt(nested_array[i, j]) if sqrt_if_real else nested_array[i, j]
    return ndarray


def get_centroid(array: np.ndarray):
    if np.iscomplexobj(array):
        array = np.abs(array)
    return scipy.ndimage.center_of_mass(array)


def calculate_contrast(array, axis=(2,3), square: bool = False):
    if np.iscomplexobj(array):
        array = np.abs(array)
    if square:
        array = np.square(array)
    max, min = np.max(array, axis=axis), np.min(array, axis=axis)
    return (max - min) / (max + min)


def make_disk_mask(shape, radius, center: tuple[float, float] = None):
    mask = np.zeros(shape, dtype=bool)

    if center is None:
        center = (shape[0] // 2, shape[1] // 2)

    rr, cc = skimage.draw.disk(center, radius, shape=mask.shape)
    mask[rr, cc] = True
    return mask


def make_threshold_mask(array, threshold: float = 0.01):
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = array / np.max(array)

    mask = np.zeros(array.shape, dtype=bool)
    mask[array >= threshold] = True
    return mask


def calculate_energy_in_mask(intensity, mask):
    if np.iscomplexobj(intensity):
        intensity = np.square(np.abs(intensity))
    return np.sum(intensity[mask]) / np.sum(intensity)


def calculate_energies_in_mask_radii(intensity, center: tuple[float, float] = None):
    radii = np.arange(np.max(intensity.shape) // 2, 0, -1)
    energy_in = []
    energy_out = []

    for i in range(len(radii)):
        mask = make_disk_mask(intensity.shape, radius=radii[i], center=center)
        energy_in.append(calculate_energy_in_mask(intensity, mask))
        energy_out.append(calculate_energy_in_mask(intensity, np.logical_not(mask)))

    return np.array(energy_in), np.array(energy_out), radii


def disk_radius_for_energy(intensity, center: tuple[float, float] = None, energy_ratio: float = 0.95):
    if energy_ratio >= 1:
        energy_ratio = 0.99999999
        print(f"Coerced energy ratio to {energy_ratio}")
        
    radius = 1
    condition = True
    while condition:
        mask = make_disk_mask(intensity.shape, radius=radius, center=center)
        energy = calculate_energy_in_mask(intensity, mask)
        radius += 1
        condition = (energy <= energy_ratio)
    return radius