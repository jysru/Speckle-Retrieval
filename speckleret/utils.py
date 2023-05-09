import numpy as np
from scipy.io import loadmat
import scipy.ndimage
import skimage
import speckleret


def get_centroid(array: np.ndarray):
    if np.iscomplexobj(array):
        array = np.abs(array)
    return scipy.ndimage.center_of_mass(array)


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