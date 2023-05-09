import numpy as np
import matplotlib.pyplot as plt
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


def extract_noise_correction(array,
                             averaging_axis: tuple[int, int] = (0,1), mask_trsh: float = 0.02,
                             plot_mask: bool = False, plot_result: bool = False,
                             return_corrected_array: bool = False,
                             ):
    # Convert to N-dimension matrix for easy computations
    array = nested_arrays_to_ndarray(array)

    # Compute averages
    mean_abs = np.mean(array, axis=averaging_axis)
    mean_abs_norm = mean_abs / np.max(mean_abs)

    # Compute mask and plot
    mask = make_threshold_mask(np.square(mean_abs_norm), threshold=mask_trsh)
    if plot_mask:
        speckleret.plots.rgb_imshow([mean_abs_norm, mask*0.2], normalize_channels=False)
        plt.title("Average amplitude + Mask")
        plt.show()

    # Compute noise average and plot result
    mean_noise = np.mean(np.abs(array) * np.reshape(mask, ((1, 1) + mask.shape)))
    array_corr = np.abs(array - mean_noise)
    if plot_result:
        idx = np.random.randint(low=0, high=array.shape[1]-1)
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        pl0 = axs[0].imshow(speckleret.transforms.pad_img(array[0, idx, ...], pad=1), cmap='gray')
        pl1 = axs[1].imshow(speckleret.transforms.pad_img(array_corr[0, idx, ...], pad=1), cmap='gray')
        axs[0].set_title("Amplitude example")
        axs[1].set_title("Corrected amplitude example")
        plt.colorbar(pl0, ax=axs[0])
        plt.colorbar(pl1, ax=axs[1])

    if return_corrected_array:
        return mean_noise, array_corr
    else:
        return mean_noise


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