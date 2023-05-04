import numpy as np
import matplotlib.pyplot as plt
from speckleret.plots import compare_complex_fields


def crop_img(img: np.ndarray, newsize: tuple[int, int]):
    """ Crop image to new size, from the center."""
    diff_row = img.shape[0] - newsize[0]
    diff_col = img.shape[1] - newsize[1]
    crop_row, crop_col = diff_row // 2, diff_col // 2
    return img[crop_row:-crop_row, crop_col:-crop_col]


def pad_img(img: np.ndarray, pad: float = 1):
    """ Pad zeroes on each axis.
        Default pad value is 1, which means that the padded image is twice the original size on each axis.
    """
    return np.pad(img, pad_width=[int(img.shape[0] * pad / 2), int(img.shape[1] * pad / 2)], mode='constant')


def fourier_transform(field: np.ndarray, pad: float = None):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)
        
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    ft = ft / np.sqrt(ft.size)
    return crop_img(ft, init_shape) if pad is not None else ft


def fresnel_transform(field: np.ndarray, dz: float = 0.0, pad: float = 2, wavelength: float = 1064e-9, pixel_size: float = 5.04e-6):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)
    
    _, _, kx, ky = fourier_grids(field, pixel_size=pixel_size)

    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    propagator = dz * np.sqrt(4 * np.square(np.pi/wavelength) - np.square(kx) - np.square(ky))
    ft = ft * np.exp(1j * propagator)
    ift = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ft)))
    return crop_img(ift, init_shape) if pad is not None else ft


def fourier_grids(field: np.ndarray, pixel_size: float):
    # Spatial plane
    dx = pixel_size
    n_pts = field.shape[0]
    grid_size = dx * (n_pts - 1)
    lim_x = n_pts / 2 * dx
    x = np.arange(-lim_x, lim_x, dx)
    x, y = np.meshgrid(x, x)

    # Conjugate plane
    dnx = 1 / grid_size
    lim_nx = (n_pts / 2) * dnx
    kx = 2 * np.pi * np.arange(-lim_nx, lim_nx, dnx)
    kx, ky = np.meshgrid(kx, kx)
    return (x, y, kx, ky)


def inverse_fourier_transform(field: np.ndarray, pad: float = None):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)
        
    ift = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field)))
    ift = ift * np.sqrt(ift.size)
    return crop_img(ift, init_shape) if pad is not None else ift


if __name__ == "__main__":
    field = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
    ft = fourier_transform(field, pad=2)
    ift = inverse_fourier_transform(ft, pad=2)

    compare_complex_fields(field, ift)
    plt.show()