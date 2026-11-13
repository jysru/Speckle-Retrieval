"""Support masks and cached Gaussian smoothing for NCHW fields."""
from functools import lru_cache
import math
import torch
import torch.nn.functional as F


def pixels_meshgrids(size, center=True, return_polar=True, device='cpu', dtype=torch.float):
    h, w = (size, size) if isinstance(size, int) else size
    x = torch.arange(w, dtype=dtype, device=device)
    y = torch.arange(h, dtype=dtype, device=device)
    if center:
        x, y = x - (w - 1) / 2, y - (h - 1) / 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    return (X, Y, (X.square() + Y.square()).sqrt(), torch.atan2(Y, X)) if return_polar else (X, Y)


def threshold_support(array, threshold=0.01):
    if array.ndim != 4 or not 0 <= threshold <= 1:
        raise ValueError('Expected NCHW input and threshold in [0,1]')
    intensity = array.abs().square()
    maximum = intensity.amax(dim=(-2, -1), keepdim=True)
    return (maximum > 0) & (intensity >= threshold * maximum)


def square_support(array, size, offsets=None):
    X, Y = pixels_meshgrids(array.shape[-2:], return_polar=False,
                           device=array.device, dtype=array.real.dtype)
    ox, oy = (0, 0) if offsets is None else offsets
    mask = ((X - ox).abs() <= size / 2) & ((Y - oy).abs() <= size / 2)
    return mask.expand(array.shape)


def disk_support(array, radius, offsets=None):
    X, Y = pixels_meshgrids(array.shape[-2:], return_polar=False,
                           device=array.device, dtype=array.real.dtype)
    ox, oy = (0, 0) if offsets is None else offsets
    return ((X - ox).square() + (Y - oy).square() <= radius ** 2).expand(array.shape)


@lru_cache(maxsize=16)
def _gaussian_kernel2d(kernel_size, sigma, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def clear_gaussian_cache():
    _gaussian_kernel2d.cache_clear()


def gaussian_smoothing(field, sigma):
    """Independent spatial blur per image/channel, preserving complex dtype."""
    if sigma <= 0:
        return field
    h, w = field.shape[-2:]
    ksize = int(math.ceil(6 * sigma)) | 1
    kernel = _gaussian_kernel2d(ksize, float(sigma), field.device, field.real.dtype)[None, None]
    def blur(part):
        return F.conv2d(part.reshape(-1, 1, h, w), kernel,
                        padding=ksize // 2).reshape(field.shape)
    if torch.is_complex(field):
        return torch.complex(blur(field.real), blur(field.imag))
    return blur(field)


def shrinkwrap(field, filter_sigma=1., threshold=0.03):
    """Gaussian-blurred normalized intensity threshold, independently per image."""
    if field.ndim != 4 or filter_sigma < 0 or not 0 <= threshold <= 1:
        raise ValueError('Expected NCHW, nonnegative sigma and threshold in [0,1]')
    intensity = field.abs().square()
    maximum = intensity.amax(dim=(-2, -1), keepdim=True)
    density = intensity / torch.where(maximum > 0, maximum, 1)
    return (maximum > 0) & (gaussian_smoothing(density, filter_sigma) >= threshold)
