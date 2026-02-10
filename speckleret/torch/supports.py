import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn.functional as F


def pixels_meshgrids(size: int, center: bool = True, return_polar: bool = True, device="cpu", dtype=torch.float):
    grid = torch.arange(size, dtype=dtype, device=device)
    if center:
        grid = grid - float(size) / 2 + 0.5
    Y, X = torch.meshgrid(grid, grid, indexing='ij')

    if return_polar:
        R = torch.sqrt(torch.square(X) + torch.square(Y))
        T = torch.atan2(Y, X)
        return X, Y, R, T
    else:
        return X, Y
    

def threshold_support(array: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Compute a binary support mask of shape [N, C, H, W],
    where the squared normalized amplitude exceeds the threshold.
    """
    if array.ndim < 4:
        raise ValueError(f"Input must be 4D [N, C, H, W], got shape {array.shape}")
    
    abs_val = array.abs()
    norm_abs = abs_val / abs_val.amax(dim=(-2, -1), keepdim=True)
    support = (norm_abs ** 2 >= threshold)
    return support
    

def square_support(array: torch.Tensor, size: int, offsets: tuple[float, float] = None) -> torch.Tensor:
    """Binary square-shaped support mask"""
    H, W = array.shape[-2:]
    X, Y = pixels_meshgrids(H, center=True, return_polar=False, device=array.device)

    if offsets is not None:
        X = X - offsets[0]
        Y = Y - offsets[1]

    mask = (X.abs() <= size / 2) & (Y.abs() <= size / 2)
    return mask.expand(*array.shape[:-2], H, W)


def disk_support(array: torch.Tensor, radius: float, offsets: tuple[float, float] = None) -> torch.Tensor:
    """Binary circular support mask"""
    H, W = array.shape[-2:]
    X, Y, R, _ = pixels_meshgrids(H, center=True, device=array.device)

    if offsets is not None:
        X = X - offsets[0]
        Y = Y - offsets[1]
        R = torch.sqrt(X**2 + Y**2)

    mask = (R <= radius)
    return mask.expand(*array.shape[:-2], H, W)


def shrinkwrap(field: torch.Tensor, filter_sigma: float = 1.0, threshold: float = 0.03) -> torch.Tensor:
    """
    Shrinkwrap support estimation per image (over H, W) for a 4D tensor [N, C, H, W].
    Applies Gaussian filtering to squared amplitude and thresholds it.
    """
    if field.ndim != 4:
        raise ValueError(f"Input must be 4D [N, C, H, W], got {field.shape}")

    N, C, H, W = field.shape

    # Compute normalized intensity (|field|^2 / max^2 per [N, C])
    abs_val = field.abs()
    norm = abs_val / abs_val.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    density = norm ** 2

    # Build Gaussian kernel
    ksize = int(2 * round(3 * filter_sigma) + 1)
    kernel = _gaussian_kernel2d(ksize, filter_sigma, device=field.device, dtype=field.dtype)
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C * N, 1, 1, 1)  # Depthwise per image

    # Prepare input for depthwise convolution: [N*C, 1, H, W]
    input_reshaped = density.view(N * C, 1, H, W)

    # Apply Gaussian blur with same padding
    blurred = F.conv2d(input_reshaped, kernel, padding=ksize // 2, groups=1)
    blurred = blurred.view(N, C, H, W)

    # Threshold to get support mask
    support = blurred >= threshold
    return support


def _gaussian_kernel2d(kernel_size: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel
