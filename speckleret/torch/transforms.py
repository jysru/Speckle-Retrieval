"""Centered unitary Fourier transforms and sampled free-space propagation."""
from functools import lru_cache
import warnings

import torch
import torch.nn.functional as F


def apply_support(field, mask):
    """Zero values outside a boolean mask broadcastable to the field."""
    return torch.where(mask, field, 0)


def _size(img, target_size):
    if img.ndim != 4 or len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError('Expected NCHW input and positive (H,W) target size')


def crop_to_size(img, target_size):
    _size(img, target_size)
    h, w = img.shape[-2:]
    nh, nw = target_size
    if nh > h or nw > w:
        raise ValueError('Crop target exceeds input size')
    top, left = (h - nh) // 2, (w - nw) // 2
    return img[..., top:top + nh, left:left + nw]


def pad_to_size(img, target_size):
    _size(img, target_size)
    h, w = img.shape[-2:]
    dh, dw = target_size[0] - h, target_size[1] - w
    if dh < 0 or dw < 0:
        raise ValueError('Padding target is smaller than input')
    return F.pad(img, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))


def resize_image(img, target_size):
    """Center crop and/or zero pad each axis independently (no resampling)."""
    _size(img, target_size)
    h, w = img.shape[-2:]
    cropped = crop_to_size(img, (min(h, target_size[0]), min(w, target_size[1])))
    return pad_to_size(cropped, target_size)


def normalize_field(field, norm_type='energy'):
    norm_type = norm_type.lower()
    if norm_type == 'energy':
        norm = field.abs().square().sum(dim=(-2, -1), keepdim=True).sqrt()
    elif norm_type == 'max':
        norm = field.abs().amax(dim=(-2, -1), keepdim=True)
    else:
        raise ValueError("norm_type must be 'energy' or 'max'")
    return field / torch.where(norm > 0, norm, 1)


def fourier_transform(field, norm='ortho'):
    """Centered FFT2 on the last two dimensions, including odd sizes."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm=norm),
        dim=(-2, -1),
    )


def inverse_fourier_transform(field, norm='ortho'):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm=norm),
        dim=(-2, -1),
    )


def _spacings(pixel_size):
    dy, dx = (pixel_size, pixel_size) if isinstance(pixel_size, (int, float)) else pixel_size
    if min(dy, dx) <= 0:
        raise ValueError('Pixel spacings must be positive (meters)')
    return float(dy), float(dx)


def fourier_grids(field, pixel_size):
    """Return x,y,kx,ky grids; pixel_size is scalar or (dy,dx) in meters."""
    h, w = field.shape[-2:]
    dy, dx = _spacings(pixel_size)
    kwargs = dict(device=field.device, dtype=field.real.dtype)
    x = (torch.arange(w, **kwargs) - w // 2) * dx
    y = (torch.arange(h, **kwargs) - h // 2) * dy
    kx = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(w, dx, **kwargs))
    ky = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(h, dy, **kwargs))
    yg, xg = torch.meshgrid(y, x, indexing='ij')
    kyg, kxg = torch.meshgrid(ky, kx, indexing='ij')
    return xg, yg, kxg, kyg


@lru_cache(maxsize=16)
def _transfer(h, w, dz, wavelength, dy, dx, device, dtype, method, evanescent):
    if wavelength <= 0:
        raise ValueError('Wavelength must be positive (meters)')
    if method not in ('angular_spectrum', 'fresnel'):
        raise ValueError("method must be 'angular_spectrum' or 'fresnel'")
    if evanescent not in ('discard', 'decay'):
        raise ValueError("evanescent must be 'discard' or 'decay'")
    field = torch.empty((h, w), device=device, dtype=dtype)
    _, _, kx, ky = fourier_grids(field, (dy, dx))
    k = 2 * torch.pi / wavelength
    transverse = kx.square() + ky.square()
    if dz == 0:
        return torch.ones_like(kx, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)
    if method == 'fresnel':
        return torch.exp(1j * dz * (k - transverse / (2 * k)))
    val = k * k - transverse
    if evanescent == 'decay':
        if dz < 0:
            raise ValueError('Backward evanescent amplification is unstable; use discard')
        kz = torch.sqrt(val.to(torch.complex128 if dtype == torch.float64 else torch.complex64))
        return torch.exp(1j * dz * kz)
    return torch.where(val >= 0, torch.exp(1j * dz * val.clamp_min(0).sqrt()), 0)


def propagation_transfer_function(field, dz=0., wavelength=1064e-9,
                                  pixel_size=5.04e-6, method='angular_spectrum',
                                  evanescent='discard'):
    """Cached HxW transfer function. Treat returned cached tensors as read-only.

    Static propagation parameters are Python scalars; distance/wavelength and
    sampling are in meters. Cache is bounded to 16 configurations; call
    clear_propagation_cache() to release retained device tensors.
    """
    dy, dx = _spacings(pixel_size)
    return _transfer(*field.shape[-2:], float(dz), float(wavelength), dy, dx,
                     field.device, field.real.dtype, method, evanescent)


def clear_propagation_cache():
    _transfer.cache_clear()


def angular_spectrum_transform(field, transfer_function=None, dz=0.,
                               wavelength=1064e-9, pixel_size=5.04e-6,
                               evanescent='discard'):
    """Scalar angular-spectrum propagation; evanescent modes discarded by default."""
    if not torch.is_complex(field):
        raise ValueError('Input field must be complex')
    if transfer_function is None:
        transfer_function = propagation_transfer_function(
            field, dz, wavelength, pixel_size, evanescent=evanescent)
    return inverse_fourier_transform(fourier_transform(field) * transfer_function)


def fresnel_propagation(field, dz=0., wavelength=1064e-9, pixel_size=5.04e-6):
    """Paraxial Fresnel propagation (same input/output pixel spacing)."""
    transfer = propagation_transfer_function(field, dz, wavelength, pixel_size, method='fresnel')
    return angular_spectrum_transform(field, transfer_function=transfer)


def fresnel_transform(field, propagator=None, dz=0., wavelength=1064e-9, pixel_size=5.04e-6):
    """Compatibility alias for historical angular-spectrum propagation.

    The legacy propagator argument is a phase in radians, not a complex transfer
    function. Use angular_spectrum_transform or fresnel_propagation in new code.
    """
    warnings.warn('fresnel_transform historically implements angular spectrum; '
                  'use angular_spectrum_transform or fresnel_propagation',
                  DeprecationWarning, stacklevel=2)
    transfer = None if propagator is None else torch.exp(1j * propagator)
    return angular_spectrum_transform(field, transfer, dz, wavelength, pixel_size)
