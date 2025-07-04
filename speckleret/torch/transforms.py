import torch
import torch.nn.functional as F


def apply_support(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a boolean mask on a 4D tensor and zero out the elements where mask is False.
    
    Args:
        field: input tensor of any dtype/shape (e.g., [N, C, H, W])
        mask: boolean tensor of the same shape as field
    
    Returns:
        Tensor with elements zeroed out where mask is False.
    """
    if field.shape != mask.shape:
        raise ValueError("field and mask must have the same shape")
    
    # Create a copy if you want to avoid modifying input in-place
    result = field.clone()
    result[~mask] = 0
    return result


def crop_to_size(img: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Crop the center H, W part of an image tensor [N, C, H, W] to newsize (new_H, new_W).
    """
    if img.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got {img.shape}")

    _, _, H, W = img.shape
    new_H, new_W = target_size

    if new_H > H or new_W > W:
        raise ValueError("newsize must be smaller than or equal to original size")

    crop_top = (H - new_H) // 2
    crop_left = (W - new_W) // 2

    return img[..., crop_top:crop_top + new_H, crop_left:crop_left + new_W]


def pad_to_size(img: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Pad a tensor of shape [N, C, H, W] with zeros to reach the target (H, W) size.
    The original image is centered in the padded output.

    Args:
        img: input tensor [N, C, H, W]
        target_size: (target_H, target_W)

    Returns:
        Padded tensor with shape [N, C, target_H, target_W]
    """
    if img.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got {img.shape}")

    _, _, H, W = img.shape
    target_H, target_W = target_size

    if target_H < H or target_W < W:
        raise ValueError("Target size must be greater than or equal to the original size")

    pad_top = (target_H - H) // 2
    pad_bottom = target_H - H - pad_top
    pad_left = (target_W - W) // 2
    pad_right = target_W - W - pad_left

    # Padding format: (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_img = F.pad(img, padding, mode='constant', value=0)
    return padded_img


def resize_image(img: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Resize an image tensor [N, C, H, W] to target_size (target_H, target_W).
    Pads if target_size > current size, crops if smaller.
    """
    _, _, H, W = img.shape
    target_H, target_W = target_size

    if target_H == H and target_W == W:
        return img  # no change

    if target_H > H or target_W > W:
        # Pad to target size
        return pad_to_size(img, target_size)

    if target_H < H or target_W < W:
        # Crop to target size
        return crop_to_size(img, target_size)

    # If mixed cases (one dim larger, other smaller), do crop then pad or vice versa:
    # For simplicity, crop then pad:
    cropped = crop_to_size(img, (
        min(H, target_H),
        min(W, target_W)
    ))
    resized = pad_to_size(cropped, target_size)
    return resized


def normalize_field(field: torch.Tensor, norm_type: str = 'energy') -> torch.Tensor:
    """
    Normalize NCHW tensor fields per sample & channel by energy or max over H,W.
    
    Args:
        field: complex or real tensor [N, C, H, W]
        norm_type: 'energy' or 'max' (case insensitive)
        
    Returns:
        normalized tensor of the same shape
    """
    norm_type = norm_type.lower()
    if norm_type not in ('energy', 'max'):
        raise ValueError("norm_type must be 'energy' or 'max'")

    abs_field = field.abs()
    
    if norm_type == 'energy':
        # Compute energy per N,C over H,W
        energy = torch.square(abs_field).sum(dim=(-2, -1), keepdim=True)
        norm_factor = torch.sqrt(energy)
    else:  # 'max'
        # Compute max magnitude per N,C over H,W
        norm_factor = abs_field.amax(dim=(-2, -1), keepdim=True)
    
    # Avoid division by zero
    norm_factor = torch.where(norm_factor == 0, torch.tensor(1., device=field.device, dtype=norm_factor.dtype), norm_factor)
    
    return field / norm_factor


def fourier_transform(field: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D Fourier transform of complex-valued input images in [N, C, H, W],
    applying FFT over the last two dims (H, W), with fftshift before and after.
    Returns the normalized FFT result.
    """
    if not torch.is_complex(field):
        raise ValueError("Input must be a complex-valued tensor.")

    # Apply fftshift before FFT
    field = torch.fft.fftshift(field, dim=(-2, -1))

    # Compute 2D FFT over H and W
    ft = torch.fft.fft2(field, dim=(-2, -1))

    # Apply fftshift after FFT
    ft = torch.fft.fftshift(ft, dim=(-2, -1))

    # Normalize
    norm_factor = torch.sqrt(torch.tensor(field.shape[-2] * field.shape[-1], dtype=field.dtype, device=field.device))
    return ft / norm_factor


def inverse_fourier_transform(field: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D inverse Fourier transform of complex-valued input images in [N, C, H, W],
    applying inverse FFT over the last two dims (H, W), with ifftshift before and after.
    Returns the appropriately scaled inverse FFT result.
    """
    if not torch.is_complex(field):
        raise ValueError("Input must be a complex-valued tensor.")

    # Apply ifftshift before iFFT
    field = torch.fft.ifftshift(field, dim=(-2, -1))

    # Compute 2D inverse FFT over H and W
    ift = torch.fft.ifft2(field, dim=(-2, -1))

    # Apply ifftshift after iFFT
    ift = torch.fft.ifftshift(ift, dim=(-2, -1))

    # Scale by sqrt(H*W), matching your NumPy code
    scale = torch.sqrt(torch.tensor(field.shape[-2] * field.shape[-1], dtype=field.dtype, device=field.device))
    return ift * scale


def fresnel_transform(field: torch.Tensor, dz: float = 0.0, wavelength: float = 1064e-9, pixel_size: float = 5.04e-6):
    """
    Fresnel propagation for a batch of fields [N, C, H, W] applied on last two dims.

    Args:
        field: complex tensor [N, C, H, W]
        dz: propagation distance
        wavelength: wavelength in meters
        pixel_size: pixel size in meters

    Returns:
        propagated field tensor [N, C, H, W]
    """
    if not torch.is_complex(field):
        raise ValueError("Input field must be a complex tensor")

    N, C, H, W = field.shape

    # Get Fourier grids (kx, ky) on CPU or GPU as per field device
    _, _, kx, ky = fourier_grids(field, pixel_size=pixel_size)
    kx = kx.to(field.device)
    ky = ky.to(field.device)

    # FFT with fftshift on last two dims
    ft = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))

    # Compute the propagator phase factor
    k = 2 * torch.pi / wavelength
    val = 4 * torch.square(k) - torch.square(kx) - torch.square(ky)
    # Clamp negative inside sqrt to zero to avoid NaNs (evanescent waves)
    val = torch.clamp(val, min=0)
    propagator = dz * torch.sqrt(val)  # shape [H,W]

    # Apply the propagator: expand dims to broadcast on N and C
    ft = ft * torch.exp(1j * propagator)[None, None, :, :]

    # Inverse FFT with ifftshift on last two dims
    propagated = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(ft, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))

    return propagated


def fourier_grids(field: torch.Tensor, pixel_size: float):
    """
    Compute Fourier coordinate grids for HxW images in tensor [N,C,H,W].

    Args:
        field: input tensor [N,C,H,W]
        pixel_size: spatial pixel size (meters)

    Returns:
        x, y, kx, ky: coordinate grids of shape [H,W]
    """
    H, W = field.shape[-2], field.shape[-1]
    dx = pixel_size
    # Spatial coordinates (centered)
    lim_x = W / 2 * dx
    x = torch.linspace(-lim_x, lim_x - dx, W)
    y = torch.linspace(-lim_x, lim_x - dx, H)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')  # shape [H,W]

    # Fourier (frequency) domain coordinates
    grid_size_x = dx * W
    grid_size_y = dx * H
    dnx = 1 / grid_size_x
    dny = 1 / grid_size_y

    lim_nx = W / 2 * dnx
    lim_ny = H / 2 * dny

    kx_vals = torch.linspace(-lim_nx, lim_nx - dnx, W) * 2 * torch.pi
    ky_vals = torch.linspace(-lim_ny, lim_ny - dny, H) * 2 * torch.pi
    ky_grid, kx_grid = torch.meshgrid(ky_vals, kx_vals, indexing='ij')  # shape [H,W]

    return x_grid, y_grid, kx_grid, ky_grid

