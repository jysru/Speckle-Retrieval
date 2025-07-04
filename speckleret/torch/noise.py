import numpy as np
import torch
import math

intens_noise = 0*0.003
wavelength = 1064e-9
phase_noise = 0 * wavelength/30


def add_intensity_noise(field: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
    """Add multiplicative intensity noise to a complex field."""
    if noise_std == 0:
        return field

    device = field.device
    dtype = field.dtype

    intensity = field.abs().pow(2)
    max_intensity = intensity.amax()
    noise = torch.randn_like(intensity, device=device, dtype=torch.float)

    perturbed_intensity = max_intensity * torch.abs(intensity / max_intensity + noise_std * noise)
    new_field = torch.sqrt(perturbed_intensity) * torch.exp(1j * field.angle())

    return new_field.to(dtype)


def add_phase_noise(field: torch.Tensor, noise_std: float = 0.0, wavelength: float = 1064e-9) -> torch.Tensor:
    """Add phase noise to a complex field."""
    if noise_std == 0:
        return field

    device = field.device

    phase = field.angle()
    noise = torch.randn_like(phase, device=device)  # real-valued noise
    noisy_phase = torch.remainder(phase + 2 * math.pi / wavelength * noise_std * noise, 2 * math.pi)
    # Re-center around -π to π
    noisy_phase = (noisy_phase + math.pi) % (2 * math.pi) - math.pi

    return field.abs() * torch.exp(1j * noisy_phase)


def add_complex_noise(field: torch.Tensor, intens_noise_std: float = 0.0, phase_noise_std: float = 0.0, wavelength: float = 1064e-9) -> torch.Tensor:
    """Apply intensity and phase noise to a complex field."""
    if intens_noise_std != 0:
        field = add_intensity_noise(field, noise_std=intens_noise_std)
    if phase_noise_std != 0:
        field = add_phase_noise(field, noise_std=phase_noise_std, wavelength=wavelength)
    return field
