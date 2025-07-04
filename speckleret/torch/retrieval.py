import numpy as np
import matplotlib.pyplot as plt

from waveoptics.metrics.numpy import pearson
from waveoptics.plots.plots import complex_to_hsv

import speckleret.plots as plots
import speckleret.supports as supports

import scipy.ndimage as ndi

import torch
from . import metrics
from . import transforms
from . import initializers

import math

import torch.nn.functional as F

from typing import Callable, Dict, Tuple, Optional


# Review paper: https://doi.org/10.1063/1.2403783
# Google collab: https://colab.research.google.com/drive/1anePjgg1fKbYrCCmDeRKblobryq-O4Rv


def P_S(
        x: torch.Tensor,
        magnitude: torch.Tensor,
        support: torch.Tensor,
        apply_support: bool = True,
    ) -> torch.Tensor:
    """
    Support projection for complex tensor [N, C, H, W].

    Args:
        x: complex input tensor [N, C, H, W]
        magnitude: real tensor [N, C, H, W] with desired magnitudes
        support: boolean mask tensor [N, C, H, W]
        apply_support: whether to zero outside support

    Returns:
        projected complex tensor [N, C, H, W]
    """
    if not torch.is_complex(x):
        raise ValueError("Input x must be complex tensor")
    if magnitude.shape != x.shape:
        raise ValueError("Magnitude shape must match input shape")
    if support.shape != x.shape:
        raise ValueError("Support mask shape must match input shape")

    # Compute new field with original phase, new magnitude
    phase = torch.angle(x)
    x_new = magnitude * torch.exp(1j * phase)

    if apply_support:
        x_new = transforms.apply_support(x_new, support)

    return x_new


def P_M(
        x: torch.Tensor,
        magnitude: torch.Tensor,
        direct_transform: callable = transforms.fourier_transform,
        inverse_transform: callable = transforms.inverse_fourier_transform,
    ) -> torch.Tensor:
    """
    Fourier magnitude projection for complex-valued tensors [N, C, H, W].

    Args:
        x: Complex input tensor.
        magnitude: Desired magnitude tensor of same shape as x.
        direct_transform: Callable performing forward transform (e.g., FFT).
        inverse_transform: Callable performing inverse transform (e.g., IFFT).

    Returns:
        Tensor after enforcing magnitude constraint in Fourier domain.
    """
    if not torch.is_complex(x):
        raise ValueError("Input x must be a complex tensor")
    if magnitude.shape != x.shape:
        raise ValueError("Magnitude shape must match input shape")

    # Forward transform (Fourier domain)
    X = direct_transform(x)

    # Enforce magnitude, keep phase
    phase = torch.angle(X)
    X_new = magnitude * torch.exp(1j * phase)

    # Inverse transform back to spatial domain
    x_proj = inverse_transform(X_new)
    return x_proj


def R_S(
        x: torch.Tensor,
        gamma_S: float,
        magnitude: torch.Tensor,
        support: torch.Tensor,
        projector_S: callable = P_S,
    ) -> torch.Tensor:
    """
    Support reflector.

    Args:
        x: Complex input tensor [N,C,H,W].
        gamma_S: Relaxation parameter (float).
        magnitude: Desired magnitude tensor [N,C,H,W].
        support: Boolean support mask tensor [N,C,H,W].
        projector_S: Callable projector function, e.g. P_S.

    Returns:
        Reflected tensor [N,C,H,W].
    """
    proj = projector_S(x, magnitude, support)
    return (1 + gamma_S) * proj - gamma_S * x


def R_M(
        x: torch.Tensor,
        gamma_M: float,
        magnitude: torch.Tensor,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    Magnitude reflector.

    Args:
        x: Complex input tensor [N,C,H,W].
        gamma_M: Relaxation parameter (float).
        magnitude: Desired magnitude tensor [N,C,H,W].
        projector_M: Callable projector function, e.g. P_M.

    Returns:
        Reflected tensor [N,C,H,W].
    """
    # Apply the magnitude projector
    proj = projector_M(x, magnitude)
    # Reflector formula
    return (1 + gamma_M) * proj - gamma_M * x


def ER(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        projector_S: callable = P_S,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    Error Reduction algorithm iteration.

    Args:
        x: Complex input tensor [N,C,H,W].
        magnitude_S: Desired support magnitude [N,C,H,W].
        magnitude_M: Desired Fourier magnitude [N,C,H,W].
        support: Boolean support mask [N,C,H,W].
        projector_S: Support projector callable.
        projector_M: Magnitude projector callable.

    Returns:
        Updated tensor after ER iteration.
    """
    x_proj_M = projector_M(x, magnitude_M)
    x_proj_S = projector_S(x_proj_M, magnitude_S, support)
    return x_proj_S


def SF(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        reflector_S: callable = R_S,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    Solvent Flipping algorithm iteration.

    Args:
        x: Complex input tensor [N,C,H,W].
        magnitude_S: Desired support magnitude [N,C,H,W].
        magnitude_M: Desired Fourier magnitude [N,C,H,W].
        support: Boolean support mask [N,C,H,W].
        reflector_S: Support reflector callable.
        projector_M: Magnitude projector callable.

    Returns:
        Updated tensor after SF iteration.
    """
    x_proj_M = projector_M(x, magnitude_M)
    x_reflect_S = reflector_S(x_proj_M, gamma_S=1.0, magnitude=magnitude_S, support=support)
    return x_reflect_S


def HIO(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        projector_S: callable = P_S,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    Hybrid Input-Output (HIO) algorithm iteration.

    Args:
        x: Complex input tensor [N, C, H, W]
        magnitude_S: Desired support magnitude [N, C, H, W]
        magnitude_M: Desired Fourier magnitude [N, C, H, W]
        support: Boolean support mask [N, C, H, W]
        beta: Relaxation parameter
        projector_S: Callable for support projection
        projector_M: Callable for magnitude projection

    Returns:
        Updated tensor after HIO iteration
    """

    # Initialize result
    x_new = x.clone()

    # Apply projections
    proj = projector_M(x, magnitude_M)
    proj_S = projector_S(proj, magnitude_S, support)

    # Apply support region update
    x_new[support] = proj[support]

    # Apply outside support region update
    x_new[~support] = x[~support] - beta * proj_S[~support]

    return x_new


def DM(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        gamma_S: float = None,
        gamma_M: float = None,
        projector_S: callable = P_S,
        projector_M: callable = P_M,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Difference Map (DM) algorithm iteration.
    (Random projections and the optimization of an algorithm for phase retrieval, J. Phys. A-Math Gen, Vol 36 pp 2995-3007)

    Args:
        x: Complex tensor of shape [N, C, H, W]
        magnitude_S: Support-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean support mask [N, C, H, W]
        beta: Relaxation parameter
        gamma_S: Parameter for support reflector (defaults to 1 / beta)
        gamma_M: Parameter for magnitude reflector (defaults to -1 / beta)
        projector_S: Callable for support projection
        projector_M: Callable for magnitude projection
        reflector_S: Callable for support reflection
        reflector_M: Callable for magnitude reflection

    Returns:
        Updated complex tensor after DM iteration
    """
    if gamma_S is None:
        gamma_S = 1.0 / beta
    if gamma_M is None:
        gamma_M = -1.0 / beta

    x_PMRS = projector_M(reflector_S(x, gamma_S, magnitude_S, support), magnitude_M)
    x_PSRM = projector_S(reflector_M(x, gamma_M, magnitude_M), magnitude_S, support)
    x_new = x + beta * (x_PMRS - x_PSRM)

    return x_new


def ASR(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Averaged Successive Reflections (ASR) algorithm iteration.

    Args:
        x: Complex tensor of shape [N, C, H, W]
        magnitude_S: Support-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean support mask [N, C, H, W]
        reflector_S: Callable for support reflection
        reflector_M: Callable for magnitude reflection

    Returns:
        Updated complex tensor after ASR iteration
    """
    r_m = reflector_M(x, gamma_M=1.0, magnitude=magnitude_M)
    r_s = reflector_S(r_m, gamma_S=1.0, magnitude=magnitude_S, support=support)
    return 0.5 * (r_s + x)


def HPR(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        projector_M: callable = P_M,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Hybrid Projection Reflection algorithm iteration (HPR).
    
    Args:
        x: Complex tensor [N, C, H, W]
        magnitude_S: Spatial-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean mask [N, C, H, W]
        beta: Relaxation parameter (default: 0.7)
        projector_M: Fourier magnitude projection function
        reflector_S: Support reflector
        reflector_M: Fourier magnitude reflector

    Returns:
        Updated complex tensor after one HPR iteration
    """
    proj = projector_M(x, magnitude_M)
    reflect_M = reflector_M(x, gamma_M=1.0, magnitude=magnitude_M)
    x1 = reflector_S(reflect_M + (beta - 1) * proj, gamma_S=1.0, magnitude=magnitude_S, support=support)
    x2 = (1 - beta) * proj
    return 0.5 * (x1 + x + x2)


def RAAR(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        projector_M: callable = P_M,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Relaxed Averaged Alternating Reflectors (RAAR) algorithm iteration.
    
    Args:
        x: Complex tensor [N, C, H, W]
        magnitude_S: Spatial-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean mask [N, C, H, W]
        beta: Relaxation parameter (default: 0.7)
        projector_M: Fourier magnitude projector
        reflector_S: Spatial support reflector
        reflector_M: Fourier magnitude reflector

    Returns:
        Updated complex tensor after one RAAR iteration
    """
    proj = projector_M(x, magnitude_M)
    asr = ASR(x, magnitude_S, magnitude_M, support, reflector_S=reflector_S, reflector_M=reflector_M)
    return beta * asr + (1 - beta) * proj


def OSS(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        sigma: float = 1.0,
        filter_decay: float = 0.99,
        iteration: int = 0,
        projector_S: callable = P_S,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    OverSampling Smoothness (OSS) algorithm iteration.

    Args:
        x: Complex tensor [N, C, H, W]
        magnitude_S: Spatial magnitude constraint
        magnitude_M: Fourier magnitude constraint
        support: Boolean mask in spatial domain
        beta: Relaxation parameter
        sigma: Initial Gaussian smoothing width
        filter_decay: Multiplicative decay per iteration
        iteration: Current iteration number
        projector_S: Spatial domain magnitude projector
        projector_M: Fourier domain magnitude projector

    Returns:
        Updated complex tensor after OSS iteration
    """
    # Step 1: Fourier magnitude projection
    x_new = projector_M(x, magnitude_M)

    # Step 2: Spatial magnitude + support projection
    x_new = projector_S(x_new, magnitude_S, support)

    # Step 3: Apply smoothing
    current_sigma = sigma * torch.pow(filter_decay, torch.tensor(iteration))
    x_new = _apply_gaussian_smoothing(x_new, sigma=current_sigma)

    # Step 4: Relaxation with relaxation parameter beta
    return x + beta * (x_new - x)


def _apply_gaussian_smoothing(
        x: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
    """
    Apply 2D Gaussian smoothing to each channel of a 4D complex tensor.
    """
    if sigma <= 0:
        return x  # No smoothing
    
    kernel_size = int(math.ceil(sigma * 6)) | 1  # Ensure odd size
    device = x.device
    gauss_kernel = _gaussian_kernel_2d(kernel_size, sigma, device=device)

    # Reshape for depthwise conv
    gauss_kernel = gauss_kernel.view(1, 1, kernel_size, kernel_size)
    gauss_kernel = gauss_kernel.expand(x.shape[1], 1, kernel_size, kernel_size)  # C groups

    # Apply real and imaginary parts separately
    x_real = F.conv2d(x.real, gauss_kernel, padding=kernel_size // 2, groups=x.shape[1])
    x_imag = F.conv2d(x.imag, gauss_kernel, padding=kernel_size // 2, groups=x.shape[1])
    
    return torch.complex(x_real, x_imag)


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(torch.square(xx) + torch.square(yy)) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def run(
        magnitudes: Tuple[torch.Tensor, torch.Tensor],
        support: Optional[torch.Tensor] = None,
        target_field: Optional[torch.Tensor] = None,
        init: Optional[torch.Tensor] = None,
        algorithm: Callable = None,
        algorithm_kwargs: Dict = {},
        max_iter: int = 100,
    ):
    with torch.no_grad():
        if algorithm is None:
            raise ValueError("You must provide an algorithm callable")

        if algorithm.__name__ == 'OSS':
            algorithm_kwargs['iteration'] = 0

        is_complex_target = target_field is not None and torch.is_complex(target_field)
        results = {
            'mse_fourier': [],
            'pearson_fourier': [],
            'pearson_fourier_intens': [],
        }
        if is_complex_target:
            results['quality'] = []
            results['quality_phi'] = []

        # Init guess
        x = initializers.flat_phases(magnitudes[0]) if init is None else init.clone()
        if support is None:
            support = torch.ones_like(magnitudes[0], dtype=torch.bool)

        for i in range(max_iter):
            x_tmp = transforms.fourier_transform(x)
            abs_x_tmp = torch.abs(x_tmp)
            
            # Metrics
            mse = metrics.mse(abs_x_tmp, magnitudes[1])
            results['mse_fourier'].append(mse.detach())

            results['pearson_fourier'].append(
                metrics.pearson(abs_x_tmp, magnitudes[1], inversed=True).detach()
            )
            results['pearson_fourier_intens'].append(
                metrics.pearson(torch.square(abs_x_tmp), torch.square(magnitudes[1]), inversed=True).detach()
            )

            if is_complex_target:
                x_supp = x[support]
                target_supp = target_field[support]
                results['quality'].append(
                    metrics.quality(x_supp, target_supp, inversed=True).detach()
                )
                results['quality_phi'].append(
                    metrics.quality(torch.exp(1j * torch.angle(x_supp)), torch.exp(1j * torch.angle(target_supp)), inversed=True).detach()
                )

            if algorithm.__name__ == 'OSS':
                algorithm_kwargs['iteration'] = i

            x = algorithm(
                x=x,
                magnitude_S=magnitudes[0],
                magnitude_M=magnitudes[1],
                support=support,
                **algorithm_kwargs,
            )

        # Stack metrics into tensors of shape [iterations, N, C]
        for key in ['mse_fourier', 'pearson_fourier', 'pearson_fourier_intens', 'quality', 'quality_phi']:
            if key in results and len(results[key]) > 0:
                results[key] = torch.stack(results[key], dim=0)

    return x, results



def report_convergence_results(results: dict, yscale: str = 'log', key: str = 'pearson_fourier_intens'):
    """Format is [iteration, N, C]"""

    plt.figure()
    for i in range(results[key].shape[1]):
        plt.plot(results[key][:, i, 0].cpu())
    plt.yscale(yscale)
    plt.title('Fourier intensity evolution (1 - PCC)')
    plt.ylabel('1 - PCC')
    plt.xlabel('Iteration')
    plt.grid(ls=':')
    # plt.legend()




def show_retrieved_fields(field, ft, x_hat, y_hat, window_crop: int = 60, power: float = 1, figsize: tuple[int, int] = (12, 7)):
    cropped = slice(window_crop, -window_crop) if window_crop >= 1 else slice(0, -1)
    pcc_fourier_intensity = pearson(
        x=np.square(np.abs(ft[cropped, cropped])),
        y=np.square(np.abs(y_hat[cropped, cropped])),
    )

    hfig = plt.figure(figsize=figsize)

    plt.suptitle(f"Phase retrieval results: Fourier intensity PCC = {pcc_fourier_intensity*100:3.4f} %\n(Brightnesses = |field|^{power})")
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.power(np.abs(field[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Measured NF: ' + r'$|x|$' + f'^{power}')

    plt.subplot(2, 3, 2)
    plt.imshow(np.power(np.abs(x_hat[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Retrieved NF: ' + r'$|\hat{x}|$' + f'^{power}')

    plt.subplot(2, 3, 3)
    plt.title('Retrieved NF: ' + r'$\hat{x}$')
    plt.imshow(complex_to_hsv(x_hat[cropped, cropped], rmin=0, rmax=np.max(np.power(np.abs(x_hat), power)), power=power))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(np.power(np.abs(ft[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Measured FF: ' + r'$|y|$' + f'^{power}')

    plt.subplot(2, 3, 5)
    plt.imshow(np.power(np.abs(y_hat[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Retrieved FF: ' + r'$|\hat{y}|$' + f'^{power}')

    plt.subplot(2, 3, 6)
    plt.imshow(complex_to_hsv(y_hat[cropped, cropped], rmin=0, rmax=np.max(np.power(np.abs(y_hat), power)), power=power))
    plt.axis('off')
    plt.title('Retrieved FF: ' + r'$\hat{y}$')

    return hfig


