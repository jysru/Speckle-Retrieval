import torch
import torch.nn as nn
import torch.optim as optim

from . import transforms
from . import metrics

import torchmetrics


def _default_ops(y_shape):
    """Default forward (FFT2) and adjoint (iFFT2) operators."""
    def A(x):
        return torch.fft.fft2(x)

    def At(z):
        return torch.fft.ifft2(z)

    return A, At


def flat_phases(tensor: torch.Tensor) -> torch.Tensor:
    """Applies a flat phase tensor to the initial tensor"""
    tensor = tensor * torch.exp(1j * _flat_phases_from_tensor(tensor))
    return tensor.type(torch.cfloat)


def random_phases(tensor: torch.Tensor) -> torch.Tensor:
    """Applies a flat phase tensor to the initial tensor"""
    tensor = tensor * torch.exp(1j * _random_phases_from_tensor(tensor))
    return tensor.type(torch.cfloat)


def _flat_phases_from_tensor(ref_tensor: torch.Tensor) -> torch.Tensor:
    """Returns a flat phase tensor with the size of the initial tensor"""
    return torch.zeros(size=ref_tensor.shape, dtype=torch.float, device=ref_tensor.device)


def _random_phases_from_tensor(ref_tensor: torch.Tensor) -> torch.Tensor:
    """Returns a random phase tensor (between -pi and +pi) with the size of the initial tensor"""
    return torch.pi * (
        2 * torch.rand(size=ref_tensor.shape, dtype=torch.float, device=ref_tensor.device) - 1
    )


    
def spectral(measured_magnitude, mask=None):
    """
    Spectral initialization for 2D Fourier magnitude measurements.
    
    Args:
        measured_magnitude: (H, W) tensor, measured |F{x}| (nonnegative real).
        mask: optional (H, W) boolean or float tensor, support mask in image domain.

    Returns:
        x0: (H, W) complex tensor, spectral initialization estimate.
    """
    device = measured_magnitude.device
    H, W = measured_magnitude.shape[-2:]
    print(f"Spectral init: H={H}, W={W}")
    
    # Step 1: squared intensity
    measured_magnitude = torch.squeeze(measured_magnitude)
    y2 = torch.square(measured_magnitude)
    
    # Step 2: inverse FFT of measured intensities (back-projection)
    # gives autocorrelation of the object
    autocorr = torch.fft.ifft2(y2)
    
    # Step 3: spectral vector = leading eigenvector of autocorrelation matrix
    # Approximation: just take the top singular vector of autocorr
    # (in practice, power iteration is enough)
    u, s, v = torch.linalg.svd(autocorr.real)  # treat as real matrix
    spectral_vec = u[:, 0]  # leading singular vector
    
    # Step 4: reshape + random phase
    x0 = spectral_vec.reshape(H, 1) @ torch.ones(1, W, device=device)
    x0 = x0.to(torch.complex64) * torch.exp(1j * 2 * torch.pi * torch.rand_like(x0))
    
    # Step 5: optional support mask
    if mask is not None:
        x0 = x0 * mask
    
    # Normalize scale
    scale = measured_magnitude.norm() / torch.fft.fft2(x0).abs().norm()
    x0 = x0 * scale
    
    x0 = x0.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
    
    return x0


import torch

def truncated_spectral(measured_magnitude: torch.Tensor, mask: torch.Tensor = None,
                       n_iter: int = 50, alpha: float = 0.25) -> torch.Tensor:
    """
    Truncated spectral initialization for 2D Fourier magnitude measurements.
    Ignores weak measurements to improve robustness to noise.

    Args:
        measured_magnitude: (H, W) real tensor, measured |F{x}|
        mask: optional (H, W) boolean/float tensor, support mask
        n_iter: number of power iterations
        alpha: fraction of weak measurements to discard (0.25 = drop lowest 25%)

    Returns:
        x0: (1, 1, H, W) complex tensor
    """
    device = measured_magnitude.device
    H, W = measured_magnitude.shape[-2:]
    measured_magnitude = measured_magnitude.squeeze()

    # Step 1: threshold
    thresh = torch.quantile(measured_magnitude, 1 - alpha)
    y_trunc = torch.where(measured_magnitude >= thresh,
                          measured_magnitude, torch.zeros_like(measured_magnitude))

    # Step 2: random phase assignment
    random_phase = torch.exp(1j * 2 * torch.pi * torch.rand_like(y_trunc))
    spectrum = y_trunc * random_phase

    # Step 3: inverse FFT
    x0 = torch.fft.ifft2(spectrum)

    # Step 4: support mask
    if mask is not None:
        x0 = x0 * mask

    # Step 5: normalize
    norm_factor = measured_magnitude.norm() / torch.fft.fft2(x0).abs().norm()
    x0 = x0 * norm_factor

    return x0.unsqueeze(0).unsqueeze(0).to(torch.complex64)


def orthogonal(measured_magnitude: torch.Tensor, mask: torch.Tensor = None,
               n_trials: int = 5) -> torch.Tensor:
    """
    Orthogonal initialization: try multiple random-phase initializations and
    pick the one with lowest measurement-domain error.

    Args:
        measured_magnitude: (H, W) real tensor, measured |F{x}|
        mask: optional (H, W) boolean/float tensor
        n_trials: number of random-phase candidates to test

    Returns:
        x0: (1, 1, H, W) complex tensor
    """
    device = measured_magnitude.device
    H, W = measured_magnitude.shape[-2:]
    measured_magnitude = measured_magnitude.squeeze()

    best_x, best_loss = None, 1e12

    for _ in range(n_trials):
        # random-phase init
        random_phase = torch.exp(1j * 2 * torch.pi * torch.rand_like(measured_magnitude))
        spectrum = measured_magnitude * random_phase
        x0 = torch.fft.ifft2(spectrum)

        if mask is not None:
            x0 = x0 * mask

        # compute error in Fourier domain
        est_mag = torch.fft.fft2(x0).abs()
        loss = torch.mean((est_mag - measured_magnitude) ** 2)

        if loss < best_loss:
            best_loss = loss
            best_x = x0

    # normalize
    norm_factor = measured_magnitude.norm() / torch.fft.fft2(best_x).abs().norm()
    best_x = best_x * norm_factor

    return best_x.unsqueeze(0).unsqueeze(0).to(torch.complex64)



# def spectral(measured_magnitude: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
#     """
#     Spectral initialization for 2D Fourier magnitude measurements.

#     Args:
#         measured_magnitude: (H, W) real tensor, measured |F{x}|
#         mask: optional (H, W) boolean/float tensor, support mask in image domain.

#     Returns:
#         x0: (1, 1, H, W) complex tensor, spectral initialization estimate.
#     """
#     device = measured_magnitude.device
#     H, W = measured_magnitude.shape[-2:]

#     # Ensure correct shape
#     measured_magnitude = measured_magnitude.squeeze()

#     # Step 1: assign random phases
#     random_phase = torch.exp(1j * 2 * torch.pi * torch.rand_like(measured_magnitude))
#     spectrum = measured_magnitude * random_phase

#     # Step 2: inverse FFT to image domain
#     x0 = torch.fft.ifft2(spectrum)

#     # Step 3: apply optional support mask
#     if mask is not None:
#         x0 = x0 * mask

#     # Step 4: normalize energy
#     norm_factor = measured_magnitude.norm() / torch.fft.fft2(x0).abs().norm()
#     x0 = x0 * norm_factor

#     # Add batch and channel dimensions
#     return x0.unsqueeze(0).unsqueeze(0).to(torch.complex64)


def power_method(y, A=None, At=None, n_iter: int = 50):
    """
    Power iteration for spectral initialization.

    Args:
        y: measurement magnitudes (tensor)
        A: forward operator (callable), defaults to FFT2
        At: adjoint operator (callable), defaults to iFFT2
        n_iter: number of power iterations

    Returns:
        v: leading eigenvector (complex tensor)
    """
    if A is None or At is None:
        A, At = _default_ops(y.shape)

    # random complex init
    v = torch.randn(y.shape, dtype=torch.complex64, device=y.device)

    for _ in range(n_iter):
        Av = A(v)
        v = At(y * Av)
        v = v / (torch.linalg.norm(v) + 1e-8)

    return v


def wirtinger_flow(y, A=None, At=None, n_iter: int = 200, step_size: float = 0.1):
    """
    Wirtinger Flow phase retrieval.

    Args:
        y: measurement magnitudes
        A: forward operator, defaults to FFT2
        At: adjoint operator, defaults to iFFT2
        n_iter: iterations
        step_size: gradient descent step size

    Returns:
        x: recovered signal
    """
    if A is None or At is None:
        A, At = _default_ops(y.shape)

    # spectral initialization
    x = power_method(y**2, A, At, n_iter=50)

    for _ in range(n_iter):
        Ax = A(x)
        residual = (torch.abs(Ax)**2 - y**2) * Ax
        grad = At(residual)
        x = x - step_size * grad

    return x


def gao_xu(y, A=None, At=None, n_iter: int = 200, step_size: float = 0.1, tau: float = 3.0):
    """
    Gao–Xu / Truncated Amplitude Flow.

    Args:
        y: measurement magnitudes
        A: forward operator, defaults to FFT2
        At: adjoint operator, defaults to iFFT2
        n_iter: iterations
        step_size: learning rate
        tau: truncation threshold factor

    Returns:
        x: recovered signal
    """
    if A is None or At is None:
        A, At = _default_ops(y.shape)

    # spectral initialization
    x = power_method(y**2, A, At, n_iter=50)

    for _ in range(n_iter):
        Ax = A(x)
        residual = (torch.abs(Ax)**2 - y**2) * Ax

        # truncation
        mask = torch.abs(residual) < tau * y
        residual = residual * mask

        grad = At(residual)
        x = x - step_size * grad

    return x


def gradient_descent(
        magnitude_near_field,
        magnitude_far_field,
        support,
        init_phase: torch.Tensor = None,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = dict(lr = 1e-1),
        max_iter: int = 50,
        cyclic_loss: bool = False,
        return_loss: bool = False,
        ):
    if init_phase is None:
        init_phase = 2 * torch.pi * torch.rand_like(torch.abs(magnitude_near_field)) * 0

    model = PhaseOnlyOptimizer(
        init_magnitude=torch.abs(magnitude_near_field),
        init_phase=init_phase,
        support=support,
    )
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.01, patience=10, threshold_mode='rel')

    _loss = []
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = model.loss(
            target_fourier_magnitude=torch.abs(magnitude_far_field),
            target_magnitude=torch.abs(magnitude_near_field) if cyclic_loss else None,
        )
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if i % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            # print(f"Iter {i}, Loss: {loss.item()}, LR: {lr}")
            print(f"Iteration {i}, Loss: {loss.item()}, LR: {lr:1.2e}")
        _loss.append(loss.item())

    x_out = torch.abs(magnitude_near_field) * torch.exp(1j * model.phase)
    
    if return_loss:
        return x_out.detach(), _loss
    else:
        return x_out.detach()



class PhaseOnlyOptimizer(nn.Module):
    def __init__(
            self,
            init_magnitude: torch.Tensor,   # [N,C,H,W], fixed magnitude
            init_phase: torch.Tensor,       # [N,C,H,W], initial phase guess
            support: torch.Tensor = None    # [N,C,H,W], optional support mask
        ):
        super().__init__()
        self.register_buffer('magnitude', init_magnitude)
        self.register_buffer('support', support if support is not None else torch.ones_like(init_magnitude, dtype=torch.bool))

        # Learnable phase parameter, initialized with init_phase
        self.phase = nn.Parameter(init_phase.clone())
        self.loss_fn = PearsonImageLoss(average_over_channels=True, average_over_batches=True, invert=True, power=1)

    def forward(self):
        # Compose complex field: fixed magnitude * learned phase
        x = self.magnitude * torch.exp(1j * self.phase)
        # Apply support mask if provided
        x = torch.where(self.support, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        return x

    def loss(self, target_fourier_magnitude: torch.Tensor, target_magnitude: torch.Tensor = None):
        x = self.forward()
        x_fft = transforms.fourier_transform(x)

        # Use Pearson correlation loss between squared magnitudes
        pearson_loss_ff = self.loss_fn(x_fft, target_fourier_magnitude)
        _loss = pearson_loss_ff
        
        # pearson_loss_ff = torch.mean(metrics.pearson(torch.square(x_fft_mag), torch.square(target_fourier_magnitude), inversed=True))
        
        
        if target_magnitude is not None:
            x_inv = target_fourier_magnitude * torch.exp(1j * torch.angle(x_fft))
            x_inv = transforms.inverse_fourier_transform(x_fft)
            pearson_loss_nf = self.loss_fn(x_inv, target_magnitude)
            _loss = 0.5 * _loss + 0.5 * pearson_loss_nf
            
        # ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=target_fourier_magnitude.max() - target_fourier_magnitude.min()).to(x.device)
        # ssim = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(data_range=target_fourier_magnitude.max() - target_fourier_magnitude.min()).to(x.device)
        # ssim = torchmetrics.image.UniversalImageQualityIndex(data_range=target_fourier_magnitude.max() - target_fourier_magnitude.min()).to(x.device)

        return _loss
    
    



class PearsonImageLoss(torch.nn.Module):

    def __init__(self, average_over_channels: bool = True, average_over_batches: bool = True, invert = True, power: int = 1):
        super(PearsonImageLoss, self).__init__()
        self.average_over_channels = average_over_channels
        self.average_over_batches = average_over_batches
        self.invert = invert
        self.power = power
        self.pearson = batch_pearson_images

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pearson_value = self.pearson(
            torch.pow(torch.abs(x), self.power), torch.pow(torch.abs(y), self.power),
            average_over_channels=self.average_over_channels,
            average_over_batches=self.average_over_batches,
            invert=self.invert,
        )
        return pearson_value
    


def batch_pearson_images(x: torch.Tensor, y: torch.Tensor, average_over_channels: bool = True, average_over_batches: bool = True, invert: bool = False) -> torch.Tensor:
    """
    Compute batch-wise Pearson Correlation Coefficient between tensors x and y.
    Shape: (N, C, H, W)
    """
    # Ensure the shapes match
    assert x.shape == y.shape, "Input tensors must have the same shape"

    x = torch.abs(x)
    y = torch.abs(y)

    # Mean normalization
    x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
    y_mean = torch.mean(y, dim=(2, 3), keepdim=True)
    x_std = torch.std(x, dim=(2, 3), keepdim=True)
    y_std = torch.std(y, dim=(2, 3), keepdim=True)
    
    # Compute covariance and standard deviations
    cov = torch.mean((x - x_mean) * (y - y_mean), dim=(2, 3), keepdim=True)
        
    # Compute Pearson correlation, avoid division by zero
    corr = cov / (x_std * y_std + 1e-8)

    # Average over channels, and then over batch
    if average_over_channels:
        corr = torch.mean(corr, dim=1, keepdim=True)
    if average_over_batches:
        corr = torch.mean(corr)

    if invert:
        corr = 1 - corr

    return corr