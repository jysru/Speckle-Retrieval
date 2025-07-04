import torch
import torch.nn as nn
import torch.optim as optim

from . import transforms
from . import metrics


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


def power_method(*args, **kwargs):
    raise NotImplementedError


def wirtinger_flow(*args, **kwargs):
    raise NotImplementedError


def gao_xu(*args, **kwargs):
    raise NotImplementedError


def gradient_descent(
        magnitude_near_field,
        magnitude_far_field,
        support,
        init_phase: torch.Tensor = None,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = dict(lr = 1e-1),
        max_iter: int = 50,
        ):
    if init_phase is None:
        init_phase = 2 * torch.pi * torch.rand_like(torch.abs(magnitude_near_field))

    model = PhaseOnlyOptimizer(
        init_magnitude=torch.abs(magnitude_near_field),
        init_phase=init_phase,
        support=support,
    )
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = model.loss(torch.abs(magnitude_far_field))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

    x_out = torch.abs(magnitude_near_field) * torch.exp(1j * model.phase)
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

    def forward(self):
        # Compose complex field: fixed magnitude * learned phase
        x = self.magnitude * torch.exp(1j * self.phase)
        # Apply support mask if provided
        x = torch.where(self.support, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        return x

    def loss(self, target_fourier_magnitude: torch.Tensor):
        x = self.forward()
        x_fft = transforms.fourier_transform(x)
        x_fft_mag = torch.abs(x_fft)

        pearson_loss = torch.mean(metrics.pearson(x_fft_mag, target_fourier_magnitude, inversed=True))

        return pearson_loss
    
