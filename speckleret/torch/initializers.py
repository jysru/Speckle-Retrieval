import torch


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
