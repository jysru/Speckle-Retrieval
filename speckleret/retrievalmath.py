import numpy as np
import matplotlib.pyplot as plt

import speckleret.transforms as transforms
import speckleret.metrics as metrics
import speckleret.plots as plots
import speckleret.supports as supports

# Review paper: https://doi.org/10.1063/1.2403783
# Google collab: https://colab.research.google.com/drive/1anePjgg1fKbYrCCmDeRKblobryq-O4Rv


def P_S(x, magnitude, support, apply_support: bool = False):
    """Support projection"""
    x_new = magnitude * np.exp(1j * np.angle(x))
    if apply_support:
        x_new[np.logical_not(support)] = 0
    return x_new


def P_M(x, magnitude):
    """Fourier magnitude projection"""
    X = transforms.fourier_transform(x)
    X = magnitude * np.exp(1j * np.angle(X))
    return transforms.inverse_fourier_transform(X)


def R_M(x, gamma_M, magnitude):
    """Magnitude reflector"""
    return (1 + gamma_M) * P_M(x, magnitude) - gamma_M * x


def R_S(x, gamma_S, magnitude, support):
    """Support reflector"""
    return (1 + gamma_S) * P_S(x, magnitude, support) - gamma_S * x


def ER(x, magnitude_S, magnitude_M, support):
    """Error Reduction algorithm iteration"""
    return P_S(P_M(x, magnitude_M), magnitude_S, support)


def SF(x, magnitude_S, magnitude_M, support):
    """Solvent Flipping algorithm iteration"""
    return R_S(P_M(x, magnitude_M), 1, magnitude_S, support)


def HIO(x, magnitude_S, magnitude_M, support, beta: float = 0.7):
    """Hybrid Input-Output algorithm iteration"""
    x_new = x.copy()
    proj = P_M(x, magnitude_M)
    x_new[support] = proj[support]
    x_new[np.logical_not(support)] = x_new[np.logical_not(support)] - beta * P_S(proj, magnitude_S, support)[np.logical_not(support)]
    return x_new


def DM(x, magnitude_S, magnitude_M, support, beta: float = 0.7, gamma_S: float = None, gamma_M: float = None):
    """Difference Maps algorithm iteration"""
    # Random projections and the optimization of an algorithm for phase retrieval, J. Phys. A-Math Gen, Vol 36 pp 2995-3007
    if gamma_S is None:
        gamma_S = 1 / beta
    if gamma_M is None:
        gamma_M = -1 / beta
    
    x_PMRS = P_M(R_S(x, gamma_S, magnitude_S, support), magnitude_M)
    x_PSRM = P_S(R_M(x, gamma_M, magnitude_M), magnitude_S, support)
    x_new = x + beta * (x_PMRS - x_PSRM)
    return x_new


def ASR(x, magnitude_S, magnitude_M, support):
    """Averaged succesive reflections algorithm iteration"""
    return 0.5 * (R_S(R_M(x, 1, magnitude_M), 1, magnitude_S, support) + x)


def HPR(x, magnitude_S, magnitude_M, support, beta: float = 0.7):
    """Hybrid projection reflection algorithm iteration"""
    proj = P_M(x, magnitude_M)
    x1 = R_S(R_M(x, 1, magnitude_M) + (beta - 1) * proj, 1, magnitude_S, support)
    x2 = (1 - beta) * proj 
    return 0.5 * (x1 + x + x2)


def RAAR(x, magnitude_S, magnitude_M, support, beta: float = 0.7):
    """Relaxed Averaged Alternating Reflectors algorithm iteration"""
    proj = P_M(x, magnitude_M)
    return beta * ASR(x, magnitude_S, magnitude_M, support)  +  (1 - beta) * proj



if __name__ == "__main__":
    sz = 50
    X, Y, R, _ = supports.pixels_meshgrids(sz)
    field = np.random.randn(sz, sz) + 1j * np.random.randn(sz, sz)
    
    field = np.ones(field.shape) * np.exp(-np.square(R/15))
    field = field * np.exp(1j * (np.power(X/10, 3) + np.power(Y/10, 3)))
    field = transforms.pad_img(field, pad=2)

    ft = transforms.fourier_transform(field, pad=None)
    # ft = transforms.fresnel_transform(field, dz=20e-6, pad=None)
    # support = supports.disk_support(field, radius=40)
    # support = supports.disk_support(field, radius=40)
    support = supports.threshold_support(field, threshold=0.01)

    # y_hat, ft_hat, results = error_reduction_fourier((np.abs(field), np.abs(ft)), support=support, pad=2, max_iter=500)
    # y_hat, ft_hat, results = hio_fourier((np.abs(field), np.abs(ft)), support, pad=None, max_iter=500, beta=0.8)
    y = np.random.randn(*field.shape) + 1j * np.random.randn(*field.shape)
    for i in range(100):
        y_hat = SF(y, np.abs(field), np.abs(ft), support)
    print(metrics.quality(y_hat[support], field[support]))

    plots.compare_complex_fields(field * support, y_hat * support)
    plt.show()
