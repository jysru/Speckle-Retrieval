import numpy as np
import matplotlib.pyplot as plt

import speckleret.transforms as transforms
import speckleret.metrics as metrics
import speckleret.plots as plots
import speckleret.supports as supports
import speckleret.initializers as inits

# Review paper: https://doi.org/10.1063/1.2403783
# Google collab: https://colab.research.google.com/drive/1anePjgg1fKbYrCCmDeRKblobryq-O4Rv


def P_S(x, magnitude, support, apply_support: bool = True):
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


def run(
        magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray = None,
        target_field: np.ndarray = None, init: np.ndarray = None,
        algorithm: callable = RAAR, algorithm_kwargs: tuple = None,
        max_iter: int = 100, rel_tol: float = 1e-3):

    if target_field is not None and np.iscomplexobj(target_field):
        results = {'mse_fourier': [], 'pearson_fourier': [], 'pearson_fourier_intens': [],'quality': [], 'quality_phi': []}
    else:
        results = {'mse_fourier': [], 'pearson_fourier': [], 'pearson_fourier_intens': []}
    
    x = inits.flat_phases(magnitude=np.abs(magnitudes[0])) if init is None else init.copy()
    support = np.ones(magnitudes[0].shape, dtype=bool) if support is None else support

    for i in range(max_iter):
        x_tmp = transforms.fourier_transform(x)
        results['mse_fourier'].append(metrics.mse(np.abs(x_tmp), np.abs(magnitudes[1])))
        results['pearson_fourier'].append(metrics.pearson(np.abs(x_tmp), np.abs(magnitudes[1]), inversed=True))
        results['pearson_fourier_intens'].append(metrics.pearson(np.square(np.abs(x_tmp)), np.square(np.abs(magnitudes[1])), inversed=True))
        if target_field is not None and np.iscomplexobj(target_field):
            results['quality'].append(metrics.quality(x[support], target_field[support], inversed=True))
            results['quality_phi'].append(metrics.quality(np.exp(1j * np.angle(x[support])), np.exp(1j * np.angle(target_field[support])), inversed=True))

        x = algorithm(x=x, magnitude_S=np.abs(magnitudes[0]), magnitude_M=np.abs(magnitudes[1]), support=support, **algorithm_kwargs)

        if i > 0:
            var_tol = (results['mse_fourier'][-1] - results['mse_fourier'][-2])/results['mse_fourier'][-2]
            if np.abs(var_tol) < rel_tol:
                break

    return x, results


def report_convergence_results(results: dict, yscale: str = 'log'):
    if 'quality' in results.keys():
        print(f"Qualities:\t Last = {1-results['quality'][-1]:3.5f},\t Last_phi = {1-results['quality_phi'][-1]:3.5f}")

    plt.figure()
    plt.plot(results['mse_fourier'], label='Fourier MSE')
    if 'quality' in results.keys():
        plt.plot(results['quality'], label='1 - Q')
        plt.plot(results['quality_phi'], label='1 - Qphi')
    plt.yscale(yscale)
    plt.title('Metrics evolution')
    plt.xlabel('Iteration')
    plt.legend()



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




