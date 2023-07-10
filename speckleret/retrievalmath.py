import numpy as np
import matplotlib.pyplot as plt

import speckleret.transforms as transforms
import speckleret.metrics as metrics
import speckleret.plots as plots
import speckleret.supports as supports
import speckleret.initializers as inits

# Review paper: https://doi.org/10.1063/1.2403783
# Google collab: https://colab.research.google.com/drive/1anePjgg1fKbYrCCmDeRKblobryq-O4Rv


def P_S(
    x: np.ndarray,
    magnitude: np.ndarray,
    support: np.ndarray,
    apply_support: bool = True,
    ):
    """Support projection"""
    x_new = magnitude * np.exp(1j * np.angle(x))
    if apply_support:
        x_new[np.logical_not(support)] = 0
    return x_new


def P_M(
    x: np.ndarray,
    magnitude: np.ndarray,
    direct_transform: callable = transforms.fourier_transform,
    inverse_transform: callable = transforms.inverse_fourier_transform,
    ):
    """Fourier magnitude projection"""
    X = direct_transform(x)
    X = magnitude * np.exp(1j * np.angle(X))
    return inverse_transform(X)


def R_M(
    x: np.ndarray,
    gamma_M: float,
    magnitude: np.ndarray,
    projector_M: callable = P_M,
    ):
    """Magnitude reflector"""
    return (1 + gamma_M) * projector_M(x, magnitude) - gamma_M * x


def R_S(
    x: np.ndarray,
    gamma_S: float,
    magnitude: np.ndarray,
    support: np.ndarray,
    projector_S: callable = P_S,
    ):
    """Support reflector"""
    return (1 + gamma_S) * projector_S(x, magnitude, support) - gamma_S * x


def ER(
    x: np.ndarray,
    magnitude_S: np.ndarray,
    magnitude_M: np.ndarray,
    support: np.ndarray,
    projector_S: callable = P_S,
    projector_M: callable = P_M,
    ):
    """Error Reduction algorithm iteration"""
    return projector_S(projector_M(x, magnitude_M), magnitude_S, support)


def SF(
    x: np.ndarray,
    magnitude_S: np.ndarray,
    magnitude_M: np.ndarray,
    support: np.ndarray,
    reflector_S: callable = R_S,
    projector_M: callable = P_M,
    ):
    """Solvent Flipping algorithm iteration"""
    return reflector_S(projector_M(x, magnitude_M), 1, magnitude_S, support)


def HIO(
    x: np.ndarray,
    magnitude_S: np.ndarray,
    magnitude_M: np.ndarray,
    support: np.ndarray,
    beta: float = 0.7,
    projector_S: callable = P_S,
    projector_M: callable = P_M,
    ):
    """Hybrid Input-Output algorithm iteration"""
    x_new = x.copy()
    proj = projector_M(x, magnitude_M)
    x_new[support] = proj[support]
    x_new[np.logical_not(support)] = x_new[np.logical_not(support)] - beta * projector_S(proj, magnitude_S, support)[np.logical_not(support)]
    return x_new


def DM(
    x: np.ndarray,
    magnitude_S: np.ndarray,
    magnitude_M: np.ndarray,
    support: np.ndarray,
    beta: float = 0.7,
    gamma_S: float = None,
    gamma_M: float = None,
    projector_S: callable = P_S,
    projector_M: callable = P_M,
    reflector_S: callable = R_S,
    reflector_M: callable = R_M,
    ):
    """Difference Maps algorithm iteration"""
    # Random projections and the optimization of an algorithm for phase retrieval, J. Phys. A-Math Gen, Vol 36 pp 2995-3007
    if gamma_S is None:
        gamma_S = 1 / beta
    if gamma_M is None:
        gamma_M = -1 / beta
    
    x_PMRS = projector_M(reflector_S(x, gamma_S, magnitude_S, support), magnitude_M)
    x_PSRM = projector_S(reflector_M(x, gamma_M, magnitude_M), magnitude_S, support)
    x_new = x + beta * (x_PMRS - x_PSRM)
    return x_new


def ASR(
    x: np.ndarray,
    magnitude_S: np.ndarray,
    magnitude_M: np.ndarray,
    support: np.ndarray,
    reflector_S: callable = R_S,
    reflector_M: callable = R_M,
    ):
    """Averaged succesive reflections algorithm iteration"""
    return 0.5 * (reflector_S(reflector_M(x, 1, magnitude_M), 1, magnitude_S, support) + x)


def HPR(
    x: np.ndarray,
    magnitude_S: np.ndarray,
    magnitude_M: np.ndarray,
    support: np.ndarray,
    beta: float = 0.7,
    projector_M: callable = P_M,
    reflector_S: callable = R_S,
    reflector_M: callable = R_M,
    ):
    """Hybrid projection reflection algorithm iteration"""
    proj = projector_M(x, magnitude_M)
    x1 = reflector_S(reflector_M(x, 1, magnitude_M) + (beta - 1) * proj, 1, magnitude_S, support)
    x2 = (1 - beta) * proj 
    return 0.5 * (x1 + x + x2)


def RAAR(
    x: np.ndarray,
    magnitude_S: np.ndarray,
    magnitude_M: np.ndarray,
    support: np.ndarray,
    beta: float = 0.7,
    projector_M: callable = P_M,
    ):
    """Relaxed Averaged Alternating Reflectors algorithm iteration"""
    proj = projector_M(x, magnitude_M)
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


class AltProjPR():
    default_direct_transform: callable = transforms.fourier_transform
    default_inverse_transform: callable = transforms.inverse_fourier_transform
    apply_planes_magnitudes: list[bool, bool] = [True, True]
    apply_planes_supports: list[bool, bool] = [True, True]

    def __init__(
        self,
        magnitudes: list[np.ndarray, np.ndarray],
        targets: list[np.ndarray, np.ndarray] = None,
        supports: list[np.ndarray, np.ndarray] = None,
        direct_transform: callable = AltProjPR.default_direct_transform,
        inverse_transform: callable = AltProjPR.default_inverse_transform,
        apply_planes_magnitudes: list[bool, bool] = [AltProjPR.apply_planes_magnitudes]
        apply_planes_supports: list[bool, bool] = [AltProjPR.apply_planes_supports]
        ) -> None:

        self._direct_transform: callable = None
        self._inverse_transform: callable = None
        self._magnitudes: list[np.ndarray, np.ndarray] = None
        self._targets: list[np.ndarray, np.ndarray] = None
        self._supports: list[np.ndarray, np.ndarray] = None
        self._init: np.ndarray = None
        self._algorithm: callable = None
        self._algorithm_kwargs: dict = None
        self._max_iter: int = None
        self._rel_tol: float = None
        self._apply_planes_magnitudes: list[bool, bool] = apply_planes_magnitudes
        self._apply_planes_supports: bool = apply_planes_supports

        self._set_transforms(direct_transform, inverse_transform)
        self._set_magnitudes(magnitudes)

    def _set_transforms(self, direct_transform: callable, inverse_transform: callable) -> None:
        are_valid = self._check_transforms(direct_transform, inverse_transform)
        if are_valid:
            self._direct_transform = direct_transform
            self._inverse_transform = inverse_transform
        else:
            raise TypeError("Transforms must be of callable type!")            

    def _check_transforms(self, *args) -> bool:
        is_valid: bool = True
        for i in range(args):
            is_valid = is_valid & callable(args[i])
        return is_valid

    def _set_magnitudes(self, magnitudes[i]: list[np.ndarray, np.ndarray]) -> None:
        self._magnitudes = self._check_magnitudes(magnitudes)

    def _check_magnitudes(self, magnitudes: list[np.ndarray, np.ndarray]) -> None:
        for i in range(len(magnitudes)):
            if np.iscomplexobj(magnitudes[i]):
                magnitudes[i] = np.abs(magnitudes[i])
        return magnitudes

    def _apply_magnitude(self, x: np.ndarray, idx: int) -> np.ndarray
        if self.magnitudes is not None:
            x = self._magnitudes[idx] * np.exp(1j * np.angle(x))
        return x

    def _apply_support(self, x: np.ndarray, idx: int) -> np.ndarray
        if self._supports is not None:
            x[np.logical_not(self._supports[idx])] = 0
        return x

    def _proj_s(self, x: np.ndarray) -> np.ndarray:
        """Plane1: Object plane projection and support application"""
        x = self._apply_magnitude(x, idx=0) if self._apply_planes_magnitude[0] else x
        return self._apply_support(x, idx=0) if self._apply_planes_supports[0] else x

    def _proj_m(self, x: np.ndarray) -> np.ndarray:
        """Plane2: Fourier magnitude projection and support application"""
        x = self._direct_transform(x)
        x = self._apply_magnitude(x, idx=1) if self._apply_planes_magnitude[1] else x
        x = self._apply_support(x, idx=1) if self._apply_planes_supports[1] else x
        return self._inverse_transform(x)

    def _refl_s(self, x: np.ndarray, g: float) -> np.ndarray:
        """Plane1: Object plane reflection and support application"""
        return (1 + g) * self._proj_s(x) - g * x

    def _refl_m(self, x: np.ndarray, g: float) -> np.ndarray:
        """Plane1: Object plane reflection and support application"""
        return (1 + g) * self._proj_m(x) - g * x

    def _iter_ER(self, x: np.ndarray) -> np.ndarray:
        """Error Reduction algorithm iteration"""
        return self._proj_s(self._proj_m(x))

    def _iter_SF(self, x: np.ndarray) -> np.ndarray:
        """Solvent Flipping algorithm iteration"""
        return self._refl_s(self._proj_m(x), g=1)

    def _iter_HIO(self, x: np.ndarray, beta: float = 0.7) -> np.ndarray:
        """Hybrid Input-Output algorithm iteration"""
        x = x.copy()
        x_proj = self._proj_m(x)
        x[self._supports[0]] = proj[self._supports[0]]
        x[np.logical_not(self._supports[0])] = x[np.logical_not(self._supports[0])] - beta * self._proj_s(x_proj)[np.logical_not(self._supports[0])]
        return x

    def _iter_ASR(self, x: np.ndarray) -> np.ndarray:
        """Averaged succesive reflections algorithm iteration"""
        return 0.5 * (self._refl_s(self._refl_m(x, g=1), g=1) + x)

    def _iter_HPR(self, x: np.ndarray, beta: float = 0.7) -> np.ndarray:
        """Hybrid projection reflection algorithm iteration"""
        x_proj = self._proj_m(x)
        x1 = self._refl_s(self._refl_m(x, g=1) + (beta - 1) * x_proj, g=1)
        x2 = (1 - beta) * x_proj
        return 0.5 * (x1 + x_proj + x2)

    def _iter_DM(self, x: np.ndarray, beta: float = 0.7, gs: float = None, gm: float = None) -> np.ndarray:
        """Difference Maps algorithm iteration"""
        # Random projections and the optimization of an algorithm for phase retrieval, J. Phys. A-Math Gen, Vol 36 pp 2995-3007
        if gs is None:
            gs = 1 / beta
        if gm is None:
            gm = -1 / beta
        
        x_pmrs = self._proj_m(self._refl_s(x, g=gs))
        x_psrm = self._proj_s(self._refl_m(x, g=gm))
        return x + beta * (x_pmrs - x_psrm)

    def _iter_RAAR(self, x: np.ndarray, beta: float = 0.7) -> np.ndarray:
        """Relaxed Averaged Alternating Reflectors algorithm iteration"""
        return beta * _iter_ASR(x) + (1 - beta) * self._proj_m(x)   

    def run(
        self,
        algorithm: callable,
        algorithm_kwargs: dict = None,
        init: np.ndarray = None,
        max_iter: int = 100,
        rel_tol: float = 1e-3,
        ) -> None:

        x, results = spr.retrievalmath.run(
            magnitudes=(np.abs(field), np.abs(ft)),
            support=support,
            init=spr.initializers.random_phases(magnitude=np.abs(field)),
            algorithm=spr.retrievalmath.RAAR,
            algorithm_kwargs={'beta': 0.7},
            max_iter=100,
            rel_tol=1e-6,
            )

    def define_support(self) -> None:
        pass

    def show(self) -> None:
        pass





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




