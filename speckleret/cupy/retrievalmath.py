import numpy as cp
import matplotlib.pyplot as plt

import speckleret as spr
import speckleret.cupy.transforms as transforms
import speckleret.cupy.metrics as metrics
import speckleret.cupy.initializers as inits

# Review paper: https://doi.org/10.1063/1.2403783
# Google collab: https://colab.research.google.com/drive/1anePjgg1fKbYrCCmDeRKblobryq-O4Rv


def P_S(
    x: cp.ndarray,
    magnitude: cp.ndarray,
    support: cp.ndarray,
    apply_support: bool = True,
    ):
    """Support projection"""
    x_new = magnitude * cp.exp(1j * cp.angle(x))
    if apply_support:
        x_new[cp.logical_not(support)] = 0
    return x_new


def P_M(
    x: cp.ndarray,
    magnitude: cp.ndarray,
    direct_transform: callable = transforms.fourier_transform,
    inverse_transform: callable = transforms.inverse_fourier_transform,
    ):
    """Fourier magnitude projection"""
    X = direct_transform(x)
    X = magnitude * cp.exp(1j * cp.angle(X))
    return inverse_transform(X)


def R_M(
    x: cp.ndarray,
    gamma_M: float,
    magnitude: cp.ndarray,
    projector_M: callable = P_M,
    ):
    """Magnitude reflector"""
    return (1 + gamma_M) * projector_M(x, magnitude) - gamma_M * x


def R_S(
    x: cp.ndarray,
    gamma_S: float,
    magnitude: cp.ndarray,
    support: cp.ndarray,
    projector_S: callable = P_S,
    ):
    """Support reflector"""
    return (1 + gamma_S) * projector_S(x, magnitude, support) - gamma_S * x


def ER(
    x: cp.ndarray,
    magnitude_S: cp.ndarray,
    magnitude_M: cp.ndarray,
    support: cp.ndarray,
    projector_S: callable = P_S,
    projector_M: callable = P_M,
    ):
    """Error Reduction algorithm iteration"""
    return projector_S(projector_M(x, magnitude_M), magnitude_S, support)


def SF(
    x: cp.ndarray,
    magnitude_S: cp.ndarray,
    magnitude_M: cp.ndarray,
    support: cp.ndarray,
    reflector_S: callable = R_S,
    projector_M: callable = P_M,
    ):
    """Solvent Flipping algorithm iteration"""
    return reflector_S(projector_M(x, magnitude_M), 1, magnitude_S, support)


def HIO(
    x: cp.ndarray,
    magnitude_S: cp.ndarray,
    magnitude_M: cp.ndarray,
    support: cp.ndarray,
    beta: float = 0.7,
    projector_S: callable = P_S,
    projector_M: callable = P_M,
    ):
    """Hybrid Icput-Output algorithm iteration"""
    x_new = x.copy()
    proj = projector_M(x, magnitude_M)
    x_new[support] = proj[support]
    x_new[cp.logical_not(support)] = x_new[cp.logical_not(support)] - beta * projector_S(proj, magnitude_S, support)[cp.logical_not(support)]
    return x_new


def DM(
    x: cp.ndarray,
    magnitude_S: cp.ndarray,
    magnitude_M: cp.ndarray,
    support: cp.ndarray,
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
    x: cp.ndarray,
    magnitude_S: cp.ndarray,
    magnitude_M: cp.ndarray,
    support: cp.ndarray,
    reflector_S: callable = R_S,
    reflector_M: callable = R_M,
    ):
    """Averaged succesive reflections algorithm iteration"""
    return 0.5 * (reflector_S(reflector_M(x, 1, magnitude_M), 1, magnitude_S, support) + x)


def HPR(
    x: cp.ndarray,
    magnitude_S: cp.ndarray,
    magnitude_M: cp.ndarray,
    support: cp.ndarray,
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
    x: cp.ndarray,
    magnitude_S: cp.ndarray,
    magnitude_M: cp.ndarray,
    support: cp.ndarray,
    beta: float = 0.7,
    projector_M: callable = P_M,
    ):
    """Relaxed Averaged Alternating Reflectors algorithm iteration"""
    proj = projector_M(x, magnitude_M)
    return beta * ASR(x, magnitude_S, magnitude_M, support)  +  (1 - beta) * proj


def run(
        magnitudes: tuple[cp.ndarray, cp.ndarray], support: cp.ndarray = None,
        target_field: cp.ndarray = None, init: cp.ndarray = None,
        algorithm: callable = RAAR, algorithm_kwargs: tuple = None,
        max_iter: int = 100, rel_tol: float = 1e-3):

    if target_field is not None and cp.iscomplexobj(target_field):
        results = {'mse_fourier': [], 'pearson_fourier': [], 'pearson_fourier_intens': [],'quality': [], 'quality_phi': []}
    else:
        results = {'mse_fourier': [], 'pearson_fourier': [], 'pearson_fourier_intens': []}
    
    x = inits.flat_phases(magnitude=cp.abs(magnitudes[0])) if init is None else init.copy()
    support = cp.ones(magnitudes[0].shape, dtype=bool) if support is None else support

    for i in range(max_iter):
        x_tmp = transforms.fourier_transform(x)
        results['mse_fourier'].append(metrics.mse(cp.abs(x_tmp), cp.abs(magnitudes[1])))
        results['pearson_fourier'].append(metrics.pearson(cp.abs(x_tmp), cp.abs(magnitudes[1]), inversed=True))
        results['pearson_fourier_intens'].append(metrics.pearson(cp.square(cp.abs(x_tmp)), cp.square(cp.abs(magnitudes[1])), inversed=True))
        if target_field is not None and cp.iscomplexobj(target_field):
            results['quality'].append(metrics.quality(x[support], target_field[support], inversed=True))
            results['quality_phi'].append(metrics.quality(cp.exp(1j * cp.angle(x[support])), cp.exp(1j * cp.angle(target_field[support])), inversed=True))

        x = algorithm(x=x, magnitude_S=cp.abs(magnitudes[0]), magnitude_M=cp.abs(magnitudes[1]), support=support, **algorithm_kwargs)

        if i > 0:
            var_tol = (results['mse_fourier'][-1] - results['mse_fourier'][-2])/results['mse_fourier'][-2]
            if cp.abs(var_tol) < rel_tol:
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
        magnitudes: list[cp.ndarray, cp.ndarray],
        targets: list[cp.ndarray, cp.ndarray] = None,
        supports: list[cp.ndarray, cp.ndarray] = None,
        direct_transform: callable = None,
        inverse_transform: callable = None,
        apply_planes_magnitudes: list[bool, bool] = None,
        apply_planes_supports: list[bool, bool] = None,
        ) -> None:

        self._direct_transform: callable = AltProjPR.default_direct_transform
        self._inverse_transform: callable = AltProjPR.default_inverse_transform
        self._magnitudes: list[cp.ndarray, cp.ndarray] = None
        self._targets: list[cp.ndarray, cp.ndarray] = None
        self._supports: list[cp.ndarray, cp.ndarray] = None
        self._init: cp.ndarray = None
        self._algorithm: callable = None
        self._algorithm_kwargs: dict = None
        self._max_iter: int = None
        self._rel_tol: float = None
        self._apply_planes_magnitudes: list[bool, bool] = AltProjPR.apply_planes_magnitudes
        self._apply_planes_supports: bool = AltProjPR.apply_planes_supports

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

    def _set_magnitudes(self, magnitudes: list[cp.ndarray, cp.ndarray]) -> None:
        self._magnitudes = self._check_magnitudes(magnitudes)

    def _check_magnitudes(self, magnitudes: list[cp.ndarray, cp.ndarray]) -> None:
        for i in range(len(magnitudes)):
            if cp.iscomplexobj(magnitudes[i]):
                magnitudes[i] = cp.abs(magnitudes[i])
        return magnitudes

    def _apply_magnitude(self, x: cp.ndarray, idx: int) -> cp.ndarray:
        if self.magnitudes is not None:
            x = self._magnitudes[idx] * cp.exp(1j * cp.angle(x))
        return x

    def _apply_support(self, x: cp.ndarray, idx: int) -> cp.ndarray:
        if self._supports is not None:
            x[cp.logical_not(self._supports[idx])] = 0
        return x

    def _proj_s(self, x: cp.ndarray) -> cp.ndarray:
        """Plane1: Object plane projection and support application"""
        x = self._apply_magnitude(x, idx=0) if self._apply_planes_magnitude[0] else x
        return self._apply_support(x, idx=0) if self._apply_planes_supports[0] else x

    def _proj_m(self, x: cp.ndarray) -> cp.ndarray:
        """Plane2: Fourier magnitude projection and support application"""
        x = self._direct_transform(x)
        x = self._apply_magnitude(x, idx=1) if self._apply_planes_magnitude[1] else x
        x = self._apply_support(x, idx=1) if self._apply_planes_supports[1] else x
        return self._inverse_transform(x)

    def _refl_s(self, x: cp.ndarray, g: float) -> cp.ndarray:
        """Plane1: Object plane reflection and support application"""
        return (1 + g) * self._proj_s(x) - g * x

    def _refl_m(self, x: cp.ndarray, g: float) -> cp.ndarray:
        """Plane1: Object plane reflection and support application"""
        return (1 + g) * self._proj_m(x) - g * x

    def _iter_ER(self, x: cp.ndarray) -> cp.ndarray:
        """Error Reduction algorithm iteration"""
        return self._proj_s(self._proj_m(x))

    def _iter_SF(self, x: cp.ndarray) -> cp.ndarray:
        """Solvent Flipping algorithm iteration"""
        return self._refl_s(self._proj_m(x), g=1)

    def _iter_HIO(self, x: cp.ndarray, beta: float = 0.7) -> cp.ndarray:
        """Hybrid Icput-Output algorithm iteration"""
        x = x.copy()
        x_proj = self._proj_m(x)
        x[self._supports[0]] = x_proj[self._supports[0]]
        x[cp.logical_not(self._supports[0])] = x[cp.logical_not(self._supports[0])] - beta * self._proj_s(x_proj)[cp.logical_not(self._supports[0])]
        return x

    def _iter_ASR(self, x: cp.ndarray) -> cp.ndarray:
        """Averaged succesive reflections algorithm iteration"""
        return 0.5 * (self._refl_s(self._refl_m(x, g=1), g=1) + x)

    def _iter_HPR(self, x: cp.ndarray, beta: float = 0.7) -> cp.ndarray:
        """Hybrid projection reflection algorithm iteration"""
        x_proj = self._proj_m(x)
        x1 = self._refl_s(self._refl_m(x, g=1) + (beta - 1) * x_proj, g=1)
        x2 = (1 - beta) * x_proj
        return 0.5 * (x1 + x_proj + x2)

    def _iter_DM(self, x: cp.ndarray, beta: float = 0.7, gs: float = None, gm: float = None) -> cp.ndarray:
        """Difference Maps algorithm iteration"""
        # Random projections and the optimization of an algorithm for phase retrieval, J. Phys. A-Math Gen, Vol 36 pp 2995-3007
        if gs is None:
            gs = 1 / beta
        if gm is None:
            gm = -1 / beta
        
        x_pmrs = self._proj_m(self._refl_s(x, g=gs))
        x_psrm = self._proj_s(self._refl_m(x, g=gm))
        return x + beta * (x_pmrs - x_psrm)

    def _iter_RAAR(self, x: cp.ndarray, beta: float = 0.7) -> cp.ndarray:
        """Relaxed Averaged Alternating Reflectors algorithm iteration"""
        return beta * self._iter_ASR(x) + (1 - beta) * self._proj_m(x)   

    def run(
        self,
        algorithm: callable,
        algorithm_kwargs: dict = None,
        init: cp.ndarray = None,
        max_iter: int = 100,
        rel_tol: float = 1e-3,
        ) -> None:

        x, results = spr.cupy.retrievalmath.run(
            magnitudes=(cp.abs(field), cp.abs(ft)),
            support=support,
            init=spr.initializers.random_phases(magnitude=cp.abs(field)),
            algorithm=spr.retrievalmath.RAAR,
            algorithm_kwargs={'beta': 0.7},
            max_iter=100,
            rel_tol=1e-6,
            )

    def define_support(self) -> None:
        pass

    def show(self) -> None:
        pass





