import numpy as np
import matplotlib.pyplot as plt

import speckleret.transforms as transforms
import speckleret.metrics as metrics
import speckleret.plots as plots
import speckleret.supports as supports



def gerchberg_saxton_fourier(magnitudes: tuple[np.ndarray, np.ndarray],
                             init = None,max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        ft_y_hat = transforms.fourier_transform(y_hat, pad=pad)
        results['mse_plane2'].append(metrics.mse(ft_y_hat, magnitudes[1]))
        ft_y_hat = np.abs(magnitudes[1]) * np.exp(1j * np.angle(ft_y_hat))
        y_hat = transforms.inverse_fourier_transform(ft_y_hat, pad=pad)
        results['mse_plane1'].append(metrics.mse(y_hat, magnitudes[0]))
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * np.angle(y_hat))

    return (y_hat, ft_y_hat, results)


def gerchberg_saxton_fresnel(magnitudes: tuple[np.ndarray, np.ndarray], 
                             dz: float, wavelength: float = 1064e-9, pixel_size: float=5.04e-6,
                             init = None, max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        ft_y_hat = transforms.fresnel_transform(y_hat, dz=dz, wavelength=wavelength, pixel_size=pixel_size, pad=pad)
        results['mse_plane2'].append(metrics.mse(ft_y_hat, magnitudes[1]))
        ft_y_hat = np.abs(magnitudes[1]) * np.exp(1j * np.angle(ft_y_hat))
        y_hat = transforms.fresnel_transform(ft_y_hat, dz=-dz, wavelength=wavelength, pixel_size=pixel_size, pad=pad)
        results['mse_plane1'].append(metrics.mse(y_hat, magnitudes[0]))
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * np.angle(y_hat))

    return (y_hat, ft_y_hat, results)


def error_reduction_fourier(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray,
                            init = None, max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        ft_y_hat = transforms.fourier_transform(y_hat, pad=pad)
        results['mse_plane2'].append(metrics.mse(ft_y_hat, magnitudes[1]))
        ft_y_hat = np.abs(magnitudes[1]) * np.exp(1j * np.angle(ft_y_hat))
        y_hat = transforms.inverse_fourier_transform(ft_y_hat, pad=pad)
        results['mse_plane1'].append(metrics.mse(y_hat, magnitudes[0]))
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * np.angle(y_hat))
        y_hat[np.logical_not(support)] = 0

    return (y_hat, ft_y_hat, results)


def error_reduction_fresnel(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray,
                            dz: float, wavelength: float = 1064e-9, pixel_size: float=5.04e-6,
                            init = None, max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        ft_y_hat = transforms.fresnel_transform(y_hat, dz=dz, wavelength=wavelength, pixel_size=pixel_size, pad=pad)
        results['mse_plane2'].append(metrics.mse(ft_y_hat, magnitudes[1]))
        ft_y_hat = np.abs(magnitudes[1]) * np.exp(1j * np.angle(ft_y_hat))
        y_hat = transforms.fresnel_transform(ft_y_hat, dz=-dz, wavelength=wavelength, pixel_size=pixel_size, pad=pad)
        results['mse_plane1'].append(metrics.mse(y_hat, magnitudes[0]))
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * np.angle(y_hat))
        y_hat[np.logical_not(support)] = 0

    return (y_hat, ft_y_hat, results)


def hio_fourier(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray,
                init = None, beta: float = 0.99, max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        ft_y_hat = transforms.fourier_transform(y_hat, pad=pad)
        results['mse_plane2'].append(metrics.mse(ft_y_hat, magnitudes[1]))
        ft_y_hat = np.abs(magnitudes[1]) * np.exp(1j * np.angle(ft_y_hat))

        y_hat_tmp = transforms.inverse_fourier_transform(ft_y_hat, pad=pad)
        results['mse_plane1'].append(metrics.mse(y_hat_tmp, magnitudes[0]))
        y_hat_tmp = np.abs(magnitudes[0]) * np.exp(1j * np.angle(y_hat_tmp))
        y_hat[np.logical_not(support)] = y_hat[np.logical_not(support)] - beta * y_hat_tmp[np.logical_not(support)]
        y_hat[support] = y_hat_tmp[support]

    return (y_hat, ft_y_hat, results)


def hio_fresnel(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray,
                dz: float, wavelength: float = 1064e-9, pixel_size: float=5.04e-6,
                init = None, beta: float = 0.99, max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        ft_y_hat = transforms.fresnel_transform(y_hat, dz=dz, wavelength=wavelength, pixel_size=pixel_size, pad=pad)
        results['mse_plane2'].append(metrics.mse(ft_y_hat, magnitudes[1]))
        ft_y_hat = np.abs(magnitudes[1]) * np.exp(1j * np.angle(ft_y_hat))

        y_hat_tmp = transforms.fresnel_transform(ft_y_hat, dz=-dz, wavelength=wavelength, pixel_size=pixel_size, pad=pad)
        results['mse_plane1'].append(metrics.mse(y_hat_tmp, magnitudes[0]))
        y_hat_tmp = np.abs(magnitudes[0]) * np.exp(1j * np.angle(y_hat_tmp))
        y_hat[np.logical_not(support)] = y_hat[np.logical_not(support)] - beta * y_hat_tmp[np.logical_not(support)]
        y_hat[support] = y_hat_tmp[support]

    return (y_hat, ft_y_hat, results)


def hio_er_fourier(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray,
                   max_iter: int = 3, init = None,
                   beta: float = 0.99,
                   max_er_iter: int = 200, max_hio_iter: int = 100, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for iter in range(max_iter):
        y_hat, ft_hat, res = hio_fourier(magnitudes, support, init=y_hat, pad=pad, max_iter=max_hio_iter, beta=beta)
        results = append_dict_keys_values(results, res)
        y_hat, ft_hat, res = error_reduction_fourier(magnitudes, support=support, init=y_hat, pad=pad, max_iter=max_er_iter)
        results = append_dict_keys_values(results, res)
        print(f"{iter + 1} / {max_iter}")

    return (y_hat, ft_hat, results)


def hio_er_fresnel(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray, 
                   dz: float, wavelength: float = 1064e-9, pixel_size: float=5.04e-6,
                   max_iter: int = 3, init = None,
                   beta: float = 0.99, max_er_iter: int = 200, max_hio_iter: int = 100, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for iter in range(max_iter):
        y_hat, ft_hat, res = hio_fresnel(magnitudes, support, dz=dz, wavelength=wavelength, pixel_size=pixel_size, init=y_hat, pad=pad, max_iter=max_hio_iter, beta=beta)
        results = append_dict_keys_values(results, res)
        y_hat, ft_hat, res = error_reduction_fresnel(magnitudes, support, dz=dz, wavelength=wavelength, pixel_size=pixel_size, init=y_hat, pad=pad, max_iter=max_er_iter)
        results = append_dict_keys_values(results, res)
        print(f"{iter + 1} / {max_iter}")

    return (y_hat, ft_hat, results)


def hio_gs_fourier(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray,
                   max_iter: int = 3, init = None,
                   beta: float = 0.99,
                   max_er_iter: int = 200, max_hio_iter: int = 100, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for iter in range(max_iter):
        y_hat, ft_hat, res = hio_fourier(magnitudes, support, init=y_hat, pad=pad, max_iter=max_hio_iter, beta=beta)
        results = append_dict_keys_values(results, res)
        y_hat, ft_hat, res = gerchberg_saxton_fourier(magnitudes, init=y_hat, pad=pad, max_iter=max_er_iter)
        results = append_dict_keys_values(results, res)
        print(f"{iter + 1} / {max_iter}")

    return (y_hat, ft_hat, results)


def hio_gs_fresnel(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray,
                   dz: float, wavelength: float = 1064e-9, pixel_size: float=5.04e-6,
                   max_iter: int = 3, init = None, beta: float = 0.99,
                   max_er_iter: int = 200, max_hio_iter: int = 100, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for iter in range(max_iter):
        y_hat, ft_hat, res = hio_fresnel(magnitudes, support, dz=dz, wavelength=wavelength, pixel_size=pixel_size, init=y_hat, pad=pad, max_iter=max_hio_iter, beta=beta)
        results = append_dict_keys_values(results, res)
        y_hat, ft_hat, res = gerchberg_saxton_fresnel(magnitudes, dz=dz, wavelength=wavelength, pixel_size=pixel_size, init=y_hat, pad=pad, max_iter=max_er_iter)
        results = append_dict_keys_values(results, res)
        print(f"{iter + 1} / {max_iter}")

    return (y_hat, ft_hat, results)


def append_dict_keys_values(dict1: dict, dict2: dict):
    for key in dict1.keys():
        if key in dict2.keys():
            dict1[key] = dict1[key] + dict2[key]
    return dict1


if __name__ == "__main__":
    sz = 50
    X, Y, R, _ = supports.pixels_meshgrids(sz)
    field = np.random.randn(sz, sz) + 1j * np.random.randn(sz, sz)
    
    field = np.ones(field.shape) * np.exp(-np.square(R/15))
    field = field * np.exp(1j * (np.power(X/10, 3) + np.power(Y/10, 3)))
    field = transforms.pad_img(field, pad=2)

    ft = transforms.fourier_transform(field, pad=None)
    # ft = transforms.fresnel_transform(field, dz=20e-6, pad=None)
    support = supports.disk_support(field, radius=40)

    # y_hat, ft_hat, results = error_reduction_fourier((np.abs(field), np.abs(ft)), support=support, pad=2, max_iter=500)
    # y_hat, ft_hat, results = hio_fourier((np.abs(field), np.abs(ft)), support, pad=None, max_iter=500, beta=0.8)
    y_hat, ft_hat, results = hio_er_fourier((np.abs(field), np.abs(ft)), support, pad=None, max_iter=3, max_er_iter=150, max_hio_iter=150)
    # y_hat, ft_hat, results = hio_gs_fourier((np.abs(field), np.abs(ft)), support, pad=None, max_iter=3, max_er_iter=150, max_hio_iter=150)
    print(metrics.quality(y_hat[support], field[support]))

    plt.figure()
    plt.plot(results['mse_plane2'], label='Fourier MSE')
    plt.title('MSE')
    plt.xlabel('Iteration #')
    plt.yscale('log')

    plots.compare_complex_fields(field, y_hat)
    plots.compare_complex_fields(ft, ft_hat)
    plt.show()
