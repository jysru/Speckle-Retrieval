import numpy as np
import matplotlib.pyplot as plt

import speckleret.transforms as transforms
import speckleret.metrics as metrics
import speckleret.plots as plots


def error_reduction_fourier(magnitudes: tuple[np.ndarray, np.ndarray], init = None, max_iter: int = 200, pad: int = 2):
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


def hio_fourier(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray, init = None, beta: float = 0.99, max_iter: int = 200, pad: int = 2):
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
        y_hat[support] = np.abs(magnitudes[0][support]) * np.exp(1j * np.angle(y_hat_tmp[support]))
        y_hat[np.logical_not(support)] = y_hat[np.logical_not(support)] - beta * np.abs(magnitudes[0][np.logical_not(support)]) * np.exp(1j * np.angle(y_hat_tmp[np.logical_not(support)]))

    return (y_hat, ft_y_hat, results)


def hio_er_fourier(magnitudes: tuple[np.ndarray, np.ndarray], support: np.ndarray, max_iter: int = 3, init = None, beta: float = 0.99, max_er_iter: int = 200, max_hio_iter: int = 100, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magnitudes[0].shape)
        y_hat = np.abs(magnitudes[0]) * np.exp(1j * phi_init)
    else:
        y_hat = init

    results = {'mse_plane1': [], 'mse_plane2': []}
    for iter in range(max_iter):
        y_hat, ft_hat, res = hio_fourier(magnitudes, support, init=y_hat, pad=pad, max_iter=max_hio_iter)
        results = append_dict_keys_values(results, res)
        y_hat, ft_hat, res = error_reduction_fourier(magnitudes, init=y_hat, pad=pad, max_iter=max_er_iter)
        results = append_dict_keys_values(results, res)
        print(f"{iter + 1} / {max_iter}")

    return (y_hat, ft_hat, results)


def append_dict_keys_values(dict1: dict, dict2: dict):
    for key in dict1.keys():
        if key in dict2.keys():
            dict1[key] = dict1[key] + dict2[key]
    return dict1


if __name__ == "__main__":
    sz = 200
    field = np.random.randn(sz, sz) + 1j * np.random.randn(sz, sz)

    grid = np.arange(start=0, stop=field.shape[0]) - field.shape[0] / 2
    X, Y = np.meshgrid(grid, grid)
    R = np.sqrt(np.square(X) + np.square(Y))

    field1 = np.ones(field.shape) * np.exp(-np.square(R/20))
    field1 = field1 * np.exp(1j * (np.power(X/10, 3) + np.power(Y/10, 3)))
    field2 = np.ones(field.shape) * np.exp(-np.square(R/10))
    field2 = field2 * np.exp(1j * (np.square(X/3) + np.square(Y/3)))
    field = field1 + field2

    ft = transforms.fourier_transform(field, pad=2)
    # compare_complex_fields(field, ft)

    support_radius = 10
    support = np.zeros(field.shape, dtype=bool)
    support[R <= support_radius] = True

    # y_hat, ft_hat, results = error_reduction_fourier((np.abs(field), np.abs(ft)), pad=2, max_iter=100)
    # y_hat, ft_hat, results = hio_fourier((np.abs(field), np.abs(ft)), support, pad=2, max_iter=100)
    y_hat, ft_hat, results = hio_er_fourier((np.abs(field), np.abs(ft)), support, pad=2, max_iter=5, max_er_iter=200, max_hio_iter=100)

    plt.figure()
    plt.plot(results['mse_plane2'], label='Fourier MSE')
    plt.title('MSE')
    plt.xlabel('Iteration #')
    plt.yscale('log')


    plots.compare_complex_fields(field, y_hat)
    plots.compare_complex_fields(ft, ft_hat)
    plt.show()
