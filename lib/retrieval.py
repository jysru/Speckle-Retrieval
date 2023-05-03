import numpy as np
import lib.transforms as transforms
import lib.metrics as metrics


def error_reduction_fourier(magn_plane1, magn_plane2, init = None, max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magn_plane1.shape)
        y_hat = np.abs(magn_plane1) * np.exp(1j * phi_init)

    metrics = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        tf_y_hat = transforms.fourier_transform(y_hat, pad=pad)
        metrics['mse_plane2'].append(metrics.mse(np.abs(tf_y_hat), np.abs(magn_plane2)))
        tf_y_hat = np.abs(magn_plane2) * np.exp(1j * np.angle(tf_y_hat))
        y_hat = transforms.inverse_fourier_transform(tf_y_hat, pad=pad)
        metrics['mse_plane1'].append(metrics.mse(np.abs(y_hat), magn_plane1))
        y_hat = np.abs(magn_plane2) * np.exp(1j * np.angle(y_hat))

    return (y_hat, tf_y_hat, metrics)


def hio_fourier(magn_plane1, magn_plane2, init = None, max_iter: int = 200, pad: int = 2):
    if init is None:
        phi_init = 2 * np.pi * np.random.rand(*magn_plane1.shape)
        y_hat = np.abs(magn_plane1) * np.exp(1j * phi_init)

    metrics = {'mse_plane1': [], 'mse_plane2': []}
    for _ in range(max_iter):
        tf_y_hat = transforms.fourier_transform(y_hat, pad=pad)
        metrics['mse_plane2'].append(metrics.mse(np.abs(tf_y_hat), np.abs(magn_plane2)))
        tf_y_hat = np.abs(magn_plane2) * np.exp(1j * np.angle(tf_y_hat))
        y_hat = transforms.inverse_fourier_transform(tf_y_hat, pad=pad)
        metrics['mse_plane1'].append(metrics.mse(np.abs(y_hat), magn_plane1))
        y_hat = np.abs(magn_plane2) * np.exp(1j * np.angle(y_hat))

    return (y_hat, tf_y_hat, metrics)



def hio_fourier(target_field):

    pass



phi_init = 2 * np.pi * np.random.rand(*field.shape)
y_hat = np.abs(field) * np.exp(1j * phi_init)
tff_y_noref = fourier_transform(field, pad=2)
hio_beta = 0.99
hio_iter = 150
er_iter = 200

grid = np.arange(start=0, stop=y_hat.shape[0]) - y_hat.shape[0] / 2
X, Y = np.meshgrid(grid, grid)
R = np.sqrt(np.square(X) + np.square(Y))

support_radius = 50
support = np.zeros(y_hat.shape, dtype=bool)
support[R <= support_radius] = True

image_loss = []
fresnel_loss = []
for i in range(1):

    for j in range(hio_iter):
        tff_y_hat = fourier_transform(y_hat, pad=2)
        fresnel_loss.append(mse(tff_y_hat, tff_y_noref))
        tff_y_hat = np.abs(tff_y_noref) * np.exp(1j * np.angle(tff_y_hat))
        y_hat_tmp = inverse_fourier_transform(tff_y_hat, pad=2)
        image_loss.append(mse(y_hat_tmp, field))
        y_hat[support] = np.abs(field[support]) * np.exp(1j * np.angle(y_hat_tmp[support]))
        y_hat[np.logical_not(support)] = y_hat[np.logical_not(support)] - hio_beta * np.abs(field[np.logical_not(support)]) * np.exp(1j * np.angle(y_hat_tmp[np.logical_not(support)]))
        image_loss.append(mse(y_hat, field))

    for j in range(er_iter):
        tff_y_hat = fourier_transform(y_hat, pad=2)
        fresnel_loss.append(mse(tff_y_hat, tff_y_noref))
        tff_y_hat = np.abs(tff_y_noref) * np.exp(1j * np.angle(tff_y_hat))
        y_hat = inverse_fourier_transform(tff_y_hat, pad=2)
        image_loss.append(mse(y_hat, field))
        y_hat = np.abs(field) * np.exp(1j * np.angle(y_hat))
        # y_hat[np.logical_not(support)] = 0 


plt.figure()
# plt.plot(image_loss, label='Image MSE')
plt.plot(fresnel_loss, label='Fourier MSE')
plt.title('MSE')
plt.xlabel('Iteration #')
plt.yscale('log')