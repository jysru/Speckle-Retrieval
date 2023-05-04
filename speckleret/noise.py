import numpy as np

intens_noise = 0*0.003
wavelength = 1064e-9
phase_noise = 0*wavelength/30


def add_intensity_noise(field, noise_std: float = 0):
    if noise_std == 0:
        return field
    else:
        intens = np.square(np.abs(field))
        max_intens = np.max(intens)
        intens = max_intens * np.abs(intens / max_intens + noise_std * np.random.randn(*intens.shape))
        return np.sqrt(intens) * np.exp(1j * np.angle(field))


def add_phase_noise(field, noise_std: float = 0, wavelength=1064e-9):
    if noise_std == 0:
        return field
    else:
        phase = np.angle(np.exp(1j * (np.angle(field) + 2 * np.pi / wavelength * noise_std * np.random.randn(*field.shape))))
        return np.abs(field) * np.exp(1j * phase)


def add_complex_noise(field, intens_noise_std: float = 0, phase_noise_std: float = 0, wavelength: float = 1064e-9):
    if intens_noise_std != 0:
        field = add_intensity_noise(field, noise_std=intens_noise_std)
    if phase_noise_std != 0:
        field = add_phase_noise(field, noise_std=phase_noise_std, wavelength=wavelength)
    return field