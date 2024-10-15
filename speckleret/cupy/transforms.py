import cupy as cp


def apply_mask(field: cp.ndarray, mask: cp.ndarray):
    field = field.copy()
    field[cp.logical_not(mask)] = 0
    return field


def crop_img(img: cp.ndarray, newsize: tuple[int, int]):
    """ Crop image to new size, from the center."""
    diff_row = img.shape[0] - newsize[0]
    diff_col = img.shape[1] - newsize[1]
    crop_row, crop_col = diff_row // 2, diff_col // 2
    return img[crop_row:-crop_row, crop_col:-crop_col]


def pad_img(img: cp.ndarray, pad: float = 1):
    """ Pad zeroes on each axis.
        Default pad value is 1, which means that the padded image is twice the original size on each axis.
    """
    return cp.pad(img, pad_width=[int(img.shape[0] * pad / 2), int(img.shape[1] * pad / 2)], mode='constant')


def resize_image(img: cp.ndarray, new_size: tuple[int, int]) -> cp.ndarray:
    """ Resize a 2D image or array in a convenient manner

        If new_size is greater than the initial size, the image is padded with zeroes corresponding to the size difference.
        If new_size is smaller than the initial size, the image is cropped to the new size, from the center.
    """
    img = img.copy()
    init_size = cp.array(img.shape)
    new_size = cp.array(new_size)

    if cp.all(new_size > init_size):
        # Image will be padded with zeros
        pad_widths = [int((new_size[0] - init_size[0]) // 2), int((new_size[1] - init_size[1]) // 2)]
        img = cp.pad(img, pad_width=pad_widths, mode='constant')
    elif cp.all(new_size < init_size):
        # Image will be cropped
        img = crop_img(img, new_size)
    else:
        raise NotImplementedError('Please icput a square image!')
    return img


def normalize_field(field: cp.ndarray, type: str='energy') -> cp.ndarray:
    """ Normalize an electromagnetic field

        type can be:
            - 'energy' (default) : Field is normalized by its energy
            - 'max': Field is normalized by the maximum value of its magnitude
            
    """
    field = field.copy()
    type = type.lower()

    if type == 'energy':
        field_energy = cp.sum(cp.square(cp.abs(field)))
        field = field / cp.sqrt(field_energy)
    elif type == 'max':
        field = field / cp.max(cp.abs(field))
    return field


def fourier_transform(field: cp.ndarray, pad: float = None):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)
        
    ft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field)))
    ft = ft / cp.sqrt(ft.size)
    return crop_img(ft, init_shape) if pad is not None else ft


def fresnel_transform(field: cp.ndarray, dz: float = 0.0, pad: float = 2, wavelength: float = 1064e-9, pixel_size: float = 5.04e-6):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)
    
    _, _, kx, ky = fourier_grids(field, pixel_size=pixel_size)

    ft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field)))
    propagator = dz * cp.sqrt(4 * cp.square(cp.pi/wavelength) - cp.square(kx) - cp.square(ky))
    ft = ft * cp.exp(1j * propagator)
    ift = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(ft)))
    return crop_img(ift, init_shape) if pad is not None else ft


def fourier_grids(field: cp.ndarray, pixel_size: float):
    # Spatial plane
    dx = pixel_size
    n_pts = field.shape[0]
    grid_size = dx * (n_pts - 1)
    lim_x = n_pts / 2 * dx
    x = cp.arange(-lim_x, lim_x, dx)
    x, y = cp.meshgrid(x, x)

    # Conjugate plane
    dnx = 1 / grid_size
    lim_nx = (n_pts / 2) * dnx
    kx = 2 * cp.pi * cp.arange(-lim_nx, lim_nx, dnx)
    kx, ky = cp.meshgrid(kx, kx)
    return (x, y, kx, ky)


def inverse_fourier_transform(field: cp.ndarray, pad: float = None):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)
        
    ift = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(field)))
    ift = ift * cp.sqrt(ift.size)
    return crop_img(ift, init_shape) if pad is not None else ift

