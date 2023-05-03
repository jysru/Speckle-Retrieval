import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def Complex2HSV(z, rmin, rmax, hue_start=0):
    # get amplidude of z and limit to [rmin, rmax]
    amp = np.abs(z)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=True) + hue_start
    # HSV are values in range [0,1]
    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp -rmin) / (rmax - rmin)
    return hsv_to_rgb(np.dstack((h,s,v)))


fig, axs = plt.subplots(2, 6, figsize=(18, 6))
pl0 = axs[0,0].imshow(np.abs(output_field), cmap='gray')
pl0 = axs[0,1].imshow(np.abs(y_hat), cmap='gray')
pl2 = axs[0,2].imshow(np.angle(output_field * np.exp(-1j * np.angle(output_field[128,128]))), cmap='hsv')
pl2 = axs[0,3].imshow(np.angle(y_hat * np.exp(-1j * np.angle(y_hat[128,128]))), cmap='hsv')
pl4 = axs[0,4].imshow(Complex2HSV(output_field * np.exp(-1j * np.angle(output_field[128,128])), rmin=0, rmax=np.max(np.abs(output_field))), cmap='gray')
pl4 = axs[0,5].imshow(Complex2HSV(y_hat * np.exp(-1j * np.angle(y_hat[128,128])), rmin=0, rmax=np.max(np.abs(y_hat))), cmap='gray')

pl0 = axs[1,0].imshow(np.abs(tff_y_noref), cmap='gray')
pl0 = axs[1,1].imshow(np.abs(tff_y_hat), cmap='gray')
pl2 = axs[1,2].imshow(np.angle(tff_y_noref * np.exp(-1j * np.angle(tff_y_noref[128,128]))), cmap='hsv')
pl2 = axs[1,3].imshow(np.angle(tff_y_hat * np.exp(-1j * np.angle(tff_y_hat[128,128]))), cmap='hsv')
pl4 = axs[1,4].imshow(Complex2HSV(tff_y_noref * np.exp(-1j * np.angle(tff_y_noref[128,128])), rmin=0, rmax=np.max(np.abs(tff_y_noref))), cmap='gray')
pl4 = axs[1,5].imshow(Complex2HSV(tff_y_hat * np.exp(-1j * np.angle(tff_y_hat[128,128])), rmin=0, rmax=np.max(np.abs(tff_y_hat))), cmap='gray')

_ = axs[0,0].set_title("Target amplitude\nImage plane")
_ = axs[0,1].set_title("Retrieved amplitude\nImage plane")
_ = axs[0,2].set_title("Target phase\nImage plane")
_ = axs[0,3].set_title("Retrieved phase\nImage plane")
_ = axs[0,4].set_title("Target field (sim TM)\nImage plane")
_ = axs[0,5].set_title("Retrieved field (ref TM)\nImage plane")

_ = axs[1,0].set_title("Target amplitude\nFourier plane")
_ = axs[1,1].set_title("Retrieved amplitude\nFourier plane")
_ = axs[1,2].set_title("Target phase\nFourier plane")
_ = axs[1,3].set_title("Retrieved phase\nFourier plane")
_ = axs[1,4].set_title("Target field (sim TM)\nFourier plane")
_ = axs[1,5].set_title("Retrieved field (ref TM)\nFourier plane")

_ = [axs[j,i].set_xticks([]) for i in range(6) for j in range(2)]
_ = [axs[j,i].set_yticks([]) for i in range(6) for j in range(2)]