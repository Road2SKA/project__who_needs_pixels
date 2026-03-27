import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as colors

def circular_mask(height=512, width=512, radius=256):

    center = (height // 2, width // 2)

    y, x = np.ogrid[:height, :width]
    dist_sq = (y - center[0])**2 + (x - center[1])**2

    mask = dist_sq <= radius**2
    return mask.astype(bool)

def semicircle_mask(height=512, width=512, radius=256):

    center = (height // 2, width // 2)

    y, x = np.ogrid[:height, :width]
    dist_sq = (y - center[0])**2 + (x - center[1])**2

    mask = dist_sq*1.0
    mask -= np.min(mask)
    mask /= np.max(mask)
    return 1-mask

def gaussian_mask(height=512, width=512, sigma=60, amplitude=1.0):

    center = (height // 2, width // 2)

    y, x = np.ogrid[:height, :width]

    dy = (y - center[0])**2 / (2 * sigma**2)
    dx = (x - center[1])**2 / (2 * sigma**2)

    mask = amplitude * np.exp(-(dy + dx))
    mask -= np.min(mask)
    mask /= np.max(mask)
    return mask
# image space ---> FT ---> visibility space
def ft(img):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

# visibility space ---> FT^{-1} ---> image space
def ift(img):
    return (np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(img)))).real

# img_true is in sky image space
img_true_file = "data/3c353_gdth.fits"

from astropy.io import fits

with fits.open(img_true_file) as hdul:
    # this image is 512 x 512, rescale to 150x150
    img_true = hdul[0].data.astype(np.float32)

#visibilities are in visibility space
vis_true = ft(img_true)
# mask = np.zeros_like(img_true).astype(bool)
# mask[128:384,128:384] = 1
mask = circular_mask(512,512,radius = 128)
mask = semicircle_mask(512,512,radius=128)
mask = gaussian_mask(512,512,sigma = 100)


vis_lo = vis_true*mask
vis_hi = vis_true*(1-mask)


img_hi = ift(vis_hi)
img_lo = ift(vis_lo)


fig, axs = plt.subplots(3,5, figsize=(10, 6),layout='constrained')
imgnorm = colors.SymLogNorm(linthresh=0.0001,vmin=0,vmax=1)

axs[0,0].imshow(img_true, cmap = "cubehelix",norm=imgnorm)
axs[0,0].set_title("true image")
axs[0,1].imshow(np.abs(vis_true),        cmap = "cubehelix",norm=imgnorm)
axs[0,1].set_title("true visibilities\n (magnitude)")
axs[0,2].imshow(np.angle(vis_true),      cmap = "twilight")
axs[0,2].set_title("true visibilities\n (phase)")
axs[0,3].imshow(img_hi+img_lo,    cmap = "cubehelix",norm=imgnorm)
axs[0,3].set_title("sum of hi- and\n lo-pass images")
axs[0,4].imshow(img_hi+img_lo-img_true,    cmap = "cubehelix",norm=imgnorm)
axs[0,4].set_title("residual")


axs[1,0].imshow(img_hi, cmap = "cubehelix",norm=imgnorm)
axs[1,0].set_title("high-pass image")
axs[1,1].imshow(np.abs(vis_hi),        cmap = "cubehelix",norm=imgnorm)
axs[1,1].set_title("high-pass visibilities\n (magnitude)")
axs[1,2].imshow(np.angle(vis_hi),      cmap = "twilight")
axs[1,2].set_title("high-pass visibilities\n (phase)")
axs[1,3].imshow(1-mask,        cmap = "cubehelix")
axs[1,3].set_title("hi mask")
axs[1,4].set_axis_off()

axs[2,0].imshow(img_lo, cmap = "cubehelix",norm=imgnorm)
axs[2,0].set_title("low-pass image")
axs[2,1].imshow(np.abs(vis_lo),        cmap = "cubehelix",norm=imgnorm)
axs[2,1].set_title("low-pass visibilities\n (magnitude)")
axs[2,2].imshow(np.angle(vis_lo),      cmap = "twilight")
axs[2,2].set_title("low-pass visibilities\n (phase)")
axs[2,3].imshow(mask,        cmap = "cubehelix")
axs[2,3].set_title("lo mask")
axs[2,4].set_axis_off()


plt.show()