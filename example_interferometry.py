import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# image space ---> FT ---> visibility space
def ft(img):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

# visibility space ---> FT^{-1} ---> image space
def ift(img):
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(img))))

# img_true is in sky image space
img_true = np.load("data/true_image_83.npy")

#visibilities are in visibility space
vis_true = ft(img_true)

# S is in visiiblity space
S = np.load("data/EHT_mask_150.npy")

# sampled visibilities are in visibility space
vis_sampled = vis_true*S

# PSF is in image space
psf= ift(S)

# dirty image is in image space
img_dirty = ift(vis_sampled)

fig, axs = plt.subplots(2,4, figsize=(10, 5),layout='constrained')
axs=axs.flatten()

axs[0].imshow(img_true, cmap = "cubehelix")
axs[0].set_title("true image")
axs[1].imshow(S,        cmap = "cubehelix")
axs[1].set_title("sampling mask")
axs[2].imshow(psf,      cmap = "cubehelix")
axs[2].set_title("PSF")
axs[3].imshow(img_dirty, cmap = "cubehelix")
axs[3].set_title("dirty image")
axs[4].imshow(np.abs(vis_true),        cmap = "cubehelix")
axs[4].set_title("true visibilities\n (magnitude)")
axs[5].imshow(np.angle(vis_true),      cmap = "twilight")
axs[5].set_title("true visibilities\n (phase)")
axs[6].imshow(np.abs(vis_sampled),     cmap = "cubehelix")
axs[6].set_title("sampled visibilities\n (magnitude)")
axs[7].imshow(S*np.angle(vis_sampled), cmap = "twilight")
axs[7].set_title("sampled visibilities\n (phase)")
plt.show()