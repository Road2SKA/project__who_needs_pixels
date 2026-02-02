#pip install PyWavelets
import sys
import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


data = np.load("data/true_image_83.npy")
nx, ny = data.shape


wavelet = "db1" #["db1", "db4", "db5"]
alpha = pywt.mra2(data, wavelet)

coeffs = []
coeffs.append(alpha[0])
for j in range(1, len(alpha)):
    for n in range(3):
        coeffs.append(alpha[j][n])
reco = sum(coeffs)

fig,axs = plt.subplots(2,len(coeffs), figsize=(6, 4),layout='constrained')
axs[0,0].imshow(data,cmap = "seismic",norm=colors.CenteredNorm())  
axs[0,0].set_title("Original") 
axs[0,1].imshow(reco,cmap = "seismic",norm=colors.CenteredNorm())   
axs[0,1].set_title("Wavelet reco") 
axs[0,2].axis("off")
axs[0,3].axis("off")
for i,c in enumerate(coeffs):
    axs[1,i].set_title("Coeff {0}".format(i))
    axs[1,i].imshow(c,cmap= "seismic",norm=colors.CenteredNorm())

plt.show()  
