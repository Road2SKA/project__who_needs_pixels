# pip install ptwt
import numpy as np 
import torch, sys
import ptwt
import matplotlib.pyplot as plt
import matplotlib.colors as colors

device = "cuda" if torch.cuda.is_available() else "cpu"

src_img = np.load("data/true_image_83.npy")
src_img = np.array(src_img).astype(np.float32)
src_img_tensor = torch.from_numpy(src_img).to(device).contiguous()

cA2, cD2, cD1 = ptwt.wavedec2(src_img_tensor, "db1", level=2, mode="constant")
#approximation array
cA2= cA2.cpu().detach().numpy()
# detail level 2 (horizontal, vertical, diagonal)
cD2_h, cD2_v, cD2_d = cD2
#detail level 1 (horizontal, vertical, diagonal)
cD1_h, cD1_v, cD1_d = cD1

fig, axs = plt.subplots(3,3, figsize=(10, 5),layout='constrained')
axs[0,0].imshow(src_img, cmap = "cubehelix")
axs[0,1].axis("off")
axs[0,2].axis("off")
axs[1,0].imshow(cD1_h.cpu().detach().numpy(), cmap = "cubehelix")
axs[1,1].imshow(cD1_v.cpu().detach().numpy(), cmap = "cubehelix")
axs[1,2].imshow(cD1_d.cpu().detach().numpy(), cmap = "cubehelix")
axs[2,0].imshow(cD2_h.cpu().detach().numpy(), cmap = "cubehelix")
axs[2,1].imshow(cD2_v.cpu().detach().numpy(), cmap = "cubehelix")
axs[2,2].imshow(cD2_d.cpu().detach().numpy(), cmap = "cubehelix")
# axs[0].set_title("true image")
# for i in range(1,18):
#     axs[i].imshow(scattering_coefficients[i-1,:,:], cmap = "cubehelix")
plt.show()