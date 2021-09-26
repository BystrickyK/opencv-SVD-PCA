import numpy as np
import glob
import cv2 as cv
from tools.processing import resize_to_constant
import matplotlib.pyplot as plt

globs = glob.glob('faces/*.jpg')

image_library = []
size = (128, 128)
for i, filepath in enumerate(globs):
    image = cv.imread(filepath, 0)
    image = resize_to_constant(image, size)
    image_library.append(np.reshape(image, -1))

#%% Creating eigenface library
pc_num = 128
U, E, Vt = np.linalg.svd(np.array(image_library).T, full_matrices=False)
U_ = U[:, :pc_num]
E_ = np.diag(E[:pc_num])
Vt_ = np.array(Vt[:pc_num, :])

fig, axs = plt.subplots(nrows=int(np.sqrt(pc_num)),
                        ncols=int(np.sqrt(pc_num))+1, tight_layout=True)
axs = np.reshape(axs, -1)
[ax.axis('off') for ax in axs]
U_cols = [U_[:, i] for i in range(pc_num)]
for i, (ax, eigenface) in enumerate(zip(axs, U_cols)):
    eigenface = np.reshape(eigenface, size)
    ax.imshow(eigenface, cmap='Greys_r')
    ax.set_title("PC{}: {:.2e}".format(i, E_[i, i]))

#%% Reconstructing faces
fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout=True)
axs = np.reshape(axs, -1)
[ax.axis('off') for ax in [axs[0], axs[1], axs[2]]]
for idx, img in enumerate(image_library):
    Vt_img = Vt_[:, idx]
    Vt_img = np.reshape(Vt_img, [len(Vt_img), 1])
    img_PCA = U_ @ E_ @ Vt_img
    img = np.reshape(img, size)
    img_PCA = np.reshape(img_PCA, size)
    axs[0].imshow(img, cmap='Greys_r', vmin=0, vmax=255)
    axs[1].imshow(img_PCA, cmap='Greys_r', vmin=0, vmax=255)
    axs[2].imshow(np.abs(img-img_PCA), cmap='Blues_r')
    error = np.sum(np.abs(img-img_PCA)) / (size[0]**2)
    axs[2].set_title("Reconstruction error: {:.0f}".format(error))
    axs[3].bar(range(pc_num), np.reshape(Vt_img, -1))
    plt.draw()
    plt.pause(1)
    axs[3].clear()
