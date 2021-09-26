import pandas as pd
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from tools.detection import detect_object
from tools.drawing import annotate_objects
from tools.processing import resize
import time

implot = lambda ax, im: ax.imshow(im, cmap='Greys_r', vmin=0, vmax=255)

globs = glob.glob('datapath/*.jpg')
fig = plt.figure(tight_layout=True)
fig.set_figheight(6)
fig.set_figwidth(10)
ax_main = plt.subplot2grid(shape=(3, 4), loc=(0,0), colspan=2, rowspan=3)
axs = []
axs.append(plt.subplot2grid(shape=(3, 4), loc=(0,2)))
axs.append(plt.subplot2grid(shape=(3, 4), loc=(1,2)))
axs.append(plt.subplot2grid(shape=(3, 4), loc=(2,2)))
axs.append(plt.subplot2grid(shape=(3, 4), loc=(0,3)))
axs.append(plt.subplot2grid(shape=(3, 4), loc=(1,3)))
axs.append(plt.subplot2grid(shape=(3, 4), loc=(2,3)))
[ax.axis('off') for ax in axs]
for idx, filename in enumerate(globs):
    print("{}/{}".format(idx, len(globs)))
    image = cv.imread(filename, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    faces = detect_object(resize(image, 0.75), 'face_front', image)
    for i, face in enumerate(faces):
        x, y, w, h = face
        face_crop = image[y:y+h, x:x+w]
        implot(axs[np.mod(i, 6)], face_crop)
        facepath = filename.split('/')[-1]
        facepath = "faces/" + facepath.split('.')[0] + "_{}.jpg".format(i)
        cv.imwrite(facepath, cv.cvtColor(face_crop, cv.COLOR_BGR2RGB))
    annotate_objects(image, faces, 'Face', 1)
    implot(ax_main, image)
    # plt.draw()
    # plt.pause(0.1)
    [ax.clear() for ax in axs]
    [ax.axis('off') for ax in axs]
    ax_main.axis('off')
