import numpy as np
import imageio
import os
from pathlib import Path


def importImages(pathz):
    directory = os.fsencode(pathz)
    images = []
    for filez in os.listdir(directory):
        im = os.fsdecode(filez)
        if im.endswith(".jpg") and images is not None:
              im1 = imageio.imread(os.path.join(directory, im))
              np.vstack((images, im1))
        elif im.endswith(".jpg"):
            images = imageio.imread(os.path.join(directory, im))

directory = os.fsencode("/Users/antonis/Desktop/Themas/DP1")
images = []
for filez in os.listdir(directory):
    im = os.fsdecode(filez)
    if im.endswith(".jpg") and bool(images) = True:
        im1 = imageio.imread(os.path.join(directory, filez))
        np.vstack((images, im1))
    elif im.endswith(".jpg"):
        images = imageio.imread(os.path.join(directory, im))

pathlist = Path("/Users/antonis/Desktop/Themas/DP1").glob('*.jpg')
images = []
for path in pathlist:
    path_in_str = str(path)
    if images is not None:
        im = imageio.imread(path_in_str)
        images = np.reshape(im, (im.shape[0] * im.shape[1], im.shape[2]))
        np.vstack((images, im))
    elif images is None:
        images = imageio.imread(path_in_str)
        images = np.reshape(images, (images.shape[0] * images.shape[1], images.shape[2]))
im1 = imageio.imread("/Users/antonis/Desktop/Themas/DP1/magma_wrasse3_b.p._shutman.jpg")
im2 = imageio.imread("/Users/antonis/Desktop/Themas/DP1/220px-Georgia_Aquarium_-_Giant_Grouper_edit.jpg")


    path_in_str = str(path)
pathlist = Path("/Users/antonis/Desktop/Themas/DP1").glob('*.jpg')
images = []
for path in pathlist:
    path_in_str = str(path)
    if images is not None:
        im = imageio.imread(path_in_str)
        images = np.reshape(im, (im.shape[0] * im.shape[1], im.shape[2]))
        np.vstack((images, im))
    elif images is None:
        images = imageio.imread(path_in_str)
        images = np.reshape(images, (images.shape[0] * images.shape[1], images.shape[2]))
im1 = imageio.imread("/Users/antonis/Desktop/Themas/DP1/magma_wrasse3_b.p._shutman.jpg")
im2 = imageio.imread("/Users/antonis/Desktop/Themas/DP1/220px-Georgia_Aquarium_-_Giant_Grouper_edit.jpg")


    path_in_str = str(path)



im1 = np.reshape(file, (file.shape[0] * file.shape[1], file.shape[2]))

shape = (im1.shape[0]*im1.shape[1], im1.shape[2])
np.reshape(im1, (im1.shape[0]*im1.shape[1], im1.shape[2]))


im1 = imageio.imread("/Users/antonis/Desktop/Themas/DP1/magma_wrasse3_b.p._shutman.jpg")
im2 = imageio.imread("/Users/antonis/Desktop/Themas/DP1/220px-Georgia_Aquarium_-_Giant_Grouper_edit.jpg")

