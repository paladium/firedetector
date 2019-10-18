import numpy as py
import os
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def getPath(isFire=False):
    package_dir = os.getcwd()
    return (os.path.join(package_dir, "fire" if isFire else "no-fire"))


class Dataset:
    def __init__(self):
        fire = os.listdir(getPath(True))
        notFire = os.listdir(getPath(False))
        fireImages = []
        labelFire = []

        # Augment fire
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])

        for image in fire:
            path = os.path.join(getPath(True), image)
            loadedImage = load_image(path)
            fireImages.append(loadedImage)
            labelFire.append(1)
        fireImages = np.array(fireImages)
        fireAugmented = seq(images=fireImages)
        labelFire.extend(labelFire)

        notFireImages = []
        labelNotFire = []

        for image in notFire:
            path = os.path.join(getPath(False), image)
            loadedImage = load_image(path)
            notFireImages.append(loadedImage)
            labelNotFire.append(0)
        notFireImages = np.array(notFireImages)
        notFireAugmented = seq(images=notFireImages)
        labelNotFire.extend(labelNotFire)
        self.dataset = np.concatenate(
            (fireImages, fireAugmented, notFireAugmented, notFireImages))
        self.labels = np.array(labelFire + labelNotFire)
