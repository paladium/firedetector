from PIL import Image
import os, sys


def resize(path: str, index: int, output: str):
    package_dir = os.getcwd()
    dirs = os.listdir(os.path.join(package_dir, path))
    i = 0
    for item in dirs:
        filepath = os.path.join(package_dir, path, item)
        if os.path.isfile(filepath):
            im = Image.open(filepath)
            imResize = im.resize((28, 28), Image.ANTIALIAS).convert('RGB')
            imResize.save(output + str(index) + "_" +  str(i) + '.jpg', 'JPEG', quality=100)
        i += 1
resize("downloads/forest", 0, "no-fire/")