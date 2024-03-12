import os
import sys
import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms as tr
from torchvision.transforms import Compose
import PIL

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = os.path.join(script_location, '..')
grandstaff_root = os.path.join(git_root, 'grandstaff')
grandstaff_train_index = os.path.join(grandstaff_root, 'index.txt')

hum2xml = os.path.join(git_root, 'hum2xml')
if not os.path.exists(hum2xml):
    print("Download hum2xml from https://extras.humdrum.org/man/hum2xml/")
    sys.exit(1)

if not os.path.exists(grandstaff_root):
    print("Download grandstaff from https://sites.google.com/view/multiscore-project/datasets")
    sys.exit(1)

def _get_dark_pixels_per_row(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_pixels_per_row = np.zeros(gray.shape[0])
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] < 200:
                dark_pixels_per_row[i] += 1
    return dark_pixels_per_row

def _split_staff_image(path, basename):
    """
    This algorithm is taken from `oemer` staffline extraction algorithm. In this simplified version
    it only works with images which have no distortions.
    """
    image = cv2.imread(path)
    dark_pixels_per_row = _get_dark_pixels_per_row(image)
    norm = (dark_pixels_per_row - np.mean(dark_pixels_per_row)) / np.std(dark_pixels_per_row)
    centers, _ = find_peaks(norm, height=2.5, distance=5, prominence=1)
    if len(centers) != 10:
        return None, None
    middle = np.int32(np.round((centers[4] + centers[5]) / 2))
    upper = image[:middle]
    lower = image[middle:]
    cv2.imwrite(basename + "_upper.jpg", upper)
    cv2.imwrite(basename + "_lower.jpg", lower)
    return _distort_image(basename + "_upper.jpg"), _distort_image(basename + "_lower.jpg")

def _distort_image(path):
    image = PIL.Image.open(path)
    pipeline = Compose(
             [tr.RandomRotation(degrees = 2),
              tr.RandomPerspective(0.1),
              tr.RandomAdjustSharpness(2),])

    augmented_image = pipeline(img = image)
    augmented_image.save(path)
    return path
    
    
def _musix_xml_to_semantic(path):
    pass

def convert_grandstaff():
    if os.path.exists(grandstaff_train_index):
        return
    
    print('Indexing Grandstaff dataset')
    with open(grandstaff_train_index, 'w') as f:
        for path in Path(grandstaff_root).rglob('*.krn'):
            basename = str(path).replace(".krn", "")
            image_file = str(path).replace(".krn", ".jpg")
            distored = str(path).replace(".krn", "_distorted.jpg")
            musicxml = str(path).replace(".krn", ".musicxml")
            semantic = str(path).replace(".krn", ".semantic")
            result = os.system(f"{hum2xml} {path} > {musicxml}")
            if result != 0:
                print(f"Failed to convert {path}")
                continue
            upper, lower = _split_staff_image(image_file, basename)
            if upper is None:
                print(f"Failed to split {image_file}")
                continue
            f.write(str(Path(image_file).relative_to(git_root)) + '\n')
    print('Done indexing')

if __name__ == "__main__":
    convert_grandstaff()