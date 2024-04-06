import os
import sys
import PIL.Image
import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
from torchvision import transforms as tr
from torchvision.transforms import Compose
import PIL
import tempfile
import multiprocessing

from image_processing import add_image_into_tr_omr_canvas
from music_xml import music_xml_to_semantic

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
    upper_bound, lower_bound = _get_image_bounds(dark_pixels_per_row)
    image = image[upper_bound:-lower_bound]
    dark_pixels_per_row = dark_pixels_per_row[upper_bound:-lower_bound]
    norm = (dark_pixels_per_row - np.mean(dark_pixels_per_row)) / np.std(dark_pixels_per_row)
    centers, _ = find_peaks(norm, height=1.4, distance=3, prominence=1)
    if len(centers) != 10:
        conv_len = image.shape[0] // 4 + 1
        blurred = np.convolve(dark_pixels_per_row, np.ones(conv_len) / conv_len, mode='same')

        # Find the central valley
        peaks, _ = find_peaks(-blurred, distance=50, prominence=1)
        if len(peaks) >= 1:
            peaks = [peaks[0]]
            #print(f"INFO: Using central valley {path}")
        elif len(centers) == 5:
            upper = _prepare_image(image)
            predistorted_path = basename + "_distorted.jpg"
            if os.path.exists(predistorted_path):
                predistorted_image = cv2.imread(predistorted_path)
                single_image = _prepare_image(predistorted_image)
                cv2.imwrite(basename + "_single-pre.jpg", single_image)
                return _distort_image(basename + "_single-pre.jpg"), None
            print(f"INFO: Couldn't find pre-distorted image {path}, using custom distortions")
            cv2.imwrite(basename + "_upper-pre.jpg", upper)
            return _distort_image(basename + "_upper-pre.jpg"), None
        else:
            print(f"INFO: Failed to split {path}, found {len(centers)} centers, no central valley")
            return None, None
        middle = peaks[0]
        center_upper = peaks[0] // 2 + _random_center_offset()
        center_lower = peaks[0] + (image.shape[0] - peaks[0]) // 2 + _random_center_offset()
    else:
        middle = np.int32(np.round((centers[4] + centers[5]) / 2))
        center_upper = centers[2] + _random_center_offset()
        center_lower = centers[7] + _random_center_offset()

    if middle < 15 or middle > image.shape[0] - 15:
        print(f"INFO: Failed to split {path}, middle is at {middle}")
        return None, None
    
    overlap = np.random.randint(0, 20)
    upper = _prepare_image(_center_image(image[:middle+overlap], center_upper))
    lower = _prepare_image(_center_image(image[middle-overlap:], center_lower))
    cv2.imwrite(basename + "_upper-pre.jpg", upper)
    cv2.imwrite(basename + "_lower-pre.jpg", lower)
    return _distort_image(basename + "_upper-pre.jpg"), _distort_image(basename + "_lower-pre.jpg")

def _random_center_offset():
    return np.random.randint(-20, 20)

def _center_image(image, center):
    """
    Creates a new image so that the y coordinate in center is at the vertical center of the new image.
    """
    new_center = np.int32(np.round(image.shape[0] / 2))
    offset = new_center - center
    new_image = 255 * np.ones((image.shape[0] + abs(offset), image.shape[1], 3), dtype=np.uint8)
    new_image[max(offset, 0):max(offset, 0)+image.shape[0]] = image
    return new_image

def _prepare_image(image):
    return add_image_into_tr_omr_canvas(image)

def _get_image_bounds(dark_pixels_per_row):
    white_upper_area_size = 0
    for i in range(dark_pixels_per_row.shape[0]):
        if dark_pixels_per_row[i] > 0:
            break
        white_upper_area_size += 1
    white_lower_area_size = 1
    for i in range(dark_pixels_per_row.shape[0] - 1, 0, -1):
        if dark_pixels_per_row[i] > 0:
            break
        white_lower_area_size += 1
    return white_upper_area_size, white_lower_area_size

def _check_staff_image(path, basename):
    """
    This method helps with reprocessing a folder more quickly by skipping
    the image splitting.
    """
    if not os.path.exists(basename + "_upper.jpg"):
        print(f"INFO: Failed to split {path}")
        return None, None
    return basename + "_upper.jpg", basename + "_lower.jpg"

def _distort_image(path):
    image = PIL.Image.open(path)
    image = _add_random_gray_tone(image)
    image = _random_zoom_while_keeping_the_original_size(image)
    pipeline = Compose(
             [tr.RandomRotation(degrees = 1),
              tr.RandomPerspective(0.05),
              tr.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1),
              tr.RandomAdjustSharpness(2),])

    augmented_image = pipeline(img = image)
    augmented_image.save(path)
    return path

def _add_random_gray_tone(image):
    image = np.array(image)
    random_gray_value = 255 - np.random.randint(0, 50)

    mask = np.all(image > random_gray_value, axis=-1)

    jitter = np.random.randint(-5, 5, size=mask.shape)

    gray = np.clip(random_gray_value + jitter, 0, 255)

    image[mask] = gray[mask, None]

    return PIL.Image.fromarray(image)

def _random_zoom_while_keeping_the_original_size(image):
    image_size = (image.height, image.width)
    random_zoom = np.random.uniform(0.8, 1.2)
    image_zoom = (int(image.width * random_zoom), int(image.height * random_zoom))
    image = image.resize(image_zoom)
    image_crop = (int((image.width - image_size[1]) / 2), int((image.height - image_size[0]) / 2))
    image = image.crop((image_crop[0], image_crop[1], image_crop[0] + image_size[1], image_crop[1] + image_size[0]))
    return image

def _music_xml_to_semantic(path, basename):
    result = music_xml_to_semantic(path)
    if len(result) != 2:
        return None, None
    with open(basename + "_upper.semantic", "w") as f:
        f.write(" ".join(result[0]))
    with open(basename + "_lower.semantic", "w") as f:
        f.write(" ".join(result[1]))
    return basename + "_upper.semantic", basename + "_lower.semantic"

def _convert_file(path: Path, ony_recreate_semantic_files = False):
    basename = str(path).replace(".krn", "")
    image_file = str(path).replace(".krn", ".jpg")
    distored = str(path).replace(".krn", "_distorted.jpg")
    musicxml = str(path).replace(".krn", ".musicxml")
    semantic = str(path).replace(".krn", ".semantic")
    result = os.system(f"{hum2xml} {path} > {musicxml}")
    if result != 0:
        print(f"Failed to convert {path}")
        return []
    if ony_recreate_semantic_files:
        upper, lower = _check_staff_image(image_file, basename)
    else:
        upper, lower = _split_staff_image(image_file, basename)
    if upper is None:
        return []
    upper_semantic, lower_semantic = _music_xml_to_semantic(musicxml, basename)
    if upper_semantic is None:
        print(f"Failed to convert {musicxml}")
        return []
    if lower is None:
        return [
            str(Path(upper).relative_to(git_root)) + "," + str(Path(upper_semantic).relative_to(git_root)),
        ]
    return [
        str(Path(upper).relative_to(git_root)) + "," + str(Path(upper_semantic).relative_to(git_root)),
        str(Path(lower).relative_to(git_root)) + "," + str(Path(lower_semantic).relative_to(git_root)),
    ]

def _convert_file_only_semantic(path):
    return _convert_file(path, True)

def convert_grandstaff(ony_recreate_semantic_files: bool = False):
    index_file = grandstaff_train_index
    if ony_recreate_semantic_files:
        index_file = tempfile.mktemp()
    
    print('Indexing Grandstaff dataset, this can up to an hour. "Failed to split" messages are expected during the run as the splitting works for most but not all files.')
    with open(index_file, 'w') as f:
        file_number = 0
        skipped_files = 0
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as p:
            for result in p.imap_unordered(_convert_file_only_semantic if ony_recreate_semantic_files else _convert_file, Path(grandstaff_root).rglob('*.krn')):
                if len(result) > 0:
                    for line in result:
                        f.write(line + '\n')
                else:
                    skipped_files += 1
                file_number += 1
                if file_number % 1000 == 0:
                    print(f"Processed {file_number} files, skipped {skipped_files} files")        
    print('Done indexing')

if __name__ == "__main__":
    ony_recreate_semantic_files = False
    if "--only-semantic" in sys.argv:
        ony_recreate_semantic_files = True
    convert_grandstaff(ony_recreate_semantic_files)