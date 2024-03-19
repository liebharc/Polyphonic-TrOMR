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
import xmltodict
import tempfile

from circle_of_fifths import KeyTransformation, circle_of_fifth_to_key_signature

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
    centers, _ = find_peaks(norm, height=2.5, distance=5, prominence=1)
    if len(centers) != 10:
        return None, None
    middle = np.int32(np.round((centers[4] + centers[5]) / 2))
    upper = _center_image(image[:middle], centers[2] + _random_center_offset())
    lower = _center_image(image[middle:], centers[7] - middle + _random_center_offset())
    cv2.imwrite(basename + "_upper.jpg", upper)
    cv2.imwrite(basename + "_lower.jpg", lower)
    return _distort_image(basename + "_upper.jpg"), _distort_image(basename + "_lower.jpg")

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

def _get_image_bounds(dark_pixels_per_row):
    white_upper_area_size = 0
    for i in range(dark_pixels_per_row.shape[0]):
        if dark_pixels_per_row[i] > 0:
            break
        white_upper_area_size += 1
    white_lower_area_size = 0
    for i in range(dark_pixels_per_row.shape[0] - 1, -1, -1):
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
        return None, None
    return basename + "_upper.jpg", basename + "_lower.jpg"

def _distort_image(path):
    image = PIL.Image.open(path)
    pipeline = Compose(
             [tr.RandomRotation(degrees = 1),
              tr.RandomPerspective(0.05),
              tr.RandomAdjustSharpness(2),])

    augmented_image = pipeline(img = image)
    augmented_image.save(path)
    return path
    
def _music_xml_to_semantic(path, basename):
    result = []
    with open(path) as f:
        musicxml = xmltodict.parse(f.read())
        parts = musicxml['score-partwise']["part"]
        for part in parts:
            semantic = _music_part_to_semantic(part)
            result.append(semantic)
    if len(result) != 2:
        return None, None
    with open(basename + "_upper.semantic", "w") as f:
        f.write(" ".join(result[0]))
    with open(basename + "_lower.semantic", "w") as f:
        f.write(" ".join(result[1]))
    return basename + "_upper.semantic", basename + "_lower.semantic"

def _ensure_list(obj):
    if type(obj) is list:
        return obj
    return [obj]

def _count_dots(note):
    if "dot" not in note:
        return ""
    return "." * len(_ensure_list(note["dot"]))

def _music_part_to_semantic(part):
    try:
        semantic = []
        for measure in _ensure_list(part["measure"]):
            chord = []
            key = KeyTransformation(0)
            if "attributes" in measure:
                for attribute in _ensure_list(measure["attributes"]):
                    if "clef" in attribute:
                        semantic.append("clef-" + attribute["clef"]["sign"] + attribute["clef"]["line"])
                    if "time" in attribute:
                        semantic.append("timeSignature-" + attribute["time"]["beats"] + "/" + attribute["time"]["beat-type"])
                    if "key" in attribute:
                        semantic.append("keySignature-" + circle_of_fifth_to_key_signature(int(attribute["key"]["fifths"])))
                        key = KeyTransformation(int(attribute["key"]["fifths"]))
            if "note" in measure:
                for note in  _ensure_list(measure["note"]):
                    if "rest" in note:
                        dot = _count_dots(note)
                        semantic.append("rest-" + _translate_duration(note["type"]) + dot)
                        if len(chord) > 0:
                            # FLush the previous chord
                            semantic.append("|".join(chord))
                            chord = []
                    if "pitch" in note:
                        if not "chord" in note:
                            if len(chord) > 0:
                                # FLush the previous chord
                                semantic.append("|".join(chord))
                                chord = []
                        key.add_accidental(note["pitch"]["step"], _get_alter(note["pitch"]))
                        alter = _get_alter(note["pitch"])
                        chord.append("note-" + note["pitch"]["step"] + alter + note["pitch"]["octave"] + "_" + _translate_duration(note["type"]) + _count_dots(note))
                        
            if len(chord) > 0:
                # FLush the last chord
                semantic.append("|".join(chord))
            semantic.append("barline")
            key = key.reset_at_end_of_measure()

        # Remove the last bar line
        if len(semantic) > 0 and semantic[-1] == "barline":
            semantic.pop()
        return semantic
    except Exception as e:
        print("Failure at ", part)
        raise e

def _translate_duration(duration):
    definition = {
        "breve": "double_whole",
        "whole": "whole",
        "half": "half",
        "quarter": "quarter",
        "eighth": "eighth",
        "16th": "sixteenth",
        "32nd": "thirty_second",
        "64th": "sixty_fourth",
    }
    return definition[duration]

def _get_alter(note):
    if not "alter" in note:
        return ""
    if note["alter"] == "1":
        return "#"
    if note["alter"] == "-1":
        return "b"
    if note["alter"] == "0":
        return "0"
    return ""

def convert_grandstaff(ony_recreate_semantic_files: bool = False):
    index_file = grandstaff_train_index
    if ony_recreate_semantic_files:
        index_file = tempfile.mktemp()

    if os.path.exists(index_file):
        return
    
    print('Indexing Grandstaff dataset, this can take several hours')
    with open(index_file, 'w') as f:
        file_number = 0
        for path in Path(grandstaff_root).rglob('*.krn'):
            file_number += 1
            if file_number % 1000 == 0:
                print(f"Processed {file_number} files")
            basename = str(path).replace(".krn", "")
            image_file = str(path).replace(".krn", ".jpg")
            distored = str(path).replace(".krn", "_distorted.jpg")
            musicxml = str(path).replace(".krn", ".musicxml")
            semantic = str(path).replace(".krn", ".semantic")
            result = os.system(f"{hum2xml} {path} > {musicxml}")
            if result != 0:
                print(f"Failed to convert {path}")
                continue
            if ony_recreate_semantic_files:
                upper, lower = _check_staff_image(image_file, basename)
            else:
                upper, lower = _split_staff_image(image_file, basename)
            if upper is None:
                print(f"Failed to split {image_file}")
                continue
            upper_semantic, lower_semantic =_music_xml_to_semantic(musicxml, basename)
            if upper_semantic is None:
                print(f"Failed to convert {musicxml}")
                continue
            f.write(str(Path(upper).relative_to(git_root)) + '\n')
            f.write(str(Path(lower).relative_to(git_root)) + '\n')
    print('Done indexing')

if __name__ == "__main__":
    ony_recreate_semantic_files = False
    if "--only-semantic" in sys.argv:
        ony_recreate_semantic_files = True
    convert_grandstaff(ony_recreate_semantic_files)