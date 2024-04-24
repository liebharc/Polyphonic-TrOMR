
import os
from pathlib import Path
import random
import cv2
import multiprocessing

from image_processing import add_image_into_tr_omr_canvas


script_location = os.path.dirname(os.path.realpath(__file__))
git_root = os.path.join(script_location, '..')
primus = os.path.join(git_root, 'Corpus')
cpms = os.path.join(git_root, 'CPMS')
primus_train_index = os.path.join(primus, 'index.txt')
primus_distorted_train_index = os.path.join(primus, 'distored_index.txt')
cpms_train_index = os.path.join(git_root, 'cpms_index.txt')

def _replace_suffix(path: Path, suffix: str):
    suffixes = [
        "_distorted.jpg",
        ".jpg",
        ".jpeg",
        ".png"
    ]
    for s in suffixes:
        if s in str(path):
            return Path(str(path).replace(s, suffix))
    return None

def _find_semantic_file(path: Path):
    semantic_file = _replace_suffix(path, ".semantic")
    if semantic_file is not None and semantic_file.exists():
        return semantic_file
    return None


def _convert_file(path: Path):
    if "-pre.jpg" in str(path):
        return []
    if "," in str(path):
        return []
    image = cv2.imread(str(path))
    if image is None:
        print("Warning: Could not read image", path)
        return []
    margin_top = random.randint(0, 10)
    margin_bottom = random.randint(0, 10)
    preprocessed = add_image_into_tr_omr_canvas(image, margin_top, margin_bottom)
    preprocessed_path = _replace_suffix(path, "-pre.jpg")
    if preprocessed_path is None:
        print("Warning: Unknown extension", path)
        return []
    cv2.imwrite(str(preprocessed_path.absolute()), preprocessed)
    semantic_file = _find_semantic_file(path)
    if semantic_file is None:
        print("Warning: No semantic file found for", path)
        return []
    return [str(preprocessed_path.relative_to(git_root)) + "," + str(semantic_file.relative_to(git_root)) + '\n']


def _convert_dataset(glob_result, index_file: str):
    with open(index_file, 'w') as f:
        file_number = 0
        with multiprocessing.Pool(8) as p:
            for result in p.imap_unordered(_convert_file, glob_result):
                f.writelines(result)
                file_number += 1
                if file_number % 1000 == 0:
                    print(f"Processed {file_number} files") 


def convert_primus_dataset():
    print('Indexing PrIMuS dataset')
    _convert_dataset(Path(primus).rglob('*.png'), primus_train_index)
    print('Indexing PrIMuS Distorted dataset')
    _convert_dataset(Path(primus).rglob('*_distorted.jpg'), primus_distorted_train_index)
    print('Done indexing')


def convert_cpms_dataset():
    print('Indexing CPMS dataset')
    _convert_dataset(Path(cpms).rglob('*.jpg'), cpms_train_index)
    print('Done indexing')

if __name__ == "__main__":
    convert_primus_dataset()
    convert_cpms_dataset()