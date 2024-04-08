
import os
from pathlib import Path
import random
import cv2
import multiprocessing

from image_processing import add_image_into_tr_omr_canvas


script_location = os.path.dirname(os.path.realpath(__file__))
git_root = os.path.join(script_location, '..')
diw_root = os.path.join(git_root, 'diw', '5k', 'img')
diw_train_index = os.path.join(git_root, 'diw', 'index.txt')


def _convert_file(file: Path):
    if "-pre.png" in str(file):
        return []
    image = cv2.imread(str(file))
    
    image_width = image.shape[1]
    max_width = 1280

    width = min(max_width, image_width)
    height = random.randint(32, 128)

    top = (image.shape[0] - height) // 2
    bottom = top + height
    left = (image.shape[1] - width) // 2
    right = left + width
    cropped = image[top:bottom, left:right, :]

    target_path = str(file).replace('.png', '-pre.png')
    cv2.imwrite(target_path, add_image_into_tr_omr_canvas(cropped))
    return [f'{target_path},nosymbols\n']


def _convert_dataset(glob_result, index_file: str):
    with open(index_file, 'w') as f:
        file_number = 0
        with multiprocessing.Pool(8) as p:
            for result in p.imap_unordered(_convert_file, glob_result):
                f.writelines(result)
                file_number += 1
                if file_number % 1000 == 0:
                    print(f"Processed {file_number} files") 


def convert_diw_dataset():
    print('Indexing documents in the wild dataset')
    _convert_dataset(Path(diw_root).rglob('*[!-pre].png'), diw_train_index)
    print('Done indexing')


if __name__ == '__main__':
    convert_diw_dataset()