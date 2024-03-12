import os
import sys
import cv2
from pathlib import Path

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

def _split_staff_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', gray)

    cv2.waitKey(0)
    pass

def _musix_xml_to_semantic(path):
    pass

def convert_grandstaff():
    if os.path.exists(grandstaff_train_index):
        return
    
    print('Indexing Grandstaff dataset')
    with open(grandstaff_train_index, 'w') as f:
        for path in Path(grandstaff_root).rglob('*.krn'):
            image_file = str(path).replace(".krn", ".jpg")
            distored = str(path).replace(".krn", "_distorted.jpg")
            musicxml = str(path).replace(".krn", ".musicxml")
            semantic = str(path).replace(".krn", ".semantic")
            result = os.system(f"{hum2xml} {path} > {musicxml}")
            if result != 0:
                print(f"Failed to convert {path}")
                continue
            _split_staff_image(image_file)
            f.write(str(Path(image_file).relative_to(git_root)) + '\n')
    print('Done indexing')

if __name__ == "__main__":
    convert_grandstaff()