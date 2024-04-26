import sys
import numpy as np
import os
import cv2
import re
from termcolor import colored
from split_merge_symbols import merge_symbols, convert_alter_to_accidentals, split_symbols, merge_symbols

index_file_name = sys.argv[1]
number_of_samples_per_iteration = int(sys.argv[2])

index_file = open(index_file_name, 'r')
index_lines = index_file.readlines()
index_file.close()

np.random.shuffle(index_lines)

def print_color(text, highlights, color):
    words = text.split()
    for word in words:
        if any([highlight in word for highlight in highlights]):
            print(colored(word, color), end=" ")
        else:
            print(word, end=" ")
    print()

while True:
    batch = []
    for i in range(number_of_samples_per_iteration):
        if len(index_lines) == 0:
            break
        batch.append(index_lines.pop())

    if len(batch) == 0:
        break

    images = None
    print()
    print()
    print()
    print("==========================================")
    print()
    for line in batch:
        image_path, semantic_path = line.strip().split(",")
        agnostic_path = semantic_path.replace(".semantic", ".agnostic")
        image = cv2.imread(image_path)
        with open(semantic_path, 'r') as file:
            original_semantic = file.readline().strip().replace("+", " ")
        if os.path.exists(agnostic_path):
            with open(agnostic_path, 'r') as file:
                original_agnostic = file.readline().strip().replace("+", " ")
        else:
            original_agnostic = agnostic_path
        predlifts, predpitchs, predryhthms, _prednotes = split_symbols([original_semantic])
        semantic = merge_symbols(predryhthms, predpitchs, predlifts)
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        print()
        print(">>> " + image_path)
        print_color("Lifts " + str.join(" ", predlifts[0]), ["lift_b", "lift_#", "lift_N"], "yellow")
        print_color("Semantic " + semantic[0].replace("+", " "), ["barline", "#", "N", "b"], "red")
        print_color("Original " + original_semantic, ["barline", "#", "N", "b"], "green")
        print_color("Agnostic " + original_agnostic, ["barline", "accidental.sharp", "accidental.flat", "accidental.natural"], "cyan")
    cv2.imshow("Images", images)
    escKey = 27
    if cv2.waitKey(0) == escKey:
        break