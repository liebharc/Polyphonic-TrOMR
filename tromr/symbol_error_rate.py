from staff2score import Staff2Score
import re
import sys
from pathlib import Path
import os
import editdistance

from split_merge_symbols import merge_symbols
from configs import default_config

def calc_symbol_error_rate_for_list(checkpoint_file_path, dataset, config):
    model = Staff2Score(config, checkpoint_file_path)
    all_sers = []
    i = 0
    total = len(dataset)
    for sample in dataset:
        img_path, semantic_path = sample.strip().split(",")
        expected = _load_semantic_file(semantic_path)[0].strip()
        predrhythms, predpitchs, predlifts = model.predict(img_path)
        actual = merge_symbols(predrhythms, predpitchs, predlifts)[0].split("+")
        expected = expected.split("+")
        distance = editdistance.eval(expected, actual)
        ser = distance / len(expected)
        all_sers.append(ser)
        ser = round(100 * ser)
        ser_avg = round(100 * sum(all_sers) / len(all_sers))
        i += 1
        if i % 10 == 0:
            print(actual, "vs", expected)
        percentage = round(i / total * 100)
        print(f"Progress: {percentage}%, SER: {ser}%, SER avg: {ser_avg}%")
    ser_avg = round(100 * sum(all_sers) / len(all_sers))
    print(f"Done, SER avg: {ser_avg}%")

    result_file = checkpoint_file_path.replace(".pth", "_ser.txt")
    with open(result_file, 'w') as f:
        f.write(f"SER avg: {ser_avg}%\n")


def _load_semantic_file(semantic_path):
    with open(semantic_path, 'r') as f:
        return f.readlines()

if __name__ == "__main__":
    checkpoint_file = sys.argv[1]
    script_location = os.path.dirname(os.path.realpath(__file__))
    data_set_location = os.path.join(script_location, '..', 'CreativeCommons')
    staff_files = Path(data_set_location).rglob('staff-*.jpg')
    index = []
    for staff_file in staff_files:
        semantic_file = staff_file.with_suffix('.semantic')
        index.append(str(staff_file) + "," + str(semantic_file).strip())
    calc_symbol_error_rate_for_list(checkpoint_file, index, default_config)