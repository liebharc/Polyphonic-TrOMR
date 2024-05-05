from staff2score import Staff2Score
import re
import sys
from pathlib import Path
import os
import editdistance

from split_merge_symbols import merge_symbols, convert_alter_to_accidentals
from configs import default_config

def calc_symbol_error_rate_for_list(dataset, config):
    model = Staff2Score(config)
    for sample in dataset:
        img_path, semantic_path = sample.strip().split(",")
        expected = convert_alter_to_accidentals(_load_semantic_file(semantic_path))[0].strip()
        predrhythms, predpitchs, predlifts = model.predict(img_path)
        actual = merge_symbols(predrhythms, predpitchs, predlifts)[0].split("+")
        actual = [symbol for symbol in actual if not symbol.startswith("timeSignature")]  # reference data has no time signature
        expected = expected.split("+")
        actual = sort_chords(actual)
        expected = sort_chords(expected)
        distance = editdistance.eval(expected, actual)
        ser = distance / len(expected)
        print(sample.strip() + "," + str(ser))


def _load_semantic_file(semantic_path):
    with open(semantic_path, 'r') as f:
        return f.readlines()
    
def sort_chords(symbols):
    result = []
    for symbol in symbols:
        result.append(str.join("|", sorted(symbol.split("|"))))
    return result

def load_index(index_file):
    with open(index_file, 'r') as f:
        return f.readlines()

if __name__ == "__main__":
    index_file = sys.argv[1]
    calc_symbol_error_rate_for_list(load_index(index_file), default_config)