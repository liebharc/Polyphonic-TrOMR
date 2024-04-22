import sys
import os
import split_merge_symbols
from collections import defaultdict
import json

script_location = os.path.dirname(os.path.realpath(__file__))

rhythm_tokenizer_path = os.path.join(script_location, 'workspace', 'tokenizers', 'tokenizer_rhythm.json')
pitch_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_pitch.json')
note_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_note.json')
lift_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_lift.json')

for filepath in sys.argv[1:]:
    with open(filepath, 'r') as file:
        lines = file.readlines()

def add_all(dict, symbols):
    for symbol in symbols:
        dict[symbol] += 1

def calc_norm_value(dict):
    dict = dict.copy()
    dict.pop("nonote", None)
    return max(dict.values())

def calc_weight(norm_value, value):
    return min(round(norm_value / value), 20)

def print_stats(dict):
    sum_of_all = sum(dict.values())
    norm_value = calc_norm_value(dict)
    for key, value in dict.items():
        weight = calc_weight(norm_value, value) if key != "nonote" else 1
        print(f"{key}: {value} ({value/sum_of_all*100:.2f}%) weight: {weight}")

def generate_weights(dict, tokenizer_path):
    tokenizer_config = json.load(open(tokenizer_path,'r'))
    norm_value = calc_norm_value(dict)
    vocab = tokenizer_config['model']['vocab']
    result = []
    for key in vocab:
        if key not in dict or key == "nonote":
            value = 1
        else:
            value = calc_weight(norm_value, dict[key])
        result.append(value)
    return result
    
lift_stats = defaultdict(int)
pitch_stats = defaultdict(int)
rhythm_stats = defaultdict(int)

for line in lines:
    line = line.strip()
    if line == "":
        continue
    img_path, semantic_path = line.split(",")
    if not os.path.exists(semantic_path):
        continue
    with open(semantic_path, 'r') as file:
        semantic = file.readline().strip()
    predlifts, predpitchs, predrhythms, _prednotes = split_merge_symbols.split_symbols([semantic])
    add_all(lift_stats, predlifts[0])
    add_all(pitch_stats, predpitchs[0])
    add_all(rhythm_stats, predrhythms[0])

print("Lift stats:")
print_stats(lift_stats)
print(generate_weights(lift_stats, lift_tokenizer_path))
print()
print("Pitch stats:")
print_stats(pitch_stats)
print(generate_weights(pitch_stats, pitch_tokenizer_path))
print()
print("Rhythm stats:")
print_stats(rhythm_stats)
print(generate_weights(rhythm_stats, rhythm_tokenizer_path))