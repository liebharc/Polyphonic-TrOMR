import os
import sys
import pathlib

from split_merge_symbols import split_symbols

def compare_semantic_with_acnostic(semantic_path: str):
    agnostic_path = semantic_path.replace(".semantic", ".agnostic")
    if not os.path.exists(agnostic_path):
        return 0
    with open(semantic_path, 'r') as file:
        semantic = file.readline().strip().replace("+", " ")
    lifts, _pitch, _rhythm, _notes = split_symbols([semantic])
    semantic_sharps = lifts[0].count("lift_#")
    semantic_flats = lifts[0].count("lift_b")
    semantic_naturals = lifts[0].count("lift_N")
    with open(agnostic_path, 'r') as file:
        acnostic = file.readline().strip().split()

    acnostic_sharps = 0
    acnostic_flats = 0
    acnostic_naturals = 0
    last_symbol = None
    for symbol in acnostic:
        if symbol.startswith("note.") and last_symbol and last_symbol.startswith("accidental."):
            height_note = symbol.split("-")[-1]
            height_accidental = last_symbol.split("-")[-1]
            if height_note == height_accidental:
                if last_symbol.startswith("accidental.sharp"):
                    acnostic_sharps += 1
                elif last_symbol.startswith("accidental.flat"):
                    acnostic_flats += 1
                elif last_symbol.startswith("accidental.natural"):
                    acnostic_naturals += 1


        last_symbol = symbol
    number_of_errors = abs(semantic_sharps - acnostic_sharps) + abs(semantic_flats - acnostic_flats) + abs(semantic_naturals - acnostic_naturals)
    return number_of_errors


if __name__ == "__main__":

    script_location = os.path.dirname(os.path.realpath(__file__))

    primus_index = os.path.join(script_location, '..', 'Corpus', 'index.txt')

    with open(primus_index, 'r') as file:
        primus_entries = file.readlines()

    total_number_of_errors = 0

    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    for entry in primus_entries:
        entry = entry.strip()
        image_path, semantic_path = entry.strip().split(",")
        number_of_errors = compare_semantic_with_acnostic(semantic_path)
        if number_of_errors > 0:
            print(entry)
        total_number_of_errors += number_of_errors
    eprint("Total number of errors:", total_number_of_errors, total_number_of_errors / len(primus_entries))
        