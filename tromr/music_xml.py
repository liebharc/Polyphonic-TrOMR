import sys
from pathlib import Path
import os
import shutil
import json
import re

import xmltodict
from circle_of_fifths import KeyTransformation, NoKeyTransformation, circle_of_fifth_to_key_signature

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

def _ensure_list(obj):
    if type(obj) is list:
        return obj
    return [obj]

def _count_dots(note):
    if "dot" not in note:
        return ""
    return "." * len(_ensure_list(note["dot"]))

def _music_part_to_semantic(part, convert_alter=False):
    try:
        semantic = []
        for measure in _ensure_list(part["measure"]):
            chord = []
            key = KeyTransformation(0) if convert_alter else NoKeyTransformation()
            if "attributes" in measure:
                for attribute in _ensure_list(measure["attributes"]):
                    if "clef" in attribute:
                        clef = _ensure_list(attribute["clef"])[0]
                        semantic.append("clef-" + clef["sign"] + clef["line"])
                    if "key" in attribute:
                        semantic.append("keySignature-" + circle_of_fifth_to_key_signature(int(attribute["key"]["fifths"])))
                        if convert_alter:
                            key = KeyTransformation(int(attribute["key"]["fifths"]))
                    if "time" in attribute:
                        semantic.append("timeSignature-" + attribute["time"]["beats"] + "/" + attribute["time"]["beat-type"])
            if "note" in measure:
                for note in  _ensure_list(measure["note"]):
                    if not "chord" in note:
                        if len(chord) > 0:
                            # Flush the previous chord
                            semantic.append("|".join(chord))
                            chord = []
                    if "rest" in note:
                        dot = _count_dots(note)
                        if note["rest"] and "@measure" in note["rest"]:
                            semantic.append("rest-whole" + dot)
                        else:
                            semantic.append("rest-" + _translate_duration(note["type"]) + dot)
                    if "pitch" in note:
                        key.add_accidental(note["pitch"]["step"], _get_alter(note["pitch"]))
                        alter = _get_alter(note["pitch"])
                        chord.append("note-" + note["pitch"]["step"] + alter + note["pitch"]["octave"] + "_" + _translate_duration(note["type"]) + _count_dots(note))
                        
            if len(chord) > 0:
                # Flush the last chord
                semantic.append("|".join(chord))
            semantic.append("barline")
            key = key.reset_at_end_of_measure()
        return semantic
    except Exception as e:
        print("Failure at ", part)
        raise e
    
def music_xml_to_semantic(path):
    result = []
    with open(path) as f:
        musicxml = xmltodict.parse(f.read())
        parts = _ensure_list(musicxml['score-partwise']["part"])
        for part in parts:
            semantic = _music_part_to_semantic(part)
            result.append(semantic)
    return result

def _write_job_file(folder, xml, formats):
    jobs = []
    file_mapping = {}
    for format in formats:
        out_file_name  = os.path.join(folder, "music." + format)
        job = {
            "in": str(xml),
            "out": out_file_name
        }
        jobs.append(job)
        file_mapping[format] = out_file_name
    job_file = os.path.join(folder, "job.json")
    with open(job_file, "w") as f:
        json.dump(jobs, f)
    return job_file, file_mapping

def _run_musescore(job_file):
    script_location = os.path.dirname(os.path.realpath(__file__))
    muse_score_path = os.path.join(script_location, "..", "MuseScore.AppImage")
    if not os.path.exists(muse_score_path):
        print("MuseScore not found at ", muse_score_path)
        sys.exit(1)
    command = f"{muse_score_path} --appimage-extract-and-run -j {job_file} --force"
    ret_val = os.system(command)
    if ret_val != 0:
        print("Failed to run MuseScore")
        sys.exit(1)

def _read_all_lines(svg_path):
    with open(svg_path) as f:
        svg = f.read()
        lines = svg.split("\n")
        return lines
    
def _arrange_bar_lines(points):
    staffs = []
    for point in points:
        any_matching_x = False
        for line in staffs:
            if abs(line[0][1] - point[1]) < 1:
                line.append(point)
                any_matching_x = True
                break
        if not any_matching_x:
            staffs.append([point])
    for line in staffs:
        line.sort(key=lambda x: x[0])
    staffs.sort(key=lambda x: x[0][1])
    number_of_bar_lines = []
    for line in staffs:
        number_of_bar_lines.append(len(line))
    return number_of_bar_lines
    
def _parse_svg(svg_path):
    lines = _read_all_lines(svg_path)
    bar_lines = [line for line in lines if "BarLine" in line]
    bar_line_start_points = []
    for bar_line in bar_lines:
        match = re.match(r".*stroke-width=\"([^\"]+)\".*points=\"([^\"]+)\".*", bar_line)
        if match:
            stroke_width = float(match.group(1))
            is_double_bar = stroke_width > 10
            if is_double_bar:
                continue
            starting_point = [float(v) for v in match.group(2).split(" ")[0].split(",")]
            bar_line_start_points.append(starting_point)
    number_of_barlines_per_staff = _arrange_bar_lines(bar_line_start_points)
    return number_of_barlines_per_staff
    
def _find_bar_lines(svg_path):
    folder = os.path.dirname(svg_path)
    svg_files = list(Path.glob(Path(folder), "*.svg"))

    # Sort them by file id
    svg_files.sort(key=lambda x: int(str(x).split("-")[1].split(".")[0]))
    pages = []
    for path in svg_files:
        pages.append(_parse_svg(path))
    return pages

def _find_measures_per_voice(pages, semantic):
    bar_lines_of_all_pages = [x for xs in pages for x in xs]
    voices = []
    for i, voice in enumerate(semantic):
        bar_lines_of_this_voice = []
        j = i
        while j < len(bar_lines_of_all_pages):
            bar_lines_of_this_voice.append(bar_lines_of_all_pages[j])
            j += len(semantic)
        bar_lines_in_xml = voice.count("barline")
        print(path, "voice", i + 1, "has", sum(bar_lines_of_this_voice), "bar lines in the SVG accross", len(bar_lines_of_this_voice), "staffs and", bar_lines_in_xml, "in the XML")
        if sum(bar_lines_of_this_voice) == bar_lines_in_xml + 1:
            print("> OK (no initial bar line)")
            measures_per_line = bar_lines_of_this_voice.copy()
            measures_per_line[-1] -= 1
            voices.append(measures_per_line)
        elif sum(bar_lines_of_this_voice) == bar_lines_in_xml:
            measures_per_line = bar_lines_of_this_voice.copy()
            voices.append(measures_per_line)
            print("> OK (no initial bar line and final)")
        elif sum(bar_lines_of_this_voice) - len(bar_lines_of_this_voice) == bar_lines_in_xml + 1:
            measures_per_line = [c - 1 for c in bar_lines_of_this_voice]
            measures_per_line[-1] -= 1
            voices.append(measures_per_line)
            print("> OK")
        elif sum(bar_lines_of_this_voice) - len(bar_lines_of_this_voice) == bar_lines_in_xml:
            measures_per_line = [c - 1 for c in bar_lines_of_this_voice]
            voices.append(measures_per_line)
            print("> OK (no final bar line)")
        else:
            print("> Failed to align SVG layout with Music XML data")
            return []
    return voices

def _split_semantic(semantic, measures_per_line):
    sublists = []
    sublist = []
    measure_counter = 0
    line_counter = 0

    for i, symbol in enumerate(semantic):
        sublist.append(symbol)
        if symbol == "barline":
            measure_counter += 1
            if measure_counter == measures_per_line[line_counter]:
                sublists.append(sublist)
                sublist = []
                measure_counter = 0
                line_counter += 1
        

    # Add the last sublist if it's not empty
    if sublist:
        sublists.append(sublist)

    return sublists

def _write_semantic_file(filename, semantic):
    with open(filename, "w") as f:
        f.write(str.join("+", semantic))

if __name__ == "__main__":
    folder = sys.argv[1]
    
    for path in Path(folder).glob('*.mxl'):
        print(path)
        basename = str(path).replace(".mxl", "")
        dirname = os.path.dirname(path)
        file_folder = basename
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        job_file, file_mapping = _write_job_file(file_folder, path, ["svg", "musicxml"])
        _run_musescore(job_file)
        voices_semantic = music_xml_to_semantic(file_mapping["musicxml"])
        pages = _find_bar_lines(file_mapping["svg"])
        voices = _find_measures_per_voice(pages, voices_semantic)
        # Delete existing semantic files
        for file in Path.glob(Path(file_folder), "*.semantic"):
            os.remove(file)
        for i, voice in enumerate(voices):
            semantic = voices_semantic[i]
            symbols_per_measure = []
            semantic_staffs = _split_semantic(semantic, voice)
            clef = semantic_staffs[0][0]
            key = semantic_staffs[0][1]
            for semantic_staff in semantic_staffs[1:]:
                semantic_staff.insert(0, key)
                semantic_staff.insert(0, clef)
            for j, semantic_staff in enumerate(semantic_staffs):
                staff_number = j * len(voices) + i
                _write_semantic_file(os.path.join(file_folder, f"staff-{staff_number}.semantic"), semantic_staff)
