import re

from circle_of_fifths import KeyTransformation, NoKeyTransformation, key_signature_to_circle_of_fifth

def merge_symbols(predrhythms, predpitchs, predlifts):
    merges = []
    for i in range(len(predrhythms)):
        predlift = predlifts[i]
        predpitch = predpitchs[i]
        predrhythm = predrhythms[i]

        if len(predrhythm) == 0:
            merges.append("")
            continue
        
        merge = predrhythm[0] + '+'
        for j in range(1, len(predrhythm)):
            if predrhythm[j] == "|":
                merge = merge[:-1]+predrhythm[j]
            elif "note" in predrhythm[j]:
                lift = ""
                if predlift[j] in ("lift_##", "lift_#", "lift_bb", "lift_b", "lift_N",):
                    lift = predlift[j].split("_")[-1]
                merge += predpitch[j]+lift+"_"+predrhythm[j].split('note-')[-1]+"+"
            else:
                merge += predrhythm[j]+"+"
        merges.append(merge[:-1])
    return merges

def _get_alter(symbol):
    if symbol.startswith("note") or symbol.startswith("gracenote"):
        note = symbol.split("_")[0]
        if "##" in note:
            return "#"  # We have no support for double accidentals right now
        elif "#" in note:
            return "#"
        elif "bb" in note:
            return "b"  # We have no support for double accidentals right now
        elif "b" in note:
            return "b"
        return ""
    return None

def _alter_to_lift(symbol):
    if symbol == "#":
        return "lift_#"
    elif symbol == "b":
        return "lift_b"
    elif symbol == "0":
        return "lift_N"
    else:
        return "lift_null"
    
def _replace_accidentals( notename):
    notename = notename.replace("#", "")
    notename = notename.replace("b", "")
    return notename

def _symbol_to_pitch(symbol):
    if symbol.startswith("note") or symbol.startswith("gracenote"):
        without_duration = symbol.split("_")[0]
        notename = without_duration.split("-")[1]
        notename = _replace_accidentals(notename)
        notename = "note-" + notename
        return notename
    return "nonote"

def _add_dots(duration):
    # TrOMR only allows one dot
    #number_of_dots_in_duration = duration.count(".")
    #return "".join(["." for _ in range(number_of_dots_in_duration)])
    if "." in duration:
        return "."
    return ""

def _translate_duration(duration):
    duration = duration.replace("second", "breve")
    duration = duration.replace("double", "breve")
    duration = duration.replace("quadruple", "breve")
    duration = duration.replace("thirty", "thirty_second")
    duration = duration.replace("sixty", "sixty_fourth")
    duration = duration.replace("hundred", "hundred_twenty_eighth")
    duration = duration.replace(".", "")  # We add dots later again
    return duration

def _symbol_to_rhythm(symbol):
    if symbol.startswith("note") or symbol.startswith("gracenote"):
        note = "note-" + _translate_duration(symbol.split("_")[1])
        return note + _add_dots(symbol)
    symbol = symbol.replace("rest-quadruple_whole", "multirest-2")
    symbol = symbol.replace("_fermata", "")
    symbol = symbol.replace(".", "")  # We add dots later again
    multirest_match = re.match(r"(rest-whole|multirest-)(\d+)", symbol)
    if multirest_match:
        rest_length = int(multirest_match[2])
        # Some multirests don't exist in the rhtythm tokenizer, for now it's good enough to just recognize them as any multirest
        if rest_length <= 1:
            return "rest-whole"
        if rest_length > 50:
            return "multirest-50"
        symbol = "multirest-" + str(rest_length)
    symbol = symbol.replace("timeSignature-2/3", "timeSignature-2/4")
    symbol = symbol.replace("timeSignature-3/6", "timeSignature-3/8")
    symbol = symbol.replace("timeSignature-8/12", "timeSignature-8/16")
    symbol = symbol.replace("timeSignature-2/48", "timeSignature-2/32")
    return symbol + _add_dots(symbol)


def _symbol_to_note(symbol):
    if symbol.startswith("note") or symbol.startswith("gracenote"):
        return "note"
    return "nonote"

def _note_name_to_sortable(note_name):
    if note_name == "C":
        return 0
    if note_name == "D":
        return 1
    if note_name == "E":
        return 2
    if note_name == "F":
        return 3
    if note_name == "G":
        return 4
    if note_name == "A":
        return 5
    if note_name == "B":
        return 6
    return 7

def _note_name_and_octave_to_sortable(note_name_with_octave):
    note_name = note_name_with_octave[:-1]
    octave = int(note_name_with_octave[-1])
    return _note_name_to_sortable(note_name) + octave * 7

def _pitch_name_to_sortable(pitch_name):
    note_name = pitch_name.split("-")[-1]
    return _note_name_and_octave_to_sortable(note_name)

def _sort_by_pitch(lifts, pitches, rhythms, notes):
    lifts = lifts.copy()
    pitches = pitches.copy()
    rhythms = rhythms.copy()
    notes = notes.copy()

    def swap(i, j):
        lifts[i], lifts[j] = lifts[j], lifts[i]
        pitches[i], pitches[j] = pitches[j], pitches[i]
        rhythms[i], rhythms[j] = rhythms[j], rhythms[i]
        notes[i], notes[j] = notes[j], notes[i]

    for i in range(len(pitches)):
        if pitches[i] == "nonote":
            continue
        for j in range(i+1, len(pitches)):
            if pitches[j] == "nonote":
                continue
            if _pitch_name_to_sortable(pitches[i]) < _pitch_name_to_sortable(pitches[j]):
                swap(i, j)
    return lifts, pitches, rhythms, notes

def convert_alter_to_accidentals(merged):
    """
    Moves alter information into accidentals.
    For example: 
    """
    all_results = []
    for line in range(len(merged)):
        key = KeyTransformation(0)
        line_result = []
        for symbols in re.split("\s+", merged[line].replace("+", " ")):
            symbol_result = []
            for symbol in re.split("(\|)", symbols):
                if symbol.startswith("keySignature"):
                    key = KeyTransformation(key_signature_to_circle_of_fifth(symbol.split("-")[-1]))
                    symbol_result.append(symbol)
                elif symbol == "barline":
                    key = key.reset_at_end_of_measure()
                    symbol_result.append(symbol)
                elif symbol.startswith("note") or symbol.startswith("gracenote"):
                    pitch = _symbol_to_pitch(symbol)
                    alter = _get_alter(symbol)
                    note_name = pitch[5]
                    accidental = key.add_accidental(note_name, alter).replace("0", "N")
                    symbol = symbol.replace(alter + "_", accidental + "_")
                    symbol_result.append(symbol)
                elif symbol != "|":
                    symbol_result.append(symbol)
                    
            if len(symbol_result) > 0:
                line_result.append(str.join("|", symbol_result))
        all_results.append(str.join("+", line_result))
    return all_results


def split_symbols(merged, convert_alter=False):
    predlifts = []
    predpitchs = []
    predrhythms = []
    prednotes = []
    for line in range(len(merged)):
        predlift = []
        predpitch = []
        predrhythm = []
        prednote = []
        key = KeyTransformation(0) if convert_alter else NoKeyTransformation()
        for symbols in re.split("\s+", merged[line]):
            symbollift = []
            symbolpitch = []
            symbolrhythm = []
            symbolnote = []
            for symbol in re.split("(\|)", symbols):
                if symbol.startswith("keySignature"):
                    if convert_alter:
                        key = KeyTransformation(key_signature_to_circle_of_fifth(symbol.split("-")[-1]))
                if symbol == "barline":
                    key = key.reset_at_end_of_measure()

                if symbol == "tie":
                    continue
                elif symbol == "|":
                    symbolrhythm.append("|")
                    symbolpitch.append("nonote")
                    symbollift.append("nonote")
                    symbolnote.append("nonote")
                else:
                    pitch = _symbol_to_pitch(symbol)
                    symbolpitch.append(pitch)
                    symbolrhythm.append(_symbol_to_rhythm(symbol))
                    symbolnote.append(_symbol_to_note(symbol))
                    alter = _get_alter(symbol)
                    if alter is not None:
                        note_name = pitch[5]
                        alter = key.add_accidental(note_name, alter)
                        symbollift.append(_alter_to_lift(alter))
                    else:
                        symbollift.append("nonote")
            if len(symbolpitch) > 0:
                symbollift, symbolpitch, symbolrhythm, symbolnote = _sort_by_pitch(symbollift, symbolpitch, symbolrhythm, symbolnote)
                predpitch += symbolpitch
                predrhythm += symbolrhythm
                prednote += symbolnote
                predlift += symbollift
        predlifts.append(predlift)
        predpitchs.append(predpitch)
        predrhythms.append(predrhythm)
        prednotes.append(prednote)
    return predlifts, predpitchs, predrhythms, prednotes