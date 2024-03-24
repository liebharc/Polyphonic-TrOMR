import re

from circle_of_fifths import KeyTransformation, key_signature_to_circle_of_fifth

def merge_symbols(predrhythms, predpitchs, predlifts):
    merges = []
    for i in range(len(predrhythms)):
        predlift = predlifts[i]
        predpitch = predpitchs[i]
        predrhythm = predrhythms[i]
        
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
        if "##" in symbol:
            return "#"  # We have no support for double accidentals right now
        elif "#" in symbol:
            return "#"
        elif "bb" in symbol:
            return "b"  # We have no support for double accidentals right now
        elif "b" in symbol:
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

def split_symbols(merged):
    predlifts = []
    predpitchs = []
    predrhythms = []
    prednotes = []
    for line in range(len(merged)):
        predlift = []
        predpitch = []
        predrhythm = []
        prednote = []
        key = KeyTransformation(0)
        for symbols in re.split("\s+", merged[line]):
            for symbol in re.split("(\|)", symbols):
                if symbol.startswith("keySignature"):
                    key = KeyTransformation(key_signature_to_circle_of_fifth(symbol.split("-")[-1]))
                if symbol == "barline":
                    key = key.reset_at_end_of_measure()

                if symbol == "tie":
                    continue
                elif symbol == "|":
                    predrhythm.append("|")
                    predpitch.append("nonote")
                    predlift.append("nonote")
                    prednote.append("nonote")
                else:
                    pitch = _symbol_to_pitch(symbol)
                    predpitch.append(pitch)
                    predrhythm.append(_symbol_to_rhythm(symbol))
                    prednote.append(_symbol_to_note(symbol))
                    alter = _get_alter(symbol)
                    if alter is not None:
                        note_name = pitch[5]
                        alter = key.add_accidental(note_name, alter)
                        predlift.append(_alter_to_lift(alter))
                    else:
                        predlift.append("nonote")
        predlifts.append(predlift)
        predpitchs.append(predpitch)
        predrhythms.append(predrhythm)
        prednotes.append(prednote)
    return predlifts, predpitchs, predrhythms, prednotes