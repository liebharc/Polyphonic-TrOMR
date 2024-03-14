import re

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

def _symbol_to_lift(symbol):
    # TODO this is incorrect as the semantic encoding handles accidentals differently
    if symbol.startswith("note") or symbol.startswith("gracenote"):
        if "##" in symbol:
            return "lift_##"
        elif "#" in symbol:
            return "lift_#"
        elif "bb" in symbol:
            return "lift_bb"
        elif "b" in symbol:
            return "lift_b"
        return "lift_null"
    return "nonote"
    
def _replace_accidentals( notename):
    # TODO this is incorrect as the semantic encoding handles accidentals differently
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
    duration = duration.replace(".", "")  # We add dots later again
    return duration

def _symbol_to_rhythm(symbol):
    if symbol.startswith("note") or symbol.startswith("gracenote"):
        note = "note-" + _translate_duration(symbol.split("_")[1])
        return note + _add_dots(symbol)
    symbol = symbol.replace("multirest-1", "rest-whole")
    symbol = symbol.replace("rest-quadruple_whole", "multirest-2")
    symbol = symbol.replace("rest-whole0", "rest-whole")
    symbol = symbol.replace("_fermata", "")
    if re.match(r"rest-whole(\d+)", symbol) or re.match(r"multirest-(\d+)", symbol):
        return "multirest-2"  # Some multirests don't exist in the rhtythm tokenizer, for now it's good enough to just recognize them as any multirest
    symbol = symbol.replace("timeSignature-2/3", "timeSignature-2/4")
    return symbol


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
        for symbols in re.split("\s+", merged[line]):
            for symbol in re.split("(\|)", symbols):
                if symbol == "tie":
                    continue
                elif symbol == "|":
                    predrhythm.append("|")
                    predpitch.append("nonote")
                    predlift.append("nonote")
                    prednote.append("nonote")
                else:
                    predpitch.append(_symbol_to_pitch(symbol))
                    predlift.append(_symbol_to_lift(symbol))
                    predrhythm.append(_symbol_to_rhythm(symbol))
                    prednote.append(_symbol_to_note(symbol))
        predlifts.append(predlift)
        predpitchs.append(predpitch)
        predrhythms.append(predrhythm)
        prednotes.append(prednote)
    return predlifts, predpitchs, predrhythms, prednotes