from typing import List

circle_of_fifth_notes_positive = ["F", "C", "G", "D", "A", "E", "B"]
circle_of_fifth_notes_negative = list(reversed(circle_of_fifth_notes_positive))

def get_circle_of_fifth_notes(circle_of_fifth: int) -> List[str]:
    if circle_of_fifth >= 0:
        return circle_of_fifth_notes_positive[0:circle_of_fifth]
    else:
        return circle_of_fifth_notes_negative[0:abs(circle_of_fifth)]
    

class KeyTransformation:

    def __init__(self, circle_of_fifth: int):
        self.circle_of_fifth = circle_of_fifth
        self.sharps = []
        self.flats = []
        if circle_of_fifth > 0:
            self.sharps = circle_of_fifth_notes_positive[0:circle_of_fifth]
        elif circle_of_fifth < 0:
            self.flats = circle_of_fifth_notes_negative[0:abs(circle_of_fifth)]

    def add_accidental(self, note: str, accidental: str) -> str:
        if accidental == "#":
            self.sharps.append(note)
        elif accidental == "b":
            self.flats.append(note)
        elif accidental == "0":
            if note in self.sharps:
                self.sharps.remove(note)
            if note in self.flats:
                self.flats.remove(note)
        return note

    def get_alter(self, note: str) -> int:
        if note in self.sharps:
            return "#"
        elif note in self.flats:
            return "b"
        else:
            return ""
        
    def reset_at_end_of_measure(self):
        return KeyTransformation(self.circle_of_fifth)