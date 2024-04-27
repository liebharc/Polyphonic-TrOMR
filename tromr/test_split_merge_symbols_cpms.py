import unittest

from split_merge_symbols import merge_symbols, split_symbols, convert_alter_to_accidentals

def split_merge(merged):
    # CPMS alraedy seems to use modified semantics
    actuallift, actualpitch, actualrhythm, actualnotes = split_symbols([merged.replace("+", "\t")], convert_to_modified_semantic=False)
    merged_again = merge_symbols(actualrhythm, actualpitch, actuallift)
    return merged_again

class TestMergeSymbolsCpms(unittest.TestCase):

    def test_merge(self):
        actual = split_merge("clef-G2 keySignature-BbM note-C5_quarter. note-B4_eighth note-A4_quarter. note-G4_eighth note-F4#_quarter. note-G4_eighth barline note-F4#_whole note-D5_quarter. note-D5_eighth barline note-D5_quarter. note-C5_eighth note-B4_half note-A4_half barline note-B4_whole note-E5_half barline note-E5_quarter note-A4_quarter note-A4_half note-D5_half barline")
        self.assertEqual(actual, ['clef-G2+keySignature-BbM+note-C5_quarter.+note-B4_eighth+note-A4_quarter.+note-G4_eighth+note-F4#_quarter.+note-G4_eighth+barline+note-F4#_whole+note-D5_quarter.+note-D5_eighth+barline+note-D5_quarter.+note-C5_eighth+note-B4_half+note-A4_half+barline+note-B4_whole+note-E5_half+barline+note-E5_quarter+note-A4_quarter+note-A4_half+note-D5_half+barline'])

if __name__ == '__main__':
    unittest.main()        