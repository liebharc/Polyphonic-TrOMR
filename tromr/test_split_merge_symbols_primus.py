import unittest

from split_merge_symbols import merge_symbols, split_symbols, convert_alter_to_accidentals

def split_merge(merged):
    actuallift, actualpitch, actualrhythm, actualnotes = split_symbols([merged.replace("+", "\t")])
    merged_again = merge_symbols(actualrhythm, actualpitch, actuallift)
    return merged_again

class TestMergeSymbolsPrimus(unittest.TestCase):

    def test_merge(self):
        actual = split_merge("clef-C1 timeSignature-C/ note-G4_double_whole note-G4_whole note-A4_whole. note-G4_half note-G4_half note-F#4_half note-G4_double_whole note-G4_half")
        self.assertEqual(actual, ["clef-C1+timeSignature-C/+note-G4_breve+note-G4_whole+note-A4_whole.+note-G4_half+note-G4_half+note-F4#_half+note-G4_breve+note-G4_half"])

if __name__ == '__main__':
    unittest.main()        