import unittest

from split_merge_symbols import merge_symbols, split_symbols

predlift = [['nonote', 'nonote', 'nonote', 'lift_null', 'nonote', 'lift_null', 'nonote', 'lift_null', 'nonote', 'lift_null', 'nonote', 'lift_null', 'lift_null', 'lift_null', 'nonote', 'lift_#', 'nonote', 'lift_null', 'nonote', 'lift_null']]
predpitch = [['nonote', 'nonote', 'nonote', 'note-C4', 'nonote', 'note-F4', 'nonote', 'note-G4', 'nonote', 'note-B4', 'nonote', 'note-B4', 'note-C5', 'note-D5', 'nonote', 'note-E4', 'nonote', 'note-G4', 'nonote', 'note-C5']]
predryhthm = [['clef-G2', 'keySignature-EM', 'timeSignature-6/8', 'note-half.', 'barline', 'note-half.', 'barline', 'note-half.', 'barline', 'note-half.', 'barline', 'note-half', 'note-eighth', 'note-eighth', 'barline', 'note-eighth', '|', 'note-eighth', '|', 'note-eighth']]
prednotes = [['nonote', 'nonote', 'nonote', 'note', 'nonote', 'note', 'nonote', 'note', 'nonote', 'note', 'nonote', 'note', 'note', 'note', 'nonote', 'note', 'nonote', 'note', 'nonote', 'note']]
merged = ['clef-G2+keySignature-EM+timeSignature-6/8+note-C4_half.+barline+note-F4_half.+barline+note-G4_half.+barline+note-B4_half.+barline+note-B4_half+note-C5_eighth+note-D5_eighth+barline+note-E4#_eighth|note-G4_eighth|note-C5_eighth']

class TestMergeSymbols(unittest.TestCase):

    def test_merge(self):
        actual = merge_symbols(predryhthm, predpitch, predlift)
        self.assertEqual(actual, merged)

    def test_split(self):
        # Replace the + with \t as this is what the input provides
        actuallift, actualpitch, actualrhythm, actualnotes = split_symbols([merged[0].replace("+", "\t")])
        self.assertEqual(actualrhythm, predryhthm)
        self.assertEqual(actuallift, predlift)
        self.assertEqual(actualpitch, predpitch)
        self.assertEqual(actualnotes, prednotes)

    def test_split_restores_accidentals(self):
        """
        The semantic encoding doesn't tell us which accidentals are present in the image. The best we can do is to restore
        this information from the lift symbols and the key information.
        """

        merged_accidentals = ["clef-G2 keySignature-FM timeSignature-4/4 rest-sixteenth note-A3_sixteenth note-C4_sixteenth note-F4_sixteenth note-A4_sixteenth note-C4_sixteenth note-F4_sixteenth rest-sixteenth note-A3_sixteenth note-A3_sixteenth note-C4_sixteenth note-F4_sixteenth note-A4_sixteenth note-C4_sixteenth note-F4_sixteenth rest-sixteenth note-A3_sixteenth rest-sixteenth note-A3_quarter.. note-A3_quarter.. barline rest-sixteenth note-C4_sixteenth note-Eb4_sixteenth note-F4_sixteenth note-C5_sixteenth note-Eb4_sixteenth note-F4_sixteenth rest-sixteenth note-C4_sixteenth note-C4_sixteenth note-D4_sixteenth note-F#4_sixteenth note-C5_sixteenth note-D4_sixteenth note-F#4_sixteenth rest-sixteenth note-C4_sixteenth rest-sixteenth note-C4_quarter.. note-C4_quarter.. barline rest-sixteenth note-C4_sixteenth note-D4_sixteenth note-A4_sixteenth note-C5_sixteenth note-D4_sixteenth note-A4_sixteenth rest-sixteenth note-C4_sixteenth note-Bb3_sixteenth note-D4_sixteenth note-G4_sixteenth note-Bb4_sixteenth note-D4_sixteenth note-G4_sixteenth rest-sixteenth note-Bb3_sixteenth rest-sixteenth note-C4_quarter.. note-Bb3_quarter.. "]
        actuallift, actualpitch, _actualrhythm, _actualnotes = split_symbols(merged_accidentals)
        actuallift = [actualpitch[0][i] + l for i, l in enumerate(actuallift[0]) if l != "nonote" and l != "lift_null"]
        self.assertEqual(actuallift, ['note-E4lift_b', 'note-F4lift_#'])

    def test_replace_multirests(self):
        merged_multirests = ["multirest-1 multirest-2 multirest-3 multirest-50 multirest-100 rest-whole2"]
        _actuallift, _actualpitch, actualrhythm, _actualnotes = split_symbols(merged_multirests)
        self.assertEqual(actualrhythm, [['rest-whole', 'multirest-2', 'multirest-3', 'multirest-50', 'multirest-50', 'multirest-2']])

if __name__ == '__main__':
    unittest.main()        