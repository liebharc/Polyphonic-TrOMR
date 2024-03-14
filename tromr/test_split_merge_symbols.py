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

if __name__ == '__main__':
    unittest.main()        