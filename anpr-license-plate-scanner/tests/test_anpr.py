import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anpr.plate_format import correct_by_position, normalise  # noqa: E402


class TestPlateFormat(unittest.TestCase):
    def test_valid_plate_formatted(self):
        self.assertEqual(normalise("LND123AB"), ("LND-123AB", True))

    def test_punctuation_and_case_removed(self):
        self.assertEqual(normalise(" lnd-123 ab. "), ("LND-123AB", True))

    def test_letter_o_in_digit_position_fixed(self):
        self.assertEqual(correct_by_position("ABCO23DE"), "ABC023DE")

    def test_digit_in_letter_position_fixed(self):
        self.assertEqual(correct_by_position("A8C123DE"), "ABC123DE")

    def test_extra_characters_ignored(self):
        plate, valid = normalise("LAGOS KJA456CD")
        self.assertTrue(valid)
        self.assertEqual(plate, "KJA-456CD")

    def test_wrong_length_is_invalid(self):
        self.assertFalse(normalise("AB12")[1])


@unittest.skipUnless(shutil.which("tesseract"), "Tesseract not installed")
class TestEndToEnd(unittest.TestCase):
    def check(self, text, skew):
        from anpr import detect
        from make_test_image import make_scene
        results = detect(make_scene(text, skew))
        self.assertTrue(results, "no plate found")
        self.assertEqual(results[0].text, f"{text[:3]}-{text[3:]}")
        self.assertTrue(results[0].valid)

    def test_straight_plate(self):
        self.check("LND123AB", 0)

    def test_skewed_plate(self):
        self.check("KJA456CD", 14)

    def test_no_plate_in_empty_scene(self):
        import numpy as np
        from anpr import detect
        blank = np.full((480, 640, 3), 90, np.uint8)
        self.assertEqual(detect(blank), [])


if __name__ == "__main__":
    unittest.main()
