"""
Nigerian licence plate format checks and OCR clean-up.

Standard Nigerian private plates follow the pattern:  ABC-123DE
  - 3 letters: local government area code
  - 3 digits:  serial number
  - 2 letters: series

OCR often confuses similar-looking characters (O/0, I/1, S/5, B/8...).
Because we know which positions must be letters and which must be digits,
we can correct these confusions by position.
"""

import re

PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{3}[A-Z]{2}$")

# Digit-looking letters -> digit, for the digit positions
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6", "T": "7"}
# Letter-looking digits -> letter, for the letter positions
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}

LETTER_POSITIONS = {0, 1, 2, 6, 7}
DIGIT_POSITIONS = {3, 4, 5}


def clean(text):
    """Keep only letters and digits, upper-case."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def correct_by_position(text):
    """
    Fix common OCR confusions using the known position of letters and digits.
    Only applies to 8-character strings (the standard format).
    """
    if len(text) != 8:
        return text
    chars = list(text)
    for i, ch in enumerate(chars):
        if i in DIGIT_POSITIONS and not ch.isdigit():
            chars[i] = TO_DIGIT.get(ch, ch)
        elif i in LETTER_POSITIONS and ch.isdigit():
            chars[i] = TO_LETTER.get(ch, ch)
    return "".join(chars)


def normalise(raw_text):
    """
    Turn raw OCR output into a plate string.
    Returns (plate, is_valid). Plate is formatted as ABC-123DE when valid.
    """
    text = clean(raw_text)
    # OCR sometimes picks up extra characters (state name, borders); look for an 8-char window
    candidates = [text] + [text[i:i + 8] for i in range(max(0, len(text) - 7))]
    for c in candidates:
        fixed = correct_by_position(c)
        if PLATE_RE.match(fixed):
            return f"{fixed[:3]}-{fixed[3:]}", True
    return text, False
