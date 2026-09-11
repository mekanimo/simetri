"""Musical notes and scales.  MIDI and frequency conversions."""

from collections.abc import Sequence
from math import log2
from typing import Any

# from simetri.graphics.extensions.all_enums import MusicScale


def midi_freq(m: int) -> float:
    """Return the frequency of a MIDI note number."""
    return 2 ** ((m - 69) / 12) * 440


def midi_m(freq: float) -> int:
    """Return the MIDI note number of a frequency."""
    return 12 * log2(freq / 440) + 69


def note_name(midi_note: int) -> str:
    """
    Returns the name of a MIDI note given its number.

    Args:
        midi_note: An integer representing the MIDI note number (0-127).

    Returns:
        A string representing the note name (e.g., "C4", "G#5"), or None if the input is invalid.
    """
    if not isinstance(midi_note, int) or midi_note < 0 or midi_note > 127:
        return None

    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi_note // 12 - 1
    note_index = midi_note % 12
    return notes[note_index] + str(octave)


def scale(m: int, scale_type: str = "major") -> list:
    """Return a list of MIDI notes in a scale."""
    steps = {
        "major": [2, 2, 1, 2, 2, 2, 1],
        "minor": [2, 1, 2, 2, 1, 2, 2],
        "chromatic": [1] * 12,
        "pentatonic": [2, 2, 3, 2, 3],
    }
    s = [m]
    for step in steps[scale_type]:
        m += step
        s.append(m)
    return s


def jump_scale(scale: Sequence, reference: Any, step: int) -> Any:
    """Return the note in a scale that is a certain number of steps away from a reference note.
    Args:
        scale: A list of MIDI notes in a scale.
        reference: The reference note.
        step: The number of steps away from the reference note.
    Returns:
        The note in the scale that is a certain number of steps away from the reference note.

    Example:
        jump_scale([60, 62, 64, 65, 67, 69, 71], 60, -2) returns 69.
    """
    i = scale.index(reference)
    n = len(scale)

    return scale[(i + step) % n]
