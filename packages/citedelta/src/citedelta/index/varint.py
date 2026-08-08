"""LEB128 variable-length integers."""

from __future__ import annotations


def write_varint(value: int, out: bytearray) -> None:
    """7 bits of payload per byte; the high bit says 'more to come'.

    Small numbers cost one byte, large ones cost more. Postings are stored as
    GAPS between consecutive document ids — and gaps are small even when ids
    are large — so nearly every number written here fits in one or two bytes
    instead of a fixed four.
    """
    if value < 0:
        msg = f"varint is unsigned; got {value}"
        raise ValueError(msg)
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return


def read_varint(buf: memoryview | bytes | bytearray, pos: int) -> tuple[int, int]:
    """Returns (value, next_position)."""
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
