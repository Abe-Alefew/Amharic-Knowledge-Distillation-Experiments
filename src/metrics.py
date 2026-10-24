"""Shared text-quality metrics (used for the teacher pilot, later for reports)."""
import re


def _is_geez(c):
    o = ord(c)
    return 0x1200 <= o <= 0x139F or 0x2D80 <= o <= 0x2DDF or 0xAB00 <= o <= 0xAB2F


def geez_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(_is_geez(c) for c in letters) / max(1, len(letters))


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1].strip()


def quality_metrics(text):
    letters = [c for c in text if c.isalpha()]
    n = max(1, len(letters))
    words = text.split()
    tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return {
        "geez_ratio": round(sum(_is_geez(c) for c in letters) / n, 3),
        "latin_ratio": round(sum(c.isascii() for c in letters) / n, 3),
        "cjk_chars": sum("\u4e00" <= c <= "\u9fff" for c in text),
        "distinct3": round(len(set(tri)) / len(tri), 3) if tri else 1.0,
        "ends_with_fullstop": text.strip().endswith(("።", "?", "፧", "!")),
        "n_words": len(words),
    }
