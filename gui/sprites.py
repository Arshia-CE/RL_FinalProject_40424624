"""Pixel sprites from the MazeMario design: 16px-wide pixel maps rendered to
tk.PhotoImage ('.' = transparent, letters index PAL)."""

from __future__ import annotations

import tkinter as tk

PAL = {
    # hero (original character - teal cap, gold shirt, indigo overalls)
    "T": "#1f9e8e", "t": "#137063",
    "S": "#f6c99b", "s": "#d99a66",
    "A": "#5d3a1a",
    "Y": "#f5b715",
    "O": "#3a55c9", "o": "#22337f",
    "B": "#6e4116",
    "E": "#1a1a1a", "W": "#ffffff",
    "R": "#e0402e",
    # princess
    "P": "#f27bb6", "p": "#c74e8c", "y": "#ffd94a",
    "L": "#dfe6f0",  # silver crown
    "d": "#0d0d18",
}

_HERO_HEAD_DOWN = [
    "......TTTT......",
    "....TTTTTTTT....",
    "...TTTTTTTTTT...",
    "...TTTTTTTTTTT..",
    "....AASSSSSA....",
    "....SESSSESS....",
    "....SSSsSSSS....",
    "....SAAAAAS.....",
]
_HERO_HEAD_UP = [
    "......TTTT......",
    "....TTTTTTTT....",
    "...TTTTTTTTTT...",
    "..TTTTTTTTTTT...",
    "....AAAAAAAA....",
    "....AAAAAAAA....",
    "....AAAAAAAA....",
    ".....AAAAAA.....",
]
_HERO_BODY_0 = [
    "....YYYYYYYY....",
    "..WWOYYYYYYOWW..",
    "..WWOyOOOOyOWW..",
    "....OOOOOOOO....",
    "....OOO..OOO....",
    "....OOO..OOO....",
    "...BBBB..BBBB...",
    "...BBBB..BBBB...",
]
_HERO_BODY_1 = [
    "....YYYYYYYY....",
    "..WWOYYYYYYOWW..",
    "..WWOyOOOOyOWW..",
    "....OOOOOOOO....",
    "....OOO..OOO....",
    "...OOO....OOO...",
    "..BBBB....BBBB..",
    "..BBB......BBB..",
]

HERO = {
    "down": [_HERO_HEAD_DOWN + _HERO_BODY_0, _HERO_HEAD_DOWN + _HERO_BODY_1],
    "up": [_HERO_HEAD_UP + _HERO_BODY_0, _HERO_HEAD_UP + _HERO_BODY_1],
    "side": [
        [
            "......TTTT......",
            ".....TTTTTTT....",
            "....TTTTTTTT....",
            "....TTTTTTTTTT..",
            ".....ASSSSSS....",
            ".....ASSSESS....",
            ".....ASSSSSs....",
            "......SAAAA.....",
            ".....YYYYYY.....",
            ".....OYYYYYOW...",
            ".....OOOOOOO....",
            ".....OOOOOO.....",
            "......OOOO......",
            ".....OOO.OO.....",
            ".....BBB.BBB....",
            "....BBBB.BBB....",
        ],
        [
            "......TTTT......",
            ".....TTTTTTT....",
            "....TTTTTTTT....",
            "....TTTTTTTTTT..",
            ".....ASSSSSS....",
            ".....ASSSESS....",
            ".....ASSSSSs....",
            "......SAAAA.....",
            ".....YYYYYY.....",
            ".....OYYYYYOW...",
            ".....OOOOOOO....",
            "......OOOOO.....",
            "......OOOO......",
            ".....OO..OOO....",
            "....BBB...BBB...",
            "...BBB.....BBB..",
        ],
    ],
}

# ink-outlined (like the key) so she pops against the sand floor;
# silver crown over dark hair so both read against sand and each other
PRINCESS = [
    "....LL.LL.LL....",
    "...dLLLRRLLLd...",
    "..dAAAAAAAAAAd..",
    "..dAASSSSSSAAd..",
    "..dAASESSESAAd..",
    "..dAASSSSSSAAd..",
    "..dAASSssSSAAd..",
    "...dAASSSSAAd...",
    "....dPPPPPPd....",
    "...dPPWPPWPPd...",
    "...dPpPPPPpPd...",
    "..dPPPPPPPPPPd..",
    "..dPPpPPPPpPPd..",
    ".dPPPPPPPPPPPPd.",
    ".dPPpPPPPPPpPPd.",
    ".dddddddddddddd.",
]

# ink-outlined so it pops against the sand floor
KEY_SPRITE = [
    "................",
    "....EEEE........",
    "...EyyyyE.......",
    "..EyyEEyyE......",
    "..EyyEEyyE......",
    "...EyyyyE.......",
    "....EyyE........",
    "....EyyE........",
    "....EyyE........",
    "....EyyyyE......",
    "....EyyEEE......",
    "....EyyyyE......",
    "....EEEEEE......",
    "................",
    "................",
    "................",
]

HEART = [
    ".RR..RR.",
    "RRRRRRRR",
    "RRRRRRRR",
    ".RRRRRR.",
    "..RRRR..",
    "...RR...",
]

# energy bolt for the HUD bar
BOLT = [
    "...YYYY.",
    "..YYYY..",
    ".YYYY...",
    "YYYYWYY.",
    "...YYY..",
    "..YYY...",
    ".YYY....",
    ".YY.....",
]

_cache: dict = {}


def build(rows: list[str], scale: int, flip: bool = False) -> tk.PhotoImage:
    """Render a pixel map to a PhotoImage; untouched pixels stay transparent."""
    key = (id(rows), scale, flip)
    if key in _cache:
        return _cache[key]
    width = max(len(r) for r in rows)
    img = tk.PhotoImage(width=width * scale, height=len(rows) * scale)
    for r, row in enumerate(rows):
        c = 0
        while c < len(row):
            ch = row[len(row) - 1 - c] if flip else row[c]
            if ch == "." or ch not in PAL:
                c += 1
                continue
            run = c  # find the horizontal run of identical color
            while run < len(row):
                nxt = row[len(row) - 1 - run] if flip else row[run]
                if nxt != ch:
                    break
                run += 1
            img.put(PAL[ch], to=(c * scale, r * scale,
                                 run * scale, (r + 1) * scale))
            c = run
    _cache[key] = img
    return img
