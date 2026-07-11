"""MazeMario GUI theme: palette, fonts and timing."""

import tkinter.font as tkfont

# chrome
SKY = "#5c94fc"
HUD_BG = "#1b2447"
HUD_BORDER = "#10162e"
GOLD = "#ffd94a"
RED = "#e0402e"
GREEN = "#7bc86c"
BLUE = "#3a55c9"
INK = "#1a1a1a"
WHITE = "#ffffff"
LABEL = "#8a94c8"
DIM = "#626ea0"
DISABLED = "#4a5486"
FOOTER = "#2a4ba8"
OVERLAY_BG = "#0d1022"
BOARD_OVERLAY = "#1b2447"
TIMEOUT_OVERLAY = "#2e1010"
POP_BAD = "#ff6b5e"
POP_PIT = "#ff3b2e"
MENU_HOVER = "#26305c"

# tile palettes (assets THEMES)
THEMES = {
    "overworld": {
        "floor": "#f0d8a0", "floorDot": "#e0c284",
        "brick": "#c8582c", "brickDark": "#7a2d0c", "brickLight": "#f0a068",
        "startPad": "#7bc86c",
    },
    "dungeon": {
        "floor": "#8b93b8", "floorDot": "#7a82a8",
        "brick": "#3858a8", "brickDark": "#1c2c60", "brickLight": "#7c9be0",
        "startPad": "#5aa0d8",
    },
}

# timing (design: step every 460/speed ms, ~60 fps loop)
STEP_MS = 460
TICK_MS = 30
SPEEDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

_PIXEL_FAMILY = None


def pixel_font(size: int):
    """The design's 'Press Start 2P' when installed, else a chunky fallback."""
    global _PIXEL_FAMILY
    if _PIXEL_FAMILY is None:
        families = set(tkfont.families())
        for candidate in ("Press Start 2P", "Fixedsys", "Courier New"):
            if candidate in families:
                _PIXEL_FAMILY = candidate
                break
        else:
            _PIXEL_FAMILY = "Courier"
    if _PIXEL_FAMILY == "Press Start 2P":
        return (_PIXEL_FAMILY, size)
    return (_PIXEL_FAMILY, size + 2, "bold")
