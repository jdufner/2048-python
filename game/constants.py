SIZE: int = 400
GRID_LEN: int = 4
GRID_PADDING: int = 10

BACKGROUND_COLOR_GAME: str = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY: str = "#9e948a"

BACKGROUND_COLOR_DICT: dict[int, str] = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
    4096: "#eee4da",
    8192: "#edc22e",
    16384: "#f2b179",
    32768: "#f59563",
    65536: "#f67c5f",
}

CELL_COLOR_DICT: dict[int, str] = {
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
    32: "#f9f6f2",
    64: "#f9f6f2",
    128: "#f9f6f2",
    256: "#f9f6f2",
    512: "#f9f6f2",
    1024: "#f9f6f2",
    2048: "#f9f6f2",
    4096: "#776e65",
    8192: "#f9f6f2",
    16384: "#776e65",
    32768: "#776e65",
    65536: "#f9f6f2",
}

FONT: tuple[str, int, str] = ("Verdana", 40, "bold")

KEY_QUIT: str = "Escape"
KEY_BACK: str = "b"

KEY_UP: str = "Up"
KEY_DOWN: str = "Down"
KEY_LEFT: str = "Left"
KEY_RIGHT: str = "Right"

KEY_UP_ALT1: str = "w"
KEY_DOWN_ALT1: str = "s"
KEY_LEFT_ALT1: str = "a"
KEY_RIGHT_ALT1: str = "d"

KEY_UP_ALT2: str = "i"
KEY_DOWN_ALT2: str = "k"
KEY_LEFT_ALT2: str = "j"
KEY_RIGHT_ALT2: str = "l"
