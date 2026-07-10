"""MazeMario chrome: HUD bar, control bar, footer, board overlays and the
pause/title menu, styled after the design's pixel look."""

from __future__ import annotations

import tkinter as tk

from gui import theme
from gui.controller import WORLDS


def pixel_button(parent, text, bg, fg, command, size=8,
                 min_width=0) -> tk.Frame:
    """Chunky button: 3px ink border plus a solid 'shadow' below."""
    outer = tk.Frame(parent, bg=theme.INK)
    btn = tk.Button(outer, text=text, bg=bg, fg=fg, bd=0,
                    activebackground=bg, activeforeground=fg,
                    font=theme.pixel_font(size), command=command,
                    padx=10, pady=6, cursor="hand2")
    if min_width:
        btn.config(width=min_width)
    btn.pack(padx=3, pady=(3, 6))
    outer.button = btn
    return outer


def _panel(parent) -> tk.Frame:
    """Navy panel with the 4px dark border + drop shadow."""
    shadow = tk.Frame(parent, bg=theme.HUD_BORDER)
    inner = tk.Frame(shadow, bg=theme.HUD_BG)
    inner.pack(padx=4, pady=(4, 8), fill="x")
    shadow.inner = inner
    return shadow


class HudBar:
    def __init__(self, parent, on_pause):
        self.frame = _panel(parent)
        inner = self.frame.inner
        tk.Label(inner, text="MAZEMARIO", font=theme.pixel_font(11),
                 bg=theme.HUD_BG, fg=theme.GOLD).pack(side="left",
                                                      padx=(10, 6), pady=8)
        holder = tk.Frame(inner, bg=theme.HUD_BG)
        holder.pack(side="left", expand=True)
        self.score = self._stat(holder, "SCORE")
        self.steps = self._stat(holder, "STEPS")
        self.key = self._stat(holder, "KEY")
        self.dragon = self._stat(holder, "DRAGON")
        pixel_button(inner, "PAUSE", theme.GOLD, theme.INK,
                     on_pause).pack(side="right", padx=8, pady=4)

    def _stat(self, parent, label) -> tk.Label:
        column = tk.Frame(parent, bg=theme.HUD_BG)
        column.pack(side="left", padx=10)
        tk.Label(column, text=label, font=theme.pixel_font(6),
                 bg=theme.HUD_BG, fg=theme.LABEL).pack()
        value = tk.Label(column, text="0", font=theme.pixel_font(9),
                         bg=theme.HUD_BG, fg=theme.WHITE)
        value.pack()
        return value

    def update(self, score, steps_text, has_key, gate_open, countdown):
        self.score.config(text=str(score))
        self.steps.config(text=steps_text)
        if has_key:
            self.key.config(text="GOT!", fg=theme.GOLD)
        else:
            self.key.config(text="---", fg=theme.DISABLED)
        if gate_open:
            self.dragon.config(text=f"IN·{countdown}", fg=theme.GREEN)
        else:
            self.dragon.config(text=f"OUT·{countdown}", fg=theme.POP_BAD)


class ControlBar:
    def __init__(self, parent, on_toggle, on_step, on_restart, on_speed):
        self.frame = _panel(parent)
        inner = self.frame.inner
        self.play = pixel_button(inner, "PAUSE", theme.GREEN, theme.INK,
                                 on_toggle, min_width=7)
        self.play.pack(side="left", padx=(8, 4), pady=4)
        pixel_button(inner, "STEP", theme.BLUE, theme.WHITE,
                     on_step).pack(side="left", padx=4, pady=4)
        pixel_button(inner, "RESTART", theme.RED, theme.WHITE,
                     on_restart).pack(side="left", padx=4, pady=4)
        right = tk.Frame(inner, bg=theme.HUD_BG)
        right.pack(side="right", padx=10)
        tk.Label(right, text="SPEED", font=theme.pixel_font(7),
                 bg=theme.HUD_BG, fg=theme.LABEL).pack(side="left", padx=4)
        self.speed_var = tk.DoubleVar(value=1.5)
        tk.Scale(right, variable=self.speed_var, from_=0.5, to=4.0,
                 resolution=0.5, orient="horizontal", showvalue=False,
                 length=110, bg=theme.GOLD, troughcolor=theme.HUD_BORDER,
                 highlightthickness=0, bd=0, sliderrelief="flat",
                 sliderlength=22, activebackground=theme.WHITE,
                 command=on_speed).pack(side="left")
        self.speed_text = tk.Label(right, text="1.5X",
                                   font=theme.pixel_font(8),
                                   bg=theme.HUD_BG, fg=theme.WHITE)
        self.speed_text.pack(side="left", padx=4)

    def set_playing(self, playing: bool) -> None:
        self.play.button.config(text="PAUSE" if playing else "PLAY")


class BoardOverlay:
    """COURSE CLEAR / TIME UP screens shown over the board."""

    def __init__(self, board):
        self.board = board
        self.frame: tk.Frame | None = None
        self._blink_job = None

    def show(self, kind: str, score: int, steps: int) -> None:
        self.hide()
        bg = (theme.BOARD_OVERLAY if kind == "clear"
              else theme.TIMEOUT_OVERLAY)
        self.frame = tk.Frame(self.board.master, bg=bg)
        self.frame.place(in_=self.board, x=0, y=0, relwidth=1, relheight=1)
        title = tk.Canvas(self.frame, bg=bg, highlightthickness=0,
                          width=520, height=64)
        text = "COURSE CLEAR!" if kind == "clear" else "TIME UP!"
        color = theme.GOLD if kind == "clear" else theme.POP_BAD
        font = theme.pixel_font(24)
        title.create_text(264, 36, text=text, font=font, fill=theme.INK)
        if kind == "clear":
            title.create_text(262, 34, text=text, font=font, fill=theme.RED)
        title.create_text(260, 32, text=text, font=font, fill=color)
        title.pack(pady=(90, 4))
        sub = (f"SCORE {score} · {steps} STEPS" if kind == "clear"
               else f"SCORE {score} · STEP CAP REACHED")
        tk.Label(self.frame, text=sub, font=theme.pixel_font(10), bg=bg,
                 fg=theme.WHITE).pack(pady=6)
        self.hint = tk.Label(self.frame, text="PRESS RESTART TO RUN AGAIN",
                             font=theme.pixel_font(8), bg=bg,
                             fg=theme.LABEL)
        self.hint.pack(pady=8)
        self._blink(bg)

    def _blink(self, bg) -> None:
        if self.frame is None:
            return
        current = self.hint.cget("fg")
        self.hint.config(fg=bg if current != bg else theme.LABEL)
        self._blink_job = self.frame.after(600, lambda: self._blink(bg))

    def hide(self) -> None:
        if self._blink_job and self.frame:
            self.frame.after_cancel(self._blink_job)
            self._blink_job = None
        if self.frame is not None:
            self.frame.destroy()
            self.frame = None


class PauseMenu:
    """Fullscreen title / pause screen with world select."""

    def __init__(self, root, callbacks):
        self.root = root
        self.callbacks = callbacks
        self.frame: tk.Frame | None = None
        self._blink_job = None

    def show(self, selected_world: str, resume_label: str) -> None:
        self.hide()
        bg = theme.OVERLAY_BG
        self.frame = tk.Frame(self.root, bg=bg)
        self.frame.place(x=0, y=0, relwidth=1, relheight=1)

        title = tk.Canvas(self.frame, bg=bg, highlightthickness=0,
                          width=620, height=84)
        font = theme.pixel_font(36)
        title.create_text(316, 48, text="MAZEMARIO", font=font,
                          fill=theme.INK)
        title.create_text(313, 45, text="MAZEMARIO", font=font,
                          fill=theme.RED)
        title.create_text(310, 42, text="MAZEMARIO", font=font,
                          fill=theme.GOLD)
        title.pack(pady=(46, 2))
        tk.Label(self.frame, text="DYNAMIC MAZE · RL PROJECT 40424624",
                 font=theme.pixel_font(8), bg=bg,
                 fg=theme.LABEL).pack(pady=(0, 18))

        gold = tk.Frame(self.frame, bg=theme.GOLD)
        gold.pack(pady=6)
        box = tk.Frame(gold, bg=theme.HUD_BG)
        box.pack(padx=4, pady=4)
        tk.Label(box, text="SELECT WORLD", font=theme.pixel_font(8),
                 bg=theme.HUD_BG, fg=theme.LABEL,
                 anchor="w").pack(fill="x", padx=14, pady=(12, 6))
        for key, world in WORLDS.items():
            self._world_row(box, key, world, key == selected_world)

        buttons = tk.Frame(self.frame, bg=bg)
        buttons.pack(pady=20)
        pixel_button(buttons, resume_label, theme.GOLD, theme.INK,
                     self.callbacks["resume"], size=10).pack(side="left",
                                                             padx=8)
        pixel_button(buttons, "RESTART", theme.RED, theme.WHITE,
                     self.callbacks["restart"], size=10).pack(side="left",
                                                              padx=8)
        self.hint = tk.Label(
            self.frame,
            text="THE TRAINED AGENT PLAYS BY ITSELF — SIT BACK AND WATCH",
            font=theme.pixel_font(7), bg=bg, fg=theme.DIM)
        self.hint.pack(pady=10)
        self._blink()

    def _world_row(self, box, key, world, selected) -> None:
        row = tk.Frame(box, bg=theme.HUD_BG, cursor="hand2")
        row.pack(fill="x", padx=10, pady=2)
        cursor = tk.Label(row, text=">" if selected else " ",
                          font=theme.pixel_font(9), bg=theme.HUD_BG,
                          fg=theme.GOLD, width=2)
        cursor.pack(side="left")
        text = tk.Frame(row, bg=theme.HUD_BG)
        text.pack(side="left", fill="x", expand=True, pady=5)
        name = tk.Label(text, text=world["label"], font=theme.pixel_font(9),
                        bg=theme.HUD_BG,
                        fg=theme.GOLD if selected else theme.WHITE,
                        anchor="w")
        name.pack(fill="x")
        desc = tk.Label(text, text=world["desc"], font=theme.pixel_font(6),
                        bg=theme.HUD_BG, fg=theme.DIM, anchor="w")
        desc.pack(fill="x")
        widgets = (row, cursor, text, name, desc)

        def enter(_e):
            for w in widgets:
                w.config(bg=theme.MENU_HOVER)

        def leave(_e):
            for w in widgets:
                w.config(bg=theme.HUD_BG)

        for w in widgets:
            w.bind("<Enter>", enter)
            w.bind("<Leave>", leave)
            w.bind("<Button-1>",
                   lambda _e, k=key: self.callbacks["select_world"](k))

    def _blink(self) -> None:
        if self.frame is None:
            return
        current = self.hint.cget("fg")
        self.hint.config(fg=theme.OVERLAY_BG
                         if current != theme.OVERLAY_BG else theme.DIM)
        self._blink_job = self.frame.after(700, self._blink)

    def hide(self) -> None:
        if self._blink_job and self.frame:
            self.frame.after_cancel(self._blink_job)
            self._blink_job = None
        if self.frame is not None:
            self.frame.destroy()
            self.frame = None

    @property
    def visible(self) -> bool:
        return self.frame is not None
