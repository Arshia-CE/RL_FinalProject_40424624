"""MazeMario game board: pixel-tile rendering of a MazeMap plus the
animation layer (hero tween/bump/fall/death collapse, door slide, key bob,
reward popups, sparks, hearts)."""

from __future__ import annotations

import math
import random
import sys
import tkinter as tk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.maze_map import PENALTY, START, MazeMap
from gui import sprites, theme

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
FACING = {UP: ("up", False), DOWN: ("down", False),
          LEFT: ("side", True), RIGHT: ("side", False)}


class GameBoard(tk.Canvas):
    """Draws one maze as a pixel game level and animates everything on it."""

    def __init__(self, master, scale: int = 2, tile_theme: str = "overworld",
                 **kwargs):
        super().__init__(master, highlightthickness=0, bg=theme.SKY, **kwargs)
        self.scale = scale
        self.cell = 16 * scale
        self.tile_theme = tile_theme
        self.maze: MazeMap | None = None
        self.time = 0.0
        self._reset_anim()

    def _reset_anim(self) -> None:
        self.anim = {"move": None, "facing": DOWN, "frame": 0, "bump": None,
                     "fall": None, "death": None,
                     "door_open": 0.0, "door_opening": False,
                     "win_t": 0.0, "hearts": [], "popups": [], "sparks": []}
        self._agent_cell = None
        self._key_visible = True

    # coordinate helpers (design units are 16px tiles; we scale by `scale`)

    def px(self, v: float) -> float:
        return v * self.scale

    def cell_xy(self, r: float, c: float) -> tuple[float, float]:
        return c * self.cell, r * self.cell

    # scene construction

    def set_map(self, maze: MazeMap) -> None:
        self.maze = maze
        side = maze.size * self.cell
        self.config(width=side, height=side)
        self._reset_anim()
        self.delete("all")
        T = theme.THEMES[self.tile_theme]
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.is_wall((r, c)):
                    self._draw_brick(r, c, T)
                else:
                    self._draw_floor(r, c, T, maze.grid[r][c])
        self._draw_door()
        self._draw_princess()
        self._draw_key()
        self._draw_hero(maze.start, DOWN, 0)

    def _draw_brick(self, r: int, c: int, T: dict) -> None:
        x, y = self.cell_xy(r, c)
        s, cell = self.px, self.cell
        self.create_rectangle(x, y, x + cell, y + cell, fill=T["brick"],
                              outline="")
        for bx, by, bw, bh in ((0, 7, 16, 1), (0, 15, 16, 1), (7, 0, 1, 8),
                               (3, 8, 1, 8), (11, 8, 1, 8)):
            self.create_rectangle(x + s(bx), y + s(by), x + s(bx + bw),
                                  y + s(by + bh), fill=T["brickDark"],
                                  outline="")
        self.create_rectangle(x, y, x + cell, y + s(1),
                              fill=T["brickLight"], outline="")
        self.create_rectangle(x, y, x + s(1), y + s(7),
                              fill=T["brickLight"], outline="")

    def _draw_floor(self, r: int, c: int, T: dict, cell_type: str) -> None:
        x, y = self.cell_xy(r, c)
        s, cell = self.px, self.cell
        self.create_rectangle(x, y, x + cell, y + cell, fill=T["floor"],
                              outline="")
        for dx, dy in ((4, 4), (11, 10)):
            self.create_rectangle(x + s(dx), y + s(dy), x + s(dx + 1),
                                  y + s(dy + 1), fill=T["floorDot"],
                                  outline="")
        if cell_type == PENALTY:
            self._draw_pit(x, y)
        elif cell_type == START:
            self.create_rectangle(x + s(2), y + s(13), x + s(14), y + s(15),
                                  fill=T["startPad"], outline="")
            self.create_rectangle(x + s(6), y + s(11), x + s(10), y + s(13),
                                  fill=T["startPad"], outline="")

    def _draw_pit(self, x: float, y: float) -> None:
        """Penalty cell: a black hole ringed by curved thorns."""
        s = self.px
        px, py = x + s(8), y + s(8)
        for i in range(9):
            a0 = (i / 9) * math.tau + math.pi / 9
            ox, oy = math.cos(a0), math.sin(a0)
            nx, ny = -oy, ox
            b1 = (px + s(ox * 3.4 - nx * 2.0), py + s(oy * 3.4 - ny * 2.0))
            b2 = (px + s(ox * 3.4 + nx * 2.0), py + s(oy * 3.4 + ny * 2.0))
            tip = (px + s(ox * 8.2 + nx * 3.2), py + s(oy * 8.2 + ny * 3.2))
            mid = (px + s(ox * 6.4 + nx * 1.2), py + s(oy * 6.4 + ny * 1.2))
            self.create_polygon(*b1, *mid, *tip, *b2, fill="#3d5a1e",
                                outline="#22350f", smooth=True)
            self.create_line(*b1, *tip, fill="#7fae3c", width=self.scale * 0.7)
        self.create_oval(px - s(4.8), py - s(4.8), px + s(4.8), py + s(4.8),
                         fill="#2a2018", outline="")
        self.create_oval(px - s(3.8), py + s(0.5) - s(3.8), px + s(3.8),
                         py + s(0.5) + s(3.8), fill="#000000", outline="")

    def _draw_door(self) -> None:
        d = self.maze.door
        x, y = self.cell_xy(*d)
        s, cell = self.px, self.cell
        T = theme.THEMES[self.tile_theme]
        self.create_rectangle(x, y, x + cell, y + cell, fill=T["brickDark"],
                              outline="")
        self.create_rectangle(x + s(2), y + s(2), x + s(14), y + s(16),
                              fill="#0d0d18", outline="")
        self._door_xy = (x, y)
        self._paint_door_panel()

    def _paint_door_panel(self) -> None:
        """Wooden panel whose width shrinks as the door swings open."""
        self.delete("door_panel")
        x, y = self._door_xy
        s = self.px
        w = round(12 * (1 - self.anim["door_open"]))
        if w <= 0:
            return
        self.create_rectangle(x + s(2), y + s(2), x + s(2 + w), y + s(16),
                              fill="#8a5a24", outline="", tags="door_panel")
        for i in range(0, w, 3):
            self.create_rectangle(x + s(2 + i), y + s(2), x + s(3 + i),
                                  y + s(16), fill="#6e4116", outline="",
                                  tags="door_panel")
        if self.anim["door_open"] == 0 and self._key_visible:
            self.create_rectangle(x + s(6), y + s(7), x + s(10), y + s(11),
                                  fill=theme.GOLD, outline="",
                                  tags="door_panel")
            self.create_rectangle(x + s(7), y + s(9), x + s(9), y + s(11),
                                  fill=theme.INK, outline="",
                                  tags="door_panel")

    def _draw_princess(self) -> None:
        g = self.maze.goal
        x, y = self.cell_xy(*g)
        self._princess_base = (x, y)
        self._princess_item = self.create_image(
            x, y, anchor="nw", image=sprites.build(sprites.PRINCESS,
                                                   self.scale))

    def _draw_key(self) -> None:
        k = self.maze.key
        x, y = self.cell_xy(*k)
        s = self.px
        self._key_base = (x, y)
        self._key_shadow = self.create_oval(
            x + s(3), y + s(12.5), x + s(12), y + s(15),
            fill="#cbb180", outline="")
        self._key_item = self.create_image(
            x, y, anchor="nw", image=sprites.build(sprites.KEY_SPRITE,
                                                   self.scale))
        self._twinkle = self.create_rectangle(0, 0, 0, 0, fill="#ffffff",
                                              outline="")

    def _draw_hero(self, pos, facing: int, frame: int) -> None:
        self._agent_cell = tuple(pos)
        x, y = self.cell_xy(*pos)
        name, flip = FACING[facing]
        self._hero_item = self.create_image(
            x, y, anchor="nw",
            image=sprites.build(sprites.HERO[name][frame], self.scale, flip))

    def new_episode(self) -> None:
        """Restore per-episode visuals: key back, door shut, hero at start."""
        an = self.anim
        an.update({"move": None, "bump": None, "fall": None, "death": None,
                   "door_open": 0.0, "door_opening": False, "win_t": 0.0})
        self._key_visible = True
        self.itemconfigure(self._key_item, state="normal")
        self.itemconfigure(self._twinkle, state="normal")
        self.itemconfigure(self._key_shadow, state="normal")
        self._paint_door_panel()
        self._agent_cell = tuple(self.maze.start)

    ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    def draw_policy(self, action_fn) -> None:
        """Greedy-action arrows on every floor cell (None clears)."""
        self.delete("policy")
        if action_fn is None:
            return
        font = ("Segoe UI", max(7, 4 * self.scale), "bold")
        for r in range(self.maze.size):
            for c in range(self.maze.size):
                if self.maze.is_wall((r, c)):
                    continue
                action = action_fn(r, c)
                if action is None:
                    continue
                x, y = self.cell_xy(r, c)
                self.create_text(x + self.cell / 2, y + self.cell / 2,
                                 text=self.ARROWS[action], fill="#1b2447",
                                 font=font, tags="policy")

    # events from the game session

    def start_move(self, from_pos, to_pos, direction: int,
                   duration: float) -> None:
        self.anim["move"] = {"from": tuple(from_pos), "to": tuple(to_pos),
                             "t": 0.0, "dur": max(0.05, duration)}
        self._face(direction)

    def bump(self, direction: int) -> None:
        self.anim["bump"] = {"dir": direction, "t": 0.0}
        self._face(direction)

    def _face(self, direction: int) -> None:
        self.anim["facing"] = direction
        self.anim["frame"] = 1 - self.anim["frame"]

    def fall_in_pit(self) -> None:
        move = self.anim["move"]
        self.anim["fall"] = {"t": -(move["dur"] if move else 0.0)}

    def collect_key(self) -> None:
        self._key_visible = False
        self.itemconfigure(self._key_item, state="hidden")
        self.itemconfigure(self._twinkle, state="hidden")
        self.itemconfigure(self._key_shadow, state="hidden")
        self._paint_door_panel()
        kx, ky = self._key_base
        for _ in range(10):
            self.anim["sparks"].append({
                "x": kx + self.cell / 2, "y": ky + self.cell / 2,
                "vx": (random.random() - 0.5) * 60 * self.scale,
                "vy": (random.random() - 0.8) * 60 * self.scale, "t": 0.0})

    def open_door(self) -> None:
        self.anim["door_opening"] = True

    def energy_death(self) -> None:
        """The budget ran dry: the hero collapses and flickers out."""
        self.anim["death"] = {"t": 0.0}

    def celebrate(self) -> None:
        self.anim["win_t"] = 0.0001

    def popup(self, cell_pos, text: str, color: str) -> None:
        x, y = self.cell_xy(*cell_pos)
        self.anim["popups"].append({
            "x": x + self.cell / 2, "y": y, "text": text, "color": color,
            "t": 0.0,
            "items": (
                self.create_text(x + self.cell / 2 + self.scale, y + self.scale,
                                 text=text, fill=theme.INK,
                                 font=theme.pixel_font(7)),
                self.create_text(x + self.cell / 2, y, text=text, fill=color,
                                 font=theme.pixel_font(7)))})

    def popup_between(self, from_pos, direction: int, text: str,
                      color: str) -> None:
        dr, dc = DELTA[direction]
        self.popup((from_pos[0] + dr * 0.5, from_pos[1] + dc * 0.5), text,
                   color)

    # per-frame animation tick

    def tick(self, dt: float) -> None:
        if self.maze is None:
            return
        self.time += dt
        an = self.anim
        if an["move"]:
            an["move"]["t"] += dt
            if an["move"]["t"] >= an["move"]["dur"]:
                self._agent_cell = an["move"]["to"]
                an["move"] = None
        if an["bump"]:
            an["bump"]["t"] += dt
            if an["bump"]["t"] > 0.3:
                an["bump"] = None
        if an["fall"]:
            an["fall"]["t"] += dt
            if an["fall"]["t"] > 0.9:
                an["fall"] = None
        if an["death"] and an["death"]["t"] < 1.2:
            an["death"]["t"] += dt  # dict stays: the hero remains down
        if an["door_opening"] and an["door_open"] < 1:
            an["door_open"] = min(1.0, an["door_open"] + dt * 3)
            self._paint_door_panel()
        if an["win_t"]:
            self._tick_win(dt)
        self._tick_effects(dt)
        self._redraw_dynamic()

    def _tick_win(self, dt: float) -> None:
        an = self.anim
        an["win_t"] += dt
        gx, gy = self._princess_base
        if len(an["hearts"]) < 14 and int(an["win_t"] * 6) > len(an["hearts"]):
            an["hearts"].append({
                "x": gx + self.cell / 2 + (random.random() - 0.5) * 30 * self.scale / 2,
                "y": gy, "v": (14 + random.random() * 10) * self.scale,
                "t": 0.0,
                "item": self.create_image(0, 0, image=sprites.build(
                    sprites.HEART, self.scale))})
        for h in an["hearts"]:
            h["t"] += dt
            h["y"] -= h["v"] * dt
            if h["t"] > 2.4:
                self.delete(h["item"])
            else:
                self.coords(h["item"], h["x"], h["y"])
        an["hearts"] = [h for h in an["hearts"] if h["t"] <= 2.4]

    def _tick_effects(self, dt: float) -> None:
        an = self.anim
        for p in an["popups"]:
            p["t"] += dt
            p["y"] -= dt * 14 * self.scale
            for item in p["items"]:
                x, y = self.coords(item)[:2]
                self.coords(item, x, p["y"] + (self.scale
                                               if item == p["items"][0] else 0))
            if p["t"] >= 1.3:
                for item in p["items"]:
                    self.delete(item)
        an["popups"] = [p for p in an["popups"] if p["t"] < 1.3]
        for sk in an["sparks"]:
            sk["t"] += dt
            sk["x"] += sk["vx"] * dt
            sk["y"] += sk["vy"] * dt
            sk["vy"] += 60 * self.scale * dt
        for sk in an["sparks"]:
            if "item" not in sk:
                sk["item"] = self.create_rectangle(0, 0, 0, 0,
                                                   fill=theme.GOLD,
                                                   outline="")
            if sk["t"] >= 0.8:
                self.delete(sk["item"])
            else:
                self.coords(sk["item"], sk["x"], sk["y"],
                            sk["x"] + self.scale * 2, sk["y"] + self.scale * 2)
        an["sparks"] = [sk for sk in an["sparks"] if sk["t"] < 0.8]

    def _redraw_dynamic(self) -> None:
        an = self.anim
        # key bob + twinkle
        if self._key_visible:
            kx, ky = self._key_base
            bob = math.sin(self.time * 3) * 2 * self.scale
            self.coords(self._key_item, kx, ky + bob)
            tw = int(self.time * 4) % 4
            sx = kx + self.px([2, 12, 13, 3][tw])
            sy = ky + self.px([2, 3, 11, 12][tw]) + bob
            self.coords(self._twinkle, sx, sy, sx + self.scale,
                        sy + self.px(3))
        # princess bob (bounces after the win)
        gx, gy = self._princess_base
        if an["win_t"]:
            bob = -abs(math.sin(self.time * 6)) * 4 * self.scale
        else:
            bob = math.sin(self.time * 2.5) * 1.5 * self.scale
        self.coords(self._princess_item, gx, gy + bob)
        # hero position (tween / bump / fall / death collapse)
        r, c = self._agent_cell
        hop = 0.0
        if an["move"]:
            p = min(1.0, an["move"]["t"] / an["move"]["dur"])
            fr, to = an["move"]["from"], an["move"]["to"]
            r = fr[0] + (to[0] - fr[0]) * p
            c = fr[1] + (to[1] - fr[1]) * p
            hop = -math.sin(math.pi * p) * 3 * self.scale
        elif an["bump"]:
            p = math.sin(math.pi * min(1.0, an["bump"]["t"] / 0.3)) * 0.3
            dr, dc = DELTA[an["bump"]["dir"]]
            r += dr * p
            c += dc * p
        sink, hidden = 0.0, False
        if an["fall"] and an["fall"]["t"] > 0:
            f = min(1.0, an["fall"]["t"] / 0.9)
            sink = (f * 2 if f < 0.5 else (1 - f) * 2) * 10 * self.scale
            hidden = int(an["fall"]["t"] * 12) % 2 == 0
        if an["death"]:
            f = min(1.0, an["death"]["t"] / 1.1)
            sink += f * 10 * self.scale  # collapse into the floor
            hidden = (f >= 1.0  # flicker faster as the lights go out
                      or int(an["death"]["t"] * (6 + 14 * f)) % 2 == 0)
        name, flip = FACING[an["facing"]]
        self.itemconfigure(
            self._hero_item,
            image=sprites.build(sprites.HERO[name][an["frame"]], self.scale,
                                flip),
            state="hidden" if hidden else "normal")
        x, y = self.cell_xy(r, c)
        self.coords(self._hero_item, x, y + hop + sink)
        self.tag_raise(self._hero_item)
