"""MazeMario game board: pixel-tile rendering of a MazeMap plus the
animation layer (hero tween/bump/fall, wizard blink poofs, door slide,
key bob, reward popups, sparks, hearts)."""

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
                     "fall": None, "rise": 1.0, "zap": 0.0,
                     "door_open": 0.0, "door_opening": False,
                     "win_t": 0.0, "hearts": [], "popups": [], "sparks": [],
                     "puffs": []}
        self._agent_cell = None
        self._wizard_cell = None
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
        self._draw_wizard()
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

    def _draw_wizard(self) -> None:
        """The blinking wizard; set_wizard_cell() places and animates him."""
        self._wizard_item = self.create_image(0, 0, anchor="sw", image="",
                                              state="hidden", tags="wizard")

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
        an.update({"move": None, "bump": None, "fall": None,
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

    def zap(self) -> None:
        """Blocked entry: staggered zap rings plus a wizard shake."""
        self.anim["zap"] = 0.001
        if self._wizard_cell is not None:
            for delay, color in ((0.0, "#ffffff"), (-0.09, "#79e8f2")):
                self._spawn_particle("ring", self._wizard_cell, t=delay,
                                     dur=0.28, color=color)

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

    def set_wizard_cell(self, cell) -> None:
        """Blink the wizard: vanish poof at the old cell, materialize poof
        and a hat-first rise at the new one (first call places silently)."""
        cell = tuple(cell)
        if cell == self._wizard_cell:
            return
        if self._wizard_cell is not None:
            self._spawn_poof(self._wizard_cell)
            self._spawn_poof(cell)
            self.anim["rise"] = 0.0
        self._wizard_cell = cell

    def _spawn_poof(self, cell) -> None:
        """Smoke puffs, flying sparkles and an expanding ring at a cell."""
        for _ in range(7):
            ang = random.random() * math.tau
            self._spawn_particle(
                "smoke", cell, dur=0.42,
                dx=math.cos(ang) * 3, dy=math.sin(ang) * 3,
                vx=math.cos(ang) * 14 * self.scale,
                vy=math.sin(ang) * 14 * self.scale - 10 * self.scale,
                r=(2.2 + random.random() * 2.2) * self.scale)
        for i in range(8):
            ang = (i / 8) * math.tau + random.random() * 0.5
            speed = (55 + random.random() * 40) * self.scale
            self._spawn_particle(
                "spark", cell, dur=0.35,
                vx=math.cos(ang) * speed, vy=math.sin(ang) * speed,
                color=theme.GOLD if i % 2 else "#79e8f2")
        self._spawn_particle("ring", cell, dur=0.3, color="#c9b8ec")

    def _spawn_particle(self, kind: str, cell, t: float = 0.0, dur: float = 0.3,
                        dx: float = 0.0, dy: float = 0.0, vx: float = 0.0,
                        vy: float = 0.0, r: float = 0.0,
                        color: str = "#ffffff") -> None:
        x, y = self.cell_xy(*cell)
        self.anim["puffs"].append({
            "kind": kind, "t": t, "dur": dur, "vx": vx, "vy": vy, "r": r,
            "color": color, "x": x + self.cell / 2 + self.px(dx),
            "y": y + self.cell / 2 + self.px(dy)})

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
        if an["zap"]:
            an["zap"] += dt
            if an["zap"] > 0.4:
                an["zap"] = 0.0
        an["rise"] = min(1.0, an["rise"] + dt * 4.5)
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
        for pf in an["puffs"]:
            pf["t"] += dt
            if pf["t"] < 0:  # staggered start (second zap ring)
                continue
            if pf["t"] >= pf["dur"]:
                if "item" in pf:
                    self.delete(pf["item"])
                continue
            pf["x"] += pf["vx"] * dt
            pf["y"] += pf["vy"] * dt
            drag = max(0.0, 1 - 4 * dt)
            pf["vx"] *= drag
            pf["vy"] *= drag
            self._draw_particle(pf)
        an["puffs"] = [pf for pf in an["puffs"] if pf["t"] < pf["dur"]]

    SMOKE_SHADES = ("#ffffff", "#e6dcf7", "#c9b8ec", "#a893da")

    def _draw_particle(self, pf: dict) -> None:
        """One puff/ring/spark frame: size and shade follow life fraction."""
        p = pf["t"] / pf["dur"]
        if pf["kind"] == "smoke":
            r = pf["r"] * (0.4 + math.sin(math.pi * p) * 1.4)
            if "item" not in pf:
                pf["item"] = self.create_oval(0, 0, 0, 0, outline="")
            self.itemconfigure(pf["item"],
                               fill=self.SMOKE_SHADES[min(3, int(p * 4))])
        elif pf["kind"] == "ring":
            r = self.px(2 + 9.5 * p)
            if "item" not in pf:
                pf["item"] = self.create_oval(0, 0, 0, 0, fill="",
                                              outline=pf["color"])
            self.itemconfigure(pf["item"],
                               width=max(1.0, self.scale * 1.5 * (1 - p)))
        else:  # spark
            r = self.scale * (1.6 - p)
            if "item" not in pf:
                pf["item"] = self.create_rectangle(0, 0, 0, 0,
                                                   fill=pf["color"],
                                                   outline="")
        self.coords(pf["item"], pf["x"] - r, pf["y"] - r,
                    pf["x"] + r, pf["y"] + r)

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
        # wizard: hat-first rise after a blink, hover bob, zap shake
        if self._wizard_cell is not None:
            wx, wy = self.cell_xy(*self._wizard_cell)
            visible = max(0, min(len(sprites.WIZARD),
                                 round(len(sprites.WIZARD) * an["rise"])))
            if visible == 0:
                self.itemconfigure(self._wizard_item, state="hidden")
            else:
                shake = (math.sin(an["zap"] * 40) * 1.6 * self.scale
                         if an["zap"] else 0.0)
                bob = (math.sin(self.time * 2.2) * 1.2 * self.scale
                       if an["rise"] >= 1.0 else 0.0)
                self.itemconfigure(
                    self._wizard_item, state="normal",
                    image=sprites.crop_top(sprites.WIZARD, self.scale,
                                           visible))
                self.coords(self._wizard_item, wx + shake,
                            wy + self.cell + bob)
        # hero position (tween / bump / fall)
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
        name, flip = FACING[an["facing"]]
        self.itemconfigure(
            self._hero_item,
            image=sprites.build(sprites.HERO[name][an["frame"]], self.scale,
                                flip),
            state="hidden" if hidden else "normal")
        x, y = self.cell_xy(r, c)
        self.coords(self._hero_item, x, y + hop + sink)
        self.tag_raise(self._hero_item)
