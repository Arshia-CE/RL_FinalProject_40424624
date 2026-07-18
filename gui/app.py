"""MazeMario application: window assembly and the game loop that lets the
trained agent play the maze."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui import theme
from gui.controller import WORLDS, GameSession
from gui.hud import BoardOverlay, ControlBar, HudBar, PauseMenu
from gui.renderer import GameBoard


class MazeMarioApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MazeMario — Dynamic Maze RL")
        self.root.configure(bg=theme.SKY)
        self.root.resizable(False, False)

        self.session = GameSession()
        self.playing = True
        self.paused = True   # boot into the title screen
        self.speed = 1.5
        self.show_policy = False
        self._acc = 0.0
        self._policy_tick = 0

        content = tk.Frame(self.root, bg=theme.SKY)
        content.pack(padx=14, pady=(12, 8))
        self.hud = HudBar(content, on_pause=self._open_menu)
        self.hud.frame.pack(fill="x", pady=(0, 8))

        usable = self.root.winfo_screenheight() - 300
        scale = 2 if usable >= self.session.maze.size * 32 else 1
        shadow = tk.Frame(content, bg=theme.HUD_BORDER)
        shadow.pack()
        border = tk.Frame(shadow, bg=theme.HUD_BG)
        border.pack(padx=0, pady=(0, 6))
        gold = tk.Frame(border, bg=theme.GOLD)
        gold.pack(padx=6, pady=6)
        self.board = GameBoard(gold, scale=scale)
        self.board.pack(padx=3, pady=3)

        self.controls = ControlBar(content, on_toggle=self._toggle_play,
                                   on_step=self._step_once,
                                   on_restart=self._restart,
                                   on_speed=self._set_speed,
                                   on_policy=self._toggle_policy)
        self.controls.frame.pack(fill="x", pady=(10, 0))

        footer = tk.Frame(content, bg=theme.SKY)
        footer.pack(fill="x", pady=(6, 0))
        self.world_label = tk.Label(footer, text="", bg=theme.SKY,
                                    fg=theme.FOOTER,
                                    font=theme.pixel_font(7))
        self.world_label.pack(side="left")
        self.status_label = tk.Label(footer, text="", bg=theme.SKY,
                                     fg=theme.FOOTER,
                                     font=theme.pixel_font(7))
        self.status_label.pack(side="right")

        self.overlay = BoardOverlay(self.board)
        self.menu = PauseMenu(self.root, {
            "resume": self._resume,
            "restart": self._restart,
            "select_world": self._select_world,
            "select_brain": self._select_brain,
            "select_mode": self._select_mode,
        })

        self._load_board()
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2
        self.root.geometry(f"+{max(0, x)}+4")
        self._show_menu("START")
        self.root.after(theme.TICK_MS, self._tick)

    # wiring

    def _load_board(self) -> None:
        self.board.set_map(self.session.maze)
        self.board.set_gate_open(self.session.gate_open())
        world = WORLDS[self.session.world]
        self.world_label.config(text=world["label"])
        maze = self.session.maze
        self.status_label.config(
            text=f"{maze.size}X{maze.size} · {maze.wall_count} WALLS · "
                 f"{len(maze.penalty_cells)} PITS · GATE T={maze.gate_period}")
        self._update_hud()

    def _update_hud(self) -> None:
        self.hud.update(round(self.session.score),
                        f"{self.session.steps}/{self.session.step_cap}",
                        self.session.has_key, self.session.gate_open(),
                        self.session.gate_countdown(),
                        self.session.episode_number(),
                        self.session.current_epsilon(),
                        self.session.win_rate(),
                        self.session.training_complete)

    def _refresh_policy(self) -> None:
        if self.show_policy:
            self.board.draw_policy(self.session.policy_action)
        else:
            self.board.draw_policy(None)

    # controls

    def _show_menu(self, resume_label: str) -> None:
        self.menu.show(self.session.world, resume_label, self.session.brain,
                       self.session.mode)
        self.menu.set_enabled("mode", "train", self.session.brain != "vi")

    def _open_menu(self) -> None:
        self.paused = True
        watch_done = self.session.mode == "watch" and self.session.outcome
        label = ("PLAY AGAIN" if watch_done
                 else ("RESUME" if self.session.steps
                       or self.session.mode == "train" else "START"))
        self._show_menu(label)

    def _resume(self) -> None:
        if self.session.mode == "watch" and self.session.outcome:
            self._reset_run()
        self.menu.hide()
        self.paused = False
        self.playing = True
        self.controls.set_playing(True)

    def _restart(self) -> None:
        self._reset_run()
        self.menu.hide()
        self.paused = False
        self.playing = True
        self.controls.set_playing(True)

    def _reset_run(self) -> None:
        self.overlay.hide()
        self.session.restart()
        self._load_board()
        self._refresh_policy()
        self._acc = 0.0

    def _select_world(self, key: str) -> None:
        self.overlay.hide()
        self.session.load_world(key)   # solves VI for new worlds (fast)
        self._after_menu_choice()

    def _select_brain(self, key: str) -> None:
        self.overlay.hide()
        self.session.set_brain(key)
        self._after_menu_choice()

    def _select_mode(self, key: str) -> None:
        self.overlay.hide()
        self.session.set_mode(key)
        self._after_menu_choice()

    def _after_menu_choice(self) -> None:
        """Reflect a menu selection without rebuilding the menu (no flicker);
        session may have auto-adjusted (picking VI while training -> WATCH)."""
        self._load_board()
        self._refresh_policy()
        self.menu.set_enabled("mode", "train", self.session.brain != "vi")
        self.menu.refresh(self.session.world, self.session.brain,
                          self.session.mode)

    def _toggle_policy(self) -> None:
        self.show_policy = not self.show_policy
        self.controls.set_policy_on(self.show_policy)
        self._refresh_policy()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.controls.set_playing(self.playing)

    def _step_once(self) -> None:
        self.playing = False
        self.controls.set_playing(False)
        self._do_step()

    def _set_speed(self, value: str) -> None:
        self.speed = float(value)
        self.controls.speed_text.config(text=f"{self.speed:.1f}X")

    # game loop

    def _tick(self) -> None:
        dt = theme.TICK_MS / 1000.0
        fast_train = (self.session.mode == "train" and self.speed >= 2.0
                      and not self.session.training_complete)
        if not self.paused and self.playing:
            if fast_train:
                self.session.run_batch(int(self.speed))
                self.board.new_episode()
                self.board.set_gate_open(self.session.gate_open())
                self._update_hud()
                self._policy_tick += 1
                if self.show_policy and self._policy_tick % 6 == 0:
                    self._refresh_policy()
            elif not self.session.outcome:
                self._acc += theme.TICK_MS
                if self._acc >= theme.STEP_MS / self.speed:
                    self._acc = 0.0
                    self._do_step()
        self.board.tick(dt)
        self.root.after(theme.TICK_MS, self._tick)

    def _do_step(self) -> None:
        event = self.session.step()
        if event is None:
            return
        rewards = self.session.rewards()
        prev = (event["prev"].r, event["prev"].c)
        nxt = (event["next"].r, event["next"].c)
        direction = event["direction"]
        step_s = theme.STEP_MS / self.speed / 1000.0

        if event["moved"]:
            self.board.start_move(prev, nxt, direction,
                                  min(step_s * 0.85, 0.34))
        if event["wall"]:
            self.board.bump(direction)
            self.board.popup_between(prev, direction,
                                     f"{rewards['wall_hit']:+d}",
                                     theme.POP_BAD)
        if event["wizard_blocked"]:
            self.board.bump(direction)
            self.board.roar()
            self.board.popup_between(prev, direction,
                                     f"{rewards['wizard_blocked']:+d}",
                                     theme.POP_BAD)
        if event["door_locked"]:
            self.board.bump(direction)
            self.board.popup_between(prev, direction,
                                     f"{rewards['locked_door_attempt']:+d}",
                                     theme.POP_BAD)
        if event["key"]:
            self.board.collect_key()
            self.board.popup(nxt, f"{rewards['key_pickup']:+d}", theme.GOLD)
        if event["pit"]:
            self.board.fall_in_pit()
            self.board.popup(nxt, f"{rewards['penalty_cell']:+d}",
                             theme.POP_PIT)
        if event["door"]:
            self.board.open_door()
        if event["episode_end"]:
            # training rolls straight into the next episode
            if event["goal"]:
                self.board.popup((nxt[0] - 1.2, nxt[1]),
                                 f"{rewards['goal']:+d}", theme.GOLD)
            delay = 900 if event["goal"] else 250
            self.root.after(delay, self._next_training_episode)
        elif event["goal"]:
            # popup a cell higher so it clears the rising hearts
            self.board.popup((nxt[0] - 1.2, nxt[1]),
                             f"{rewards['goal']:+d}", theme.GOLD)
            self.board.celebrate()
            self.root.after(1400, self._show_outcome)
        elif event["outcome"] == "timeout":
            self.root.after(400, self._show_outcome)
        self.board.set_gate_open(self.session.gate_open())
        if self.show_policy:
            self._refresh_policy()
        self._update_hud()

    def _next_training_episode(self) -> None:
        if self.session.mode != "train" or self.menu.visible:
            return
        self.session.begin_next_episode()
        self.board.new_episode()
        self.board.set_gate_open(self.session.gate_open())
        self._update_hud()

    def _show_outcome(self) -> None:
        if self.session.outcome and not self.menu.visible:
            self.overlay.show(self.session.outcome,
                              round(self.session.score), self.session.steps)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    MazeMarioApp().run()


if __name__ == "__main__":
    main()
