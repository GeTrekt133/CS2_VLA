"""
Debug overlay for visualizing agent predictions.
Displays mouse delta, key probabilities, and game state on a transparent window.
"""

import threading
from typing import Dict, Set, Optional, Any
import tkinter as tk
from tkinter import font as tkfont


class DebugOverlay:
    """
    Transparent overlay window showing agent predictions.

    Displays:
    - Mouse delta (yaw, pitch) in degrees and pixels
    - Key probabilities as horizontal bars
    - Currently pressed keys highlighted
    - FPS / inference rate
    - Value estimate
    - Game state info
    """

    def __init__(
        self,
        width: int = 400,
        height: int = 520,
        x_offset: int = 10,
        y_offset: int = 100,
        alpha: float = 0.85,
        refresh_rate: int = 20,  # FPS
    ):
        """
        Initialize debug overlay.

        Args:
            width: Window width
            height: Window height
            x_offset: X position from left edge
            y_offset: Y position from top edge
            alpha: Window transparency (0-1)
            refresh_rate: Display refresh rate (Hz)
        """
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.alpha = alpha
        self.refresh_rate = refresh_rate

        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Data to display (thread-safe access via lock)
        self._mouse_delta = (0.0, 0.0)
        self._mouse_pixels = (0, 0)
        self._key_probs: Dict[str, float] = {}
        self._pressed_keys: Set[str] = set()
        self._fps = 0.0
        self._value = 0.0
        self._inference_ms = 0.0
        self._game_state: Dict[str, Any] = {}

        # Colors
        self.bg_color = '#1a1a1a'
        self.text_color = '#ffffff'
        self.accent_color = '#00ff88'
        self.bar_active = '#00ff88'
        self.bar_inactive = '#404040'
        self.bar_pressed = '#ff8800'

    def start(self):
        """Start overlay in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop overlay."""
        self._running = False
        if self.root:
            try:
                # Schedule destroy from Tk thread
                self.root.after(0, self._destroy)
            except:
                pass

    def _destroy(self):
        """Destroy root window (called from Tk thread)."""
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def _run_tk(self):
        """Tkinter main loop (runs in background thread)."""
        self.root = tk.Tk()
        self.root.title("CS2 VLA Agent")

        # Window setup
        self.root.geometry(f"{self.width}x{self.height}+{self.x_offset}+{self.y_offset}")
        self.root.attributes('-alpha', self.alpha)
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)  # Remove window borders

        # Make window click-through on Windows
        try:
            self.root.wm_attributes('-transparentcolor', '')
        except:
            pass

        # Canvas for drawing
        self.canvas = tk.Canvas(
            self.root,
            bg=self.bg_color,
            highlightthickness=0,
            width=self.width,
            height=self.height
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Allow dragging the window
        self._drag_data = {"x": 0, "y": 0}
        self.canvas.bind("<Button-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)

        # Start update loop
        self._schedule_update()

        # Run main loop
        try:
            self.root.mainloop()
        except:
            pass

    def _on_drag_start(self, event):
        """Start dragging window."""
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag_motion(self, event):
        """Handle window drag."""
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    def _schedule_update(self):
        """Schedule next display update."""
        if not self._running:
            return

        self._update_display()

        interval = int(1000 / self.refresh_rate)
        self.root.after(interval, self._schedule_update)

    def _update_display(self):
        """Redraw overlay content."""
        if self.canvas is None:
            return

        self.canvas.delete("all")

        with self._lock:
            mouse_delta = self._mouse_delta
            mouse_pixels = self._mouse_pixels
            key_probs = self._key_probs.copy()
            pressed_keys = self._pressed_keys.copy()
            fps = self._fps
            value = self._value
            inference_ms = self._inference_ms

        y = 10
        x_margin = 10

        # === Title ===
        self.canvas.create_text(
            x_margin, y,
            text="CS2 VLA Agent",
            fill=self.accent_color,
            font=("Consolas", 14, "bold"),
            anchor="nw"
        )
        y += 28

        # === FPS / Performance ===
        self.canvas.create_text(
            x_margin, y,
            text=f"Inference: {fps:.1f} Hz  |  {inference_ms:.1f} ms",
            fill=self.text_color,
            font=("Consolas", 10),
            anchor="nw"
        )
        y += 20

        # === Value estimate ===
        self.canvas.create_text(
            x_margin, y,
            text=f"Value: {value:.3f}",
            fill='#aaaaaa',
            font=("Consolas", 10),
            anchor="nw"
        )
        y += 25

        # === Mouse Delta ===
        self.canvas.create_text(
            x_margin, y,
            text="Mouse Delta",
            fill=self.text_color,
            font=("Consolas", 11, "bold"),
            anchor="nw"
        )
        y += 18

        # Yaw
        yaw_text = f"Δyaw:   {mouse_delta[0]:+8.2f}° → {mouse_pixels[0]:+5d} px"
        self.canvas.create_text(
            x_margin + 10, y,
            text=yaw_text,
            fill='#00ccff',
            font=("Consolas", 10),
            anchor="nw"
        )
        y += 18

        # Pitch
        pitch_text = f"Δpitch: {mouse_delta[1]:+8.2f}° → {mouse_pixels[1]:+5d} px"
        self.canvas.create_text(
            x_margin + 10, y,
            text=pitch_text,
            fill='#00ccff',
            font=("Consolas", 10),
            anchor="nw"
        )
        y += 28

        # === Key Probabilities ===
        self.canvas.create_text(
            x_margin, y,
            text="Key Probabilities",
            fill=self.text_color,
            font=("Consolas", 11, "bold"),
            anchor="nw"
        )
        y += 20

        # Sort by probability, show top keys
        sorted_keys = sorted(key_probs.items(), key=lambda x: -x[1])

        bar_width_max = 120
        bar_height = 14
        label_width = 100  # More space for key names

        for key, prob in sorted_keys[:15]:
            # Key label
            is_pressed = key in pressed_keys
            label_color = self.bar_pressed if is_pressed else self.text_color

            self.canvas.create_text(
                x_margin + 10, y,
                text=f"{key:>12}",
                fill=label_color,
                font=("Consolas", 9),
                anchor="nw"
            )

            # Probability bar - moved right to not overlap labels
            bar_x = x_margin + label_width + 10
            bar_width = int(prob * bar_width_max)

            # Background bar
            self.canvas.create_rectangle(
                bar_x, y + 2,
                bar_x + bar_width_max, y + bar_height,
                fill=self.bar_inactive,
                outline=""
            )

            # Filled bar
            bar_color = self.bar_pressed if is_pressed else self.bar_active
            if bar_width > 0:
                self.canvas.create_rectangle(
                    bar_x, y + 2,
                    bar_x + bar_width, y + bar_height,
                    fill=bar_color,
                    outline=""
                )

            # Probability text
            self.canvas.create_text(
                bar_x + bar_width_max + 8, y,
                text=f"{prob:.2f}",
                fill='#aaaaaa',
                font=("Consolas", 9),
                anchor="nw"
            )

            y += 18

        # === Pressed Keys Summary ===
        y += 5
        if pressed_keys:
            pressed_text = "Active: " + " ".join(sorted(pressed_keys))
        else:
            pressed_text = "Active: (none)"

        self.canvas.create_text(
            x_margin, y,
            text=pressed_text,
            fill=self.bar_pressed if pressed_keys else '#666666',
            font=("Consolas", 9),
            anchor="nw"
        )

    def update(
        self,
        mouse_delta: tuple = None,
        mouse_pixels: tuple = None,
        key_probs: Dict[str, float] = None,
        pressed_keys: Set[str] = None,
        fps: float = None,
        game_state: Dict[str, Any] = None,
    ):
        """
        Update overlay data (thread-safe).

        Args:
            mouse_delta: (yaw, pitch) in degrees
            mouse_pixels: (dx, dy) in pixels
            key_probs: Dict of key name -> probability
            pressed_keys: Set of currently pressed key names
            fps: Current inference FPS
            game_state: Additional game state info
        """
        with self._lock:
            if mouse_delta is not None:
                self._mouse_delta = mouse_delta
            if mouse_pixels is not None:
                self._mouse_pixels = mouse_pixels
            if key_probs is not None:
                self._key_probs = key_probs
            if pressed_keys is not None:
                self._pressed_keys = pressed_keys
            if fps is not None:
                self._fps = fps
            if game_state is not None:
                self._game_state = game_state
                if 'value' in game_state:
                    self._value = game_state['value']
                if 'inference_ms' in game_state:
                    self._inference_ms = game_state['inference_ms']


def test_overlay():
    """Test debug overlay."""
    import time
    import random

    print("Testing DebugOverlay...")
    print("Overlay will appear in top-left corner")
    print("Drag it to move. Press Ctrl+C to stop.")

    overlay = DebugOverlay()
    overlay.start()

    # Give tkinter time to initialize
    time.sleep(0.5)

    # Simulate data updates
    keys = ['forward', 'back', 'left', 'right', 'jump', 'crouch',
            'fire', 'reload', 'weapon1', 'weapon2']

    try:
        while True:
            # Random mouse delta
            yaw = random.uniform(-5, 5)
            pitch = random.uniform(-3, 3)
            dx = int(yaw / 0.022)
            dy = int(pitch / 0.022)

            # Random key probabilities
            key_probs = {k: random.random() for k in keys}

            # Some pressed keys
            pressed = {k for k, p in key_probs.items() if p > 0.5}

            overlay.update(
                mouse_delta=(yaw, pitch),
                mouse_pixels=(dx, dy),
                key_probs=key_probs,
                pressed_keys=pressed,
                fps=random.uniform(14, 18),
                game_state={
                    'value': random.uniform(-1, 1),
                    'inference_ms': random.uniform(20, 50),
                }
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping overlay...")
        overlay.stop()
        print("Done!")


if __name__ == "__main__":
    test_overlay()
