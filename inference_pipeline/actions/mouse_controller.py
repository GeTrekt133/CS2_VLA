"""
Mouse controller for converting model predictions to mouse movement.
"""

from typing import Tuple
from .input_sender import InputSender


class MouseController:
    """
    Converts model mouse delta predictions to actual mouse movement.

    Model outputs yaw/pitch deltas in degrees, which are converted
    to pixel movements for SendInput.
    """

    def __init__(
        self,
        input_sender: InputSender,
        sensitivity: float = 1.0,
        degrees_per_pixel: float = 0.022,
        invert_y: bool = False,
    ):
        """
        Initialize mouse controller.

        Args:
            input_sender: InputSender instance for sending mouse events
            sensitivity: Multiplier for mouse movement
            degrees_per_pixel: Conversion factor (CS2 default ~0.022)
            invert_y: Whether to invert vertical movement
        """
        self.input_sender = input_sender
        self.sensitivity = sensitivity
        self.degrees_per_pixel = degrees_per_pixel
        self.invert_y = invert_y

        # Movement accumulator for sub-pixel precision
        self._accum_x = 0.0
        self._accum_y = 0.0

        # Stats
        self._total_movements = 0

    def apply_delta(self, yaw_delta: float, pitch_delta: float) -> Tuple[int, int]:
        """
        Apply mouse delta from model prediction.

        Args:
            yaw_delta: Horizontal rotation delta in degrees (positive = look right)
            pitch_delta: Vertical rotation delta in degrees (positive = look down)

        Returns:
            (dx, dy) actual pixel movement applied
        """
        # Convert degrees to pixels
        # Formula: pixels = degrees / degrees_per_pixel
        raw_dx = yaw_delta / self.degrees_per_pixel * self.sensitivity
        raw_dy = pitch_delta / self.degrees_per_pixel * self.sensitivity

        # Invert Y if configured
        if self.invert_y:
            raw_dy = -raw_dy

        # Accumulate for sub-pixel precision
        self._accum_x += raw_dx
        self._accum_y += raw_dy

        # Extract integer part
        dx = int(self._accum_x)
        dy = int(self._accum_y)

        # Keep fractional part for next frame
        self._accum_x -= dx
        self._accum_y -= dy

        # Send mouse movement
        if dx != 0 or dy != 0:
            self.input_sender.send_mouse_move(dx, dy)
            self._total_movements += 1

        return (dx, dy)

    def reset_accumulator(self):
        """Reset sub-pixel accumulator."""
        self._accum_x = 0.0
        self._accum_y = 0.0

    def set_sensitivity(self, sensitivity: float):
        """Update mouse sensitivity."""
        self.sensitivity = sensitivity

    @property
    def total_movements(self) -> int:
        """Total number of mouse movements sent."""
        return self._total_movements


def test_mouse_controller():
    """Test mouse controller."""
    import time

    print("Testing MouseController...")
    print("WARNING: This will move the mouse!")
    print("Press Ctrl+C within 3 seconds to cancel...")

    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("Cancelled")
        return

    sender = InputSender()
    controller = MouseController(
        input_sender=sender,
        sensitivity=0.5,  # Reduced sensitivity for testing
        degrees_per_pixel=0.022
    )

    # Simulate some model predictions
    print("\nSimulating model predictions...")
    deltas = [
        (5.0, 0.0),   # Look right
        (0.0, 3.0),   # Look down
        (-5.0, 0.0),  # Look left
        (0.0, -3.0),  # Look up
    ]

    for yaw, pitch in deltas:
        dx, dy = controller.apply_delta(yaw, pitch)
        print(f"  Delta ({yaw:.1f}, {pitch:.1f}) -> Pixels ({dx}, {dy})")
        time.sleep(0.3)

    print(f"\nTotal movements: {controller.total_movements}")
    print("Test complete!")


if __name__ == "__main__":
    test_mouse_controller()
