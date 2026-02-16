"""
Keyboard controller for converting model predictions to key presses.
"""

from typing import Dict, Set, List, Tuple, Optional
import numpy as np
import torch

from .input_sender import InputSender


class KeyboardController:
    """
    Converts model key predictions to actual key presses.

    Manages key state (pressed/released) and handles hysteresis
    to prevent rapid key toggling.
    """

    def __init__(
        self,
        input_sender: InputSender,
        key_mapping: Dict[int, str],
        threshold: float = 0.5,
        hysteresis: float = 0.1,
    ):
        """
        Initialize keyboard controller.

        Args:
            input_sender: InputSender instance for sending key events
            key_mapping: Map from model output index to key name
            threshold: Probability threshold for pressing a key
            hysteresis: Hysteresis for key release (release at threshold - hysteresis)
        """
        self.input_sender = input_sender
        self.key_mapping = key_mapping
        self.threshold = threshold
        self.hysteresis = hysteresis

        # Currently pressed keys
        self._pressed_keys: Set[str] = set()

        # Stats
        self._total_presses = 0
        self._total_releases = 0

    def apply_actions(
        self,
        key_logits: torch.Tensor,
    ) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
        """
        Apply key actions from model predictions.

        Args:
            key_logits: Raw logits from model (N,) tensor

        Returns:
            (probs, actions_taken):
            - probs: (N,) numpy array of probabilities after sigmoid
            - actions_taken: List of (key, action, prob) for logging
        """
        # Convert to probabilities
        probs = torch.sigmoid(key_logits).cpu().numpy()
        actions_taken = []

        for idx, prob in enumerate(probs):
            key = self.key_mapping.get(idx)
            if key is None:
                continue

            is_pressed = key in self._pressed_keys

            # Hysteresis: press at threshold, release at threshold - hysteresis
            press_threshold = self.threshold
            release_threshold = self.threshold - self.hysteresis

            if prob > press_threshold and not is_pressed:
                # Press key
                self.input_sender.send_key(key, down=True)
                self._pressed_keys.add(key)
                self._total_presses += 1
                actions_taken.append((key, 'press', float(prob)))

            elif prob < release_threshold and is_pressed:
                # Release key
                self.input_sender.send_key(key, down=False)
                self._pressed_keys.discard(key)
                self._total_releases += 1
                actions_taken.append((key, 'release', float(prob)))

        return probs, actions_taken

    def press_key(self, key: str):
        """Manually press a key."""
        if key not in self._pressed_keys:
            self.input_sender.send_key(key, down=True)
            self._pressed_keys.add(key)
            self._total_presses += 1

    def release_key(self, key: str):
        """Manually release a key."""
        if key in self._pressed_keys:
            self.input_sender.send_key(key, down=False)
            self._pressed_keys.discard(key)
            self._total_releases += 1

    def release_all(self):
        """Release all pressed keys (call on shutdown)."""
        for key in list(self._pressed_keys):
            self.input_sender.send_key(key, down=False)
            self._total_releases += 1
        self._pressed_keys.clear()

    def set_threshold(self, threshold: float):
        """Update key press threshold."""
        self.threshold = threshold

    @property
    def pressed_keys(self) -> Set[str]:
        """Currently pressed keys."""
        return self._pressed_keys.copy()

    @property
    def stats(self) -> Dict[str, int]:
        """Get controller statistics."""
        return {
            "total_presses": self._total_presses,
            "total_releases": self._total_releases,
            "currently_pressed": len(self._pressed_keys),
        }


def test_keyboard_controller():
    """Test keyboard controller (simulated, no real input)."""
    print("Testing KeyboardController (simulated)...")

    # Create a mock input sender that just prints
    class MockInputSender:
        def send_key(self, key, down):
            action = "DOWN" if down else "UP"
            print(f"  [{action}] {key}")

    sender = MockInputSender()

    # Simple key mapping
    key_mapping = {
        0: 'w',      # forward
        1: 's',      # back
        2: 'a',      # left
        3: 'd',      # right
        4: 'space',  # jump
    }

    controller = KeyboardController(
        input_sender=sender,
        key_mapping=key_mapping,
        threshold=0.5,
        hysteresis=0.1,
    )

    # Simulate some predictions
    print("\nSimulating predictions...")

    # Frame 1: W and A pressed
    print("\nFrame 1: W=0.8, S=0.2, A=0.7, D=0.3, Space=0.1")
    logits = torch.tensor([0.8, 0.2, 0.7, 0.3, 0.1])
    probs, actions = controller.apply_actions(logits)
    print(f"  Pressed: {controller.pressed_keys}")

    # Frame 2: W still pressed, A released, D pressed
    print("\nFrame 2: W=0.6, S=0.2, A=0.3, D=0.8, Space=0.1")
    logits = torch.tensor([0.6, 0.2, 0.3, 0.8, 0.1])
    probs, actions = controller.apply_actions(logits)
    print(f"  Pressed: {controller.pressed_keys}")

    # Frame 3: Jump!
    print("\nFrame 3: W=0.6, S=0.2, A=0.2, D=0.6, Space=0.9")
    logits = torch.tensor([0.6, 0.2, 0.2, 0.6, 0.9])
    probs, actions = controller.apply_actions(logits)
    print(f"  Pressed: {controller.pressed_keys}")

    # Release all
    print("\nReleasing all keys...")
    controller.release_all()
    print(f"  Pressed: {controller.pressed_keys}")

    print(f"\nStats: {controller.stats}")
    print("\nTest complete!")


if __name__ == "__main__":
    test_keyboard_controller()
