"""
DataExporter — converts synthetic scenario results to DatasetIntent JSON format.

Output format matches train_dataset.json exactly:
{
  "demos": [
    {
      "demo_path": "path/to/frames/scenario_id",
      "audio_path": "path/to/audio/scenario_id",
      "server_start_tick": null,
      "rounds": [
        {
          "round_id": 1,
          "start_tick": 1000,
          "end_tick": 1252,
          "winner": "T",
          "player_team": "T",
          "score": [0, 0],
          "events": [],
          "states": [ ... ],
          "drift_info": {},
          "frame_coverage": 1.0
        }
      ]
    }
  ]
}

Each scenario becomes one "demo" with one "round".
"""

import os
import json
from typing import List, Dict, Any, Optional


class DataExporter:
    """Exports synthetic data to DatasetIntent-compatible JSON."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.dataset_path = os.path.join(output_dir, "synthetic_dataset.json")

    def export(
        self,
        scenario_results: List[Dict[str, Any]],
        append: bool = False,
    ) -> str:
        """Export scenario results to dataset JSON.

        Args:
            scenario_results: List of dicts from CS2DataEngine.run_scenario()
            append: If True, append to existing dataset file

        Returns:
            Path to the saved JSON file
        """
        existing_demos = []
        if append and os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            existing_demos = existing.get('demos', [])

        new_demos = []
        for result in scenario_results:
            demo = self._result_to_demo(result)
            if demo:
                new_demos.append(demo)

        all_demos = existing_demos + new_demos

        dataset = {"demos": all_demos}

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"[DataExporter] Saved {len(new_demos)} new scenarios "
              f"({len(all_demos)} total) to {self.dataset_path}")
        return self.dataset_path

    def _result_to_demo(self, result: Dict[str, Any]) -> Optional[Dict]:
        """Convert a single scenario result to a demo entry."""
        states = result.get("states", [])
        if not states:
            return None

        start_tick = states[0]["tick"]
        end_tick = states[-1]["tick"]

        # Build round entry
        round_entry = {
            "round_id": 1,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "winner": "T",
            "player_team": "T",
            "score": [0, 0],
            "events": self._build_events(result),
            "states": states,
            "drift_info": {},
            "frame_coverage": 1.0,
        }

        demo = {
            "demo_path": result["demo_path"],
            "audio_path": result.get("audio_path"),
            "server_start_tick": None,
            "rounds": [round_entry],
        }

        return demo

    def _build_events(self, result: Dict[str, Any]) -> List[Dict]:
        """Build events list.

        For flick/transfer: kill event when crosshair reaches target + shoots.
        """
        events = []
        states = result.get("states", [])
        if not states:
            return events

        scenario_type = result.get("scenario_type", "")

        if scenario_type in ("flick", "transfer"):
            # Find first tick with MOUSE_LEFT — that's the kill
            for s in states:
                if "MOUSE_LEFT" in s["keys"]:
                    events.append({
                        "type": "kill",
                        "tick": s["tick"],
                        "weapon": "ak47",
                        "headshot": True,
                    })
                    break

        return events

    def merge_with_real_dataset(
        self,
        real_dataset_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Merge synthetic dataset with real demo dataset.

        Args:
            real_dataset_path: Path to existing train_dataset.json
            output_path: Output path (default: merged_dataset.json in output_dir)

        Returns:
            Path to merged JSON
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "merged_dataset.json")

        # Load real
        with open(real_dataset_path, 'r', encoding='utf-8') as f:
            real = json.load(f)

        # Load synthetic
        if not os.path.exists(self.dataset_path):
            print("[DataExporter] No synthetic dataset found, copying real dataset")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(real, f, indent=2, ensure_ascii=False)
            return output_path

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            synthetic = json.load(f)

        # Merge
        merged = {
            "demos": real.get("demos", []) + synthetic.get("demos", [])
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        n_real = len(real.get("demos", []))
        n_synth = len(synthetic.get("demos", []))
        print(f"[DataExporter] Merged: {n_real} real + {n_synth} synthetic = "
              f"{n_real + n_synth} total demos -> {output_path}")
        return output_path
