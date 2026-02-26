"""
Dataset JSON Builder — produces dataset.json in the format expected by
final_model/DatasetIntent.py CSRoundDataset.

Schema:
{
  "demos": [
    {
      "demo_path": "D:/dataset/frames/demo_name",   # dir with tick_N.jpg
      "audio_path": "D:/dataset/audio/demo_name",   # dir with round_N.wav
      "rounds": [
        {
          "round_id": 0,
          "start_tick": 4568,
          "end_tick": 8340,
          "states": [{tick, keys, mouse, hp, armor, ...}],
          "events": [{type, tick, ...}]
        }
      ]
    }
  ]
}
"""

import json
import os
from typing import List, Dict, Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to Python native types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DatasetJsonBuilder:

    def save(self, demo_entries: List[Dict[str, Any]], output_path: str):
        """Save the dataset JSON to disk."""
        dataset = {"demos": demo_entries}

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

        total_rounds = sum(len(d.get('rounds', [])) for d in demo_entries)
        print(f"[JSON] Saved {len(demo_entries)} demos, {total_rounds} rounds → {output_path}")

    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load existing dataset.json and return demo entries."""
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('demos', [])

    def merge(self, existing_path: str, new_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new entries with existing dataset, deduplicating by demo_path."""
        existing = self.load(existing_path)
        existing_paths = {e['demo_path'] for e in existing}

        for entry in new_entries:
            if entry['demo_path'] not in existing_paths:
                existing.append(entry)
                existing_paths.add(entry['demo_path'])

        return existing

    def validate(self, path: str) -> bool:
        """Validate dataset JSON structure for CSRoundDataset compatibility."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            demos = data.get('demos', [])
            if not demos:
                print("[Validate] No demos found")
                return False

            for i, demo in enumerate(demos):
                if 'demo_path' not in demo:
                    print(f"[Validate] Demo {i}: missing demo_path")
                    return False
                if 'rounds' not in demo:
                    print(f"[Validate] Demo {i}: missing rounds")
                    return False

                for j, rnd in enumerate(demo['rounds']):
                    required = ['round_id', 'start_tick', 'states']
                    for key in required:
                        if key not in rnd:
                            print(f"[Validate] Demo {i} round {j}: missing {key}")
                            return False

                    if rnd['states']:
                        state = rnd['states'][0]
                        # Required by DatasetIntent.py state_vec + action history
                        state_keys = [
                            'tick', 'keys', 'mouse', 'hp', 'armor', 'helmet',
                            'weapon', 'weapon_list', 'side',
                            'ct_alive', 't_alive',
                        ]
                        for key in state_keys:
                            if key not in state:
                                print(f"[Validate] Demo {i} round {j} state: missing {key}")
                                return False
                        # Optional but recommended (DatasetIntent uses .get() with defaults)
                        optional_keys = ['ammo', 'round_time_left', 'bomb_planted', 'freeze_time']
                        missing_optional = [k for k in optional_keys if k not in state]
                        if missing_optional:
                            print(f"[Validate] Demo {i} round {j} state: missing optional {missing_optional}")

            print(f"[Validate] OK: {len(demos)} demos, "
                  f"{sum(len(d['rounds']) for d in demos)} rounds")
            return True

        except Exception as e:
            print(f"[Validate] Error: {e}")
            return False
