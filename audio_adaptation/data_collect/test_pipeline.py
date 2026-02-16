"""
Quick test for CS2 dataset collection pipeline.

Tests:
1. Import all modules
2. Parse a single demo (dry run)
3. Check output JSON structure
"""

import os
import sys
import json


def test_imports():
    """Test that all required modules are available."""
    print("Testing imports...")

    try:
        import demoparser2
        print("  ✓ demoparser2")
    except ImportError:
        print("  ✗ demoparser2 not found. Install: pip install demoparser2")
        return False

    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas not found. Install: pip install pandas")
        return False

    try:
        import tqdm
        print("  ✓ tqdm")
    except ImportError:
        print("  ✗ tqdm not found. Install: pip install tqdm")
        return False

    try:
        import keyboard
        print("  ✓ keyboard")
    except ImportError:
        print("  ✗ keyboard not found. Install: pip install keyboard")
        return False

    try:
        import pyautogui
        print("  ✓ pyautogui")
    except ImportError:
        print("  ✗ pyautogui not found. Install: pip install pyautogui")
        return False

    try:
        import obsws_python
        print("  ✓ obsws-python")
    except ImportError:
        print("  ✗ obsws-python not found. Install: pip install obsws-python")
        return False

    print("\nAll imports successful!\n")
    return True


def test_parser(demo_path: str, steam_id: int):
    """Test universal demo parser on a single demo."""
    print(f"Testing parser on: {demo_path}")

    if not os.path.exists(demo_path):
        print(f"  ✗ Demo file not found: {demo_path}")
        return False

    try:
        from universal_demo_parser import UniversalDemoParser

        parser = UniversalDemoParser(target_steam_id=steam_id)
        demo_data = parser.parse_demo(demo_path, skip_first_rounds=3)

        # Check structure
        assert 'demo_name' in demo_data
        assert 'player_steam_id' in demo_data
        assert 'rounds' in demo_data

        num_rounds = len(demo_data['rounds'])
        print(f"  ✓ Parsed {num_rounds} rounds")

        if num_rounds > 0:
            first_round = demo_data['rounds'][0]
            num_states = len(first_round['states'])
            num_events = len(first_round['events'])

            print(f"  ✓ Round 0: {num_states} states, {num_events} events")

            # Check state structure
            if num_states > 0:
                state = first_round['states'][0]
                required_keys = ['tick', 'keys', 'mouse', 'hp', 'armor', 'weapon']
                for key in required_keys:
                    if key not in state:
                        print(f"  ✗ Missing key in state: {key}")
                        return False

                print(f"  ✓ State structure valid")

            # Check event types
            if num_events > 0:
                event_types = set(e['type'] for e in first_round['events'])
                print(f"  ✓ Event types: {', '.join(event_types)}")

        print("\nParser test successful!\n")
        return True

    except Exception as e:
        print(f"  ✗ Parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_output(json_path: str):
    """Test that output JSON is valid."""
    print(f"Testing JSON output: {json_path}")

    if not os.path.exists(json_path):
        print(f"  ✗ JSON file not found: {json_path}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'demos' in data
        num_demos = len(data['demos'])

        total_rounds = sum(len(demo['rounds']) for demo in data['demos'])
        total_states = sum(
            len(r['states'])
            for demo in data['demos']
            for r in demo['rounds']
        )
        total_events = sum(
            len(r['events'])
            for demo in data['demos']
            for r in demo['rounds']
        )

        print(f"  ✓ {num_demos} demos")
        print(f"  ✓ {total_rounds} rounds")
        print(f"  ✓ {total_states} states")
        print(f"  ✓ {total_events} events")

        print("\nJSON output test successful!\n")
        return True

    except Exception as e:
        print(f"  ✗ JSON test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CS2 Dataset Collection Pipeline - Tests")
    print("="*60 + "\n")

    # Test 1: Imports
    if not test_imports():
        print("\n✗ Import test failed. Install missing dependencies:")
        print("  pip install -r requirements.txt")
        return

    # Test 2: Parser (if demo path provided)
    import argparse
    parser = argparse.ArgumentParser(description="Test CS2 dataset pipeline")
    parser.add_argument('--demo', type=str, help='Path to test .dem file')
    parser.add_argument('--steam-id', type=int, default=76561198386265483, help='Steam ID')
    parser.add_argument('--json', type=str, help='Path to test JSON output')

    args = parser.parse_args()

    if args.demo:
        if not test_parser(args.demo, args.steam_id):
            print("\n✗ Parser test failed")
            return

    # Test 3: JSON output (if provided)
    if args.json:
        if not test_json_output(args.json):
            print("\n✗ JSON output test failed")
            return

    print("="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    main()
