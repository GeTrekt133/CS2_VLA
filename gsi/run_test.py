#!/usr/bin/env python3
"""
Test GSI server - simple test to verify GSI is working.

Usage:
    python gsi/run_test.py

This will start the GSI server and print incoming data.
Use this to verify CS2 is sending data correctly.
"""
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsi.server.http_server import GSIServer, GSIPayload
from gsi.adapter.state_adapter import GSIStateAdapter


def on_update(payload: GSIPayload):
    """Print GSI update info."""
    if not payload.is_in_game():
        print("[GSI] Not in game (menu/loading)")
        return

    adapter = GSIStateAdapter()
    snapshot = adapter.parse(payload)

    if snapshot is None:
        print("[GSI] Could not parse payload")
        return

    print(f"\n{'='*50}")
    print(f"[GSI Update] Round {snapshot.round_num} | Phase: {snapshot.phase}")
    print(f"  Player: {snapshot.side} | HP: {snapshot.hp} | Armor: {snapshot.armor}")
    print(f"  Weapon: {snapshot.weapon} | Ammo: {snapshot.ammo}")
    print(f"  Alive: CT={snapshot.ct_alive} T={snapshot.t_alive}")
    print(f"  Time: {snapshot.round_time_left:.1f}s | Bomb: {'PLANTED' if snapshot.bomb_planted else 'no'}")
    print(f"  Inventory: {snapshot.weapon_list}")

    # Print state_vec shape to verify
    state_vec = adapter.to_state_vec(snapshot)
    print(f"  State vec shape: {state_vec.shape}")


def main():
    print("=" * 60)
    print("  CS2 GSI Test Server")
    print("=" * 60)
    print()
    print("  1. Copy gamestate_integration_cs2nn.cfg to CS2 cfg folder:")
    print("     C:\\Program Files (x86)\\Steam\\steamapps\\common\\")
    print("     Counter-Strike Global Offensive\\game\\csgo\\cfg\\")
    print()
    print("  2. Launch CS2 and join any match")
    print()
    print("  3. You should see game state updates below")
    print()
    print("=" * 60)
    print()

    server = GSIServer(port=3000, verbose=False)
    server.register_callback(on_update)
    server.start()

    print(f"[Server] Listening on http://127.0.0.1:3000")
    print("[Server] Waiting for CS2 data... (Ctrl+C to stop)")
    print()

    try:
        while True:
            time.sleep(1)
            if server.received_count > 0 and server.received_count % 10 == 0:
                print(f"[Stats] Received {server.received_count} updates")

    except KeyboardInterrupt:
        print("\n[!] Stopping...")

    finally:
        server.stop()


if __name__ == "__main__":
    main()
