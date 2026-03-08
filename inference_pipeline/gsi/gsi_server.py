"""
CS2 Game State Integration (GSI) HTTP server.

CS2 sends POST requests with JSON payload to this server.
We parse the game state and build the 100-dim state vector
for the TemporalCrossTransformer.

State vector (100 dims):
  [0-11]   12 scalars: hp, armor, helmet, ammo, ct_alive, t_alive,
                        round_time_left, bomb_planted, freeze_time,
                        defuser, score_ct, score_t
  [12-13]  2 one-hot:  side (CT=1,0 / T=0,1)
  [14-56]  43 one-hot: current weapon
  [57-99]  43 multi-hot: weapon inventory

To enable GSI in CS2, create a file:
  C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/
    game/csgo/cfg/gamestate_integration_cs2nn.cfg

Contents:
  "CS2 NN Agent"
  {
    "uri"               "http://127.0.0.1:3000"
    "timeout"           "5.0"
    "auth"
    {
        "token"         "cs2nn_secret_token"
    }
    "data"
    {
        "provider"          "1"
        "map"               "1"
        "map_round_wins"    "1"
        "player_id"         "1"
        "player_state"      "1"
        "player_weapons"    "1"
        "player_match_stats" "1"
        "round"             "1"
        "allplayers_id"     "1"
        "allplayers_state"  "1"
        "phase_countdowns"  "1"
    }
  }
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Callable
import numpy as np


# Weapons list: 43 items matching DatasetIntent.py WEAPONS
WEAPONS = [
    'p2000', 'usp-s', 'p250', 'five-seven', 'glock-18', 'tec-9',
    'cz75-auto', 'dual berettas', 'desert eagle', 'm249',
    'r8 revolver', 'mp9', 'mac-10', 'pp-bizon', 'mp7',
    'ump-45', 'p90', 'mp5-sd', 'famas', 'galil ar', 'sawed-off',
    'm4a4', 'm4a1-s', 'ak-47', 'aug', 'sg 553', 'ssg 08',
    'awp', 'scar-20', 'g3sg1', 'nova', 'xm1014', 'mag-7',
    'negev', 'knife', 'high explosive grenade', 'flashbang',
    'smoke grenade', 'decoy grenade', 'molotov',
    'incendiary grenade', 'c4 explosive', 'None'
]
WEAPON_TO_IDX = {w: i for i, w in enumerate(WEAPONS)}

# CS2 weapon name → our canonical name (lowercased)
CS2_WEAPON_ALIASES = {
    # Pistols
    'weapon_hkp2000': 'p2000',
    'weapon_usp_silencer': 'usp-s',
    'weapon_p250': 'p250',
    'weapon_fiveseven': 'five-seven',
    'weapon_glock': 'glock-18',
    'weapon_tec9': 'tec-9',
    'weapon_cz75a': 'cz75-auto',
    'weapon_elite': 'dual berettas',
    'weapon_deagle': 'desert eagle',
    'weapon_revolver': 'r8 revolver',
    # SMGs
    'weapon_mp9': 'mp9',
    'weapon_mac10': 'mac-10',
    'weapon_bizon': 'pp-bizon',
    'weapon_mp7': 'mp7',
    'weapon_ump45': 'ump-45',
    'weapon_p90': 'p90',
    'weapon_mp5sd': 'mp5-sd',
    # Rifles
    'weapon_famas': 'famas',
    'weapon_galilar': 'galil ar',
    'weapon_m4a1': 'm4a4',
    'weapon_m4a1_silencer': 'm4a1-s',
    'weapon_ak47': 'ak-47',
    'weapon_aug': 'aug',
    'weapon_sg556': 'sg 553',
    'weapon_ssg08': 'ssg 08',
    'weapon_awp': 'awp',
    'weapon_scar20': 'scar-20',
    'weapon_g3sg1': 'g3sg1',
    # Shotguns
    'weapon_sawedoff': 'sawed-off',
    'weapon_nova': 'nova',
    'weapon_xm1014': 'xm1014',
    'weapon_mag7': 'mag-7',
    # Heavy
    'weapon_m249': 'm249',
    'weapon_negev': 'negev',
    # Knife (all variants map to 'knife')
    'weapon_knife': 'knife',
    'weapon_knife_t': 'knife',
    'weapon_knife_ct': 'knife',
    'weapon_knifegg': 'knife',
    'weapon_bayonet': 'knife',
    'weapon_knife_karambit': 'knife',
    'weapon_knife_m9_bayonet': 'knife',
    'weapon_knife_butterfly': 'knife',
    'weapon_knife_flip': 'knife',
    'weapon_knife_gut': 'knife',
    'weapon_knife_tactical': 'knife',
    'weapon_knife_falchion': 'knife',
    'weapon_knife_survival_bowie': 'knife',
    'weapon_knife_push': 'knife',
    'weapon_knife_ursus': 'knife',
    'weapon_knife_gypsy_jackknife': 'knife',
    'weapon_knife_stiletto': 'knife',
    'weapon_knife_widowmaker': 'knife',
    'weapon_knife_css': 'knife',
    'weapon_knife_cord': 'knife',
    'weapon_knife_canis': 'knife',
    'weapon_knife_outdoor': 'knife',
    'weapon_knife_skeleton': 'knife',
    'weapon_knife_kukri': 'knife',
    # Grenades
    'weapon_hegrenade': 'high explosive grenade',
    'weapon_flashbang': 'flashbang',
    'weapon_smokegrenade': 'smoke grenade',
    'weapon_decoy': 'decoy grenade',
    'weapon_molotov': 'molotov',
    'weapon_incgrenade': 'incendiary grenade',
    # Bomb
    'weapon_c4': 'c4 explosive',
}


# Local round timer state (fallback when phase_countdowns unavailable)
_round_timer_state = {
    'phase': '',
    'phase_start': 0.0,
    'bomb_planted': False,
    'bomb_plant_time': 0.0,
}

# Round durations in seconds
_PHASE_DURATIONS = {
    'freezetime': 15.0,
    'live': 115.0,
    'over': 7.0,
}

_BOMB_TIMER = 40.0  # C4 detonation timer in competitive





def _estimate_round_time(phase: str, bomb_planted: bool) -> float:
    """Estimate normalized round_time_left from local timer."""
    now = time.time()
    if phase != _round_timer_state['phase']:
        _round_timer_state['phase'] = phase
        _round_timer_state['phase_start'] = now
        _round_timer_state['bomb_planted'] = False  # reset on phase change

    # Track bomb plant moment
    if bomb_planted and not _round_timer_state['bomb_planted']:
        _round_timer_state['bomb_planted'] = True
        _round_timer_state['bomb_plant_time'] = now

    # Bomb planted → countdown from 40s
    if bomb_planted and _round_timer_state['bomb_planted']:
        elapsed = now - _round_timer_state['bomb_plant_time']
        remaining = max(0.0, _BOMB_TIMER - elapsed)
        return remaining / 115.0

    duration = _PHASE_DURATIONS.get(phase, 0.0)
    if duration <= 0:
        return 0.0
    elapsed = now - _round_timer_state['phase_start']
    remaining = max(0.0, duration - elapsed)
    return remaining / 115.0  # always normalize by round duration (115s)


def parse_gsi_json(data: dict, auth_token: str) -> Optional[np.ndarray]:
    """
    Parse CS2 GSI JSON payload and return state vector (100,).

    Returns None if auth fails or required data is missing.
    """
    # Auth check
    auth = data.get('auth', {})
    if auth.get('token', '') != auth_token:
        return None

    state_vec = np.zeros(100, dtype=np.float32)

    # === Scalars [0-11] ===
    player = data.get('player', {})
    player_state = player.get('state', {})
    if not player_state:
        return state_vec  # Not in match — return zeros

    # Fix steamid: if spectating teammate (dead), use dead state
    my_steamid = data.get('provider', {}).get('steamid', '')
    observed_steamid = player.get('steamid', '')
    is_spectating_teammate = (my_steamid and observed_steamid
                               and my_steamid != observed_steamid)

    if is_spectating_teammate:
        # Player is dead, watching teammate — force dead state
        state_vec[0] = 0.0  # hp = 0
        state_vec[1] = 0.0  # armor = 0
        state_vec[2] = 0.0  # helmet = 0
    else:
        # 0: hp (normalized 0-1)
        state_vec[0] = player_state.get('health', 0) / 100.0
        # 1: armor (normalized 0-1)
        state_vec[1] = player_state.get('armor', 0) / 100.0
        # 2: helmet
        state_vec[2] = float(bool(player_state.get('helmet', False)))
    # 3: ammo (current clip, normalized 0-1; approximate max 100)
    #    We'll fill this from the active weapon below
    # 4: ct_alive, 5: t_alive — filled externally by AliveDigitDetector (MNIST-like CNN)
    #    Left at 0.0 here; the inference pipeline sets them from screen capture.
    map_data = data.get('map', {})
    ct_data = map_data.get('team_ct', {})
    t_data = map_data.get('team_t', {})
    # 6: round_time_left (from phase_countdowns if available, else local timer)
    round_data = data.get('round', {})
    phase = round_data.get('phase', '')
    bomb_planted = round_data.get('bomb', '') == 'planted'
    phase_countdowns = data.get('phase_countdowns', {})
    phase_ends_in = phase_countdowns.get('phase_ends_in', '')
    if phase_ends_in != '':
        try:
            state_vec[6] = float(phase_ends_in) / 115.0
        except (ValueError, TypeError):
            state_vec[6] = 0.0
    else:
        # Local timer fallback: track phase transitions + bomb plant
        state_vec[6] = _estimate_round_time(phase, bomb_planted)
    # 7: bomb_planted
    state_vec[7] = float(bomb_planted)
    # 8: freeze_time
    state_vec[8] = float(phase == 'freezetime')
    # 9: defuser (only own player)
    if not is_spectating_teammate:
        state_vec[9] = float(bool(player_state.get('defusekit', False)))
    # 10: score_ct
    score_ct = ct_data.get('score', 0)
    state_vec[10] = score_ct / 16.0  # normalized 0-1 (max 16 rounds to win)
    # 11: score_t
    score_t = t_data.get('score', 0)
    state_vec[11] = score_t / 16.0

    # === Side [12-13]: one-hot [CT, T] ===
    player_team = player.get('team', 'CT')
    if player_team == 'CT':
        state_vec[12] = 1.0
    else:
        state_vec[13] = 1.0

    # === Parse weapons (skip if spectating teammate — not our weapons) ===
    if not is_spectating_teammate:
        weapons_data = player.get('weapons', {})

        active_weapon_name = None
        weapon_inventory = set()

        for slot_key, weapon_info in weapons_data.items():
            weapon_name_raw = weapon_info.get('name', '').lower()
            canonical = CS2_WEAPON_ALIASES.get(weapon_name_raw, None)
            if canonical is None:
                canonical = 'None'

            if weapon_info.get('state', '') == 'active':
                active_weapon_name = canonical
                ammo_max = weapon_info.get('ammo_clip_max', 1) or 1
                state_vec[3] = weapon_info.get('ammo_clip', 0) / max(ammo_max, 1)

            if canonical != 'None':
                weapon_inventory.add(canonical)

        # === Current weapon [14-56]: one-hot (43 dims) ===
        if active_weapon_name and active_weapon_name in WEAPON_TO_IDX:
            state_vec[14 + WEAPON_TO_IDX[active_weapon_name]] = 1.0
        else:
            state_vec[14 + WEAPON_TO_IDX['None']] = 1.0  # idx 42

        # === Weapon inventory [57-99]: multi-hot (43 dims) ===
        for wname in weapon_inventory:
            if wname in WEAPON_TO_IDX:
                state_vec[57 + WEAPON_TO_IDX[wname]] = 1.0

    return state_vec


class GSIRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for CS2 GSI POST requests."""

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode('utf-8'))

            # Dump raw JSONs to single JSONL file (append mode)
            has_player_state = 'state' in data.get('player', {})
            self.server._request_count += 1
            if has_player_state and self.server._request_count % 50 == 0:
                with open("gsi_raw_dump.jsonl", 'a', encoding='utf-8') as df:
                    df.write(json.dumps(data) + '\n')
                print(f"[GSI] Raw JSON appended to gsi_raw_dump.jsonl (req #{self.server._request_count})")

            # Debug: print allplayers and phase_countdowns if present
            allplayers = data.get('allplayers')
            phase_countdowns = data.get('phase_countdowns')
            round_info = data.get('round', {})
            map_info = data.get('map', {})
            if self.server._request_count % 20 == 0 or allplayers or phase_countdowns:
                print(f"\n[GSI DEBUG] req #{self.server._request_count}")
                print(f"  round: {json.dumps(round_info)}")
                print(f"  map.team_ct: {json.dumps(map_info.get('team_ct', {}))}")
                print(f"  map.team_t: {json.dumps(map_info.get('team_t', {}))}")
                print(f"  allplayers: {json.dumps(allplayers) if allplayers else 'NOT PRESENT'}")
                print(f"  phase_countdowns: {json.dumps(phase_countdowns) if phase_countdowns else 'NOT PRESENT'}")

            state_vec = parse_gsi_json(data, self.server.auth_token)
            p_state = data.get('player', {}).get('state', {})
            my_sid = data.get('provider', {}).get('steamid', '')
            obs_sid = data.get('player', {}).get('steamid', '')
            spectating = my_sid and obs_sid and my_sid != obs_sid
            money = p_state.get('money', '?')
            equip = p_state.get('equip_value', '?')
            # Only update money/equip if it's our own player
            if not spectating:
                self.server._last_money = money
                self.server._last_equip = equip
            spec_tag = " [SPECTATING]" if spectating else ""
            print(f"[GSI] req #{self.server._request_count} | parse={'OK' if state_vec is not None else 'NONE'} | money=${self.server._last_money} equip=${self.server._last_equip} | alive={p_state.get('health', 0) > 0}{spec_tag}")

            if state_vec is not None and self.server.callback is not None:
                self.server.callback(state_vec)
                self.server.last_update = time.time()
                self.server.update_count += 1

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')

        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad JSON')
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'Error: {e}'.encode())

    def do_GET(self):
        """Status endpoint for diagnostics."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        status = {
            'running': True,
            'update_count': self.server.update_count,
            'last_update': self.server.last_update,
            'seconds_since_update': (
                time.time() - self.server.last_update
                if self.server.last_update > 0 else -1
            ),
        }
        self.wfile.write(json.dumps(status).encode())

    def log_message(self, format, *args):
        pass  # Suppress default request logging


class GSIHTTPServer(HTTPServer):
    """HTTPServer with extra attributes for GSI."""

    def __init__(self, server_address, RequestHandlerClass, auth_token: str,
                 callback: Optional[Callable]):
        super().__init__(server_address, RequestHandlerClass)
        self.auth_token = auth_token
        self.callback = callback
        self.last_update: float = 0.0
        self.update_count: int = 0
        self._dump_count: int = 0
        self._request_count: int = 0
        self._last_money = '?'
        self._last_equip = '?'


class GSIServer:
    """
    CS2 Game State Integration server.

    Listens for POST requests from CS2 and calls the callback with
    the parsed state vector (100,).

    Usage:
        gsi = GSIServer(host='127.0.0.1', port=3000, auth_token='...', callback=engine.update_game_state)
        gsi.start()
        ...
        gsi.stop()
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 3000,
        auth_token: str = 'cs2nn_secret_token',
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.callback = callback

        self._server: Optional[GSIHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start GSI HTTP server in background thread."""
        try:
            self._server = GSIHTTPServer(
                (self.host, self.port),
                GSIRequestHandler,
                auth_token=self.auth_token,
                callback=self.callback,
            )
            self._running = True
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            print(f"[GSI] Server started on http://{self.host}:{self.port}")
            print(f"[GSI] Waiting for CS2 game state updates...")
            print(f"[GSI] Make sure gamestate_integration_cs2nn.cfg is in your CS2 cfg folder!")
        except OSError as e:
            print(f"[GSI] Failed to start server: {e}")
            print(f"[GSI] Port {self.port} may be in use. Try a different port.")
            self._running = False

    def _serve(self):
        while self._running:
            try:
                self._server.handle_request()
            except Exception:
                if self._running:
                    import traceback
                    traceback.print_exc()

    def stop(self):
        self._running = False
        if self._server:
            self._server.server_close()
        print("[GSI] Server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_update(self) -> float:
        return self._server.last_update if self._server else 0.0

    @property
    def update_count(self) -> int:
        return self._server.update_count if self._server else 0

    def seconds_since_update(self) -> float:
        if self._server and self._server.last_update > 0:
            return time.time() - self._server.last_update
        return float('inf')


def create_gsi_cfg(output_path: str, host: str = '127.0.0.1', port: int = 3000,
                   auth_token: str = 'cs2nn_secret_token'):
    """
    Write gamestate_integration_cs2nn.cfg for CS2.

    Args:
        output_path: Full path to write the .cfg file
    """
    cfg_content = f'''"CS2 NN Agent"
{{
    "uri"               "http://{host}:{port}"
    "timeout"           "5.0"
    "buffer"            "0.1"
    "throttle"          "0.0"
    "heartbeat"         "10.0"
    "auth"
    {{
        "token"         "{auth_token}"
    }}
    "data"
    {{
        "provider"              "1"
        "map"                   "1"
        "map_round_wins"        "1"
        "player_id"             "1"
        "player_state"          "1"
        "player_weapons"        "1"
        "player_match_stats"    "1"
        "round"                 "1"
        "allplayers_id"         "1"
        "allplayers_state"      "1"
        "allplayers_weapons"    "0"
        "allplayers_match_stats" "0"
        "phase_countdowns"      "1"
    }}
}}
'''
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cfg_content)
    print(f"[GSI] Config written to: {output_path}")


if __name__ == '__main__':
    import sys

    if '--write-cfg' in sys.argv:
        default_path = (
            r"C:\Program Files (x86)\Steam\steamapps\common"
            r"\Counter-Strike Global Offensive\game\csgo\cfg"
            r"\gamestate_integration_cs2nn.cfg"
        )
        create_gsi_cfg(default_path)
        print("Done. Restart CS2 for the config to take effect.")
        sys.exit(0)

    # Test server with logging
    from datetime import datetime

    log_path = f"gsi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_file = open(log_path, 'w', encoding='utf-8')
    print(f"[GSI] Logging to: {log_path}")

    def test_callback(state_vec: np.ndarray):
        hp = state_vec[0] * 100
        armor = state_vec[1] * 100
        helmet = bool(state_vec[2] > 0.5)
        ammo = state_vec[3] * 100
        ct_alive = int(state_vec[4] * 5)
        t_alive = int(state_vec[5] * 5)
        round_time = state_vec[6] * 115
        bomb = bool(state_vec[7] > 0.5)
        freeze = bool(state_vec[8] > 0.5)
        defuser = bool(state_vec[9] > 0.5)
        score_ct = int(state_vec[10] * 16)
        score_t = int(state_vec[11] * 16)
        side = 'CT' if state_vec[12] > 0.5 else 'T'

        # Current weapon (one-hot [14:57])
        weapon_idx = np.argmax(state_vec[14:57])
        weapon = WEAPONS[weapon_idx] if state_vec[14 + weapon_idx] > 0.5 else 'None'

        # Inventory (multi-hot [57:100])
        inventory = [WEAPONS[i] for i in range(43) if state_vec[57 + i] > 0.5]

        # Console output
        print(f"\n{'='*50}")
        player_alive = hp > 0
        print(f"[GSI] Side={side} | HP={hp:.0f} Armor={armor:.0f} Helmet={helmet} | Alive={player_alive}")
        print(f"      Money=${gsi._server._last_money} EquipValue=${gsi._server._last_equip}")
        print(f"      Ammo={ammo:.0f}% | Defuser={defuser}")
        print(f"      CT alive={ct_alive} T alive={t_alive}  (from screen CNN, 0=not set)")
        print(f"      Score: CT {score_ct} - {score_t} T")
        print(f"      Bomb={bomb} Freeze={freeze} RoundTime={round_time:.0f}s")
        print(f"      Weapon: {weapon}")
        print(f"      Inventory: {inventory}")
        print(f"{'='*50}")

        # Log to file
        entry = json.dumps({
            "t": datetime.now().isoformat(timespec='milliseconds'),
            "hp": hp, "armor": armor, "helmet": helmet,
            "player_alive": player_alive,
            "money": gsi._server._last_money,
            "equip_value": gsi._server._last_equip,
            "ammo_pct": round(ammo, 1), "side": side,
            "ct_alive": ct_alive, "t_alive": t_alive,
            "score_ct": score_ct, "score_t": score_t,
            "bomb": bomb, "freeze": freeze,
            "round_time": round(round_time, 1),
            "defuser": defuser, "weapon": weapon,
            "inventory": inventory,
            "raw": state_vec.tolist(),
        })
        log_file.write(entry + '\n')
        log_file.flush()

    gsi = GSIServer(callback=test_callback)
    gsi.start()

    print("GSI server running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            if gsi.update_count > 0:
                print(f"  Updates: {gsi.update_count}, last: {gsi.seconds_since_update():.1f}s ago")
    except KeyboardInterrupt:
        log_file.close()
        print(f"\n[GSI] Log saved: {log_path}")
        gsi.stop()
