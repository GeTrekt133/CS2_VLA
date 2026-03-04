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
        "allplayers_id"     "0"
        "allplayers_state"  "0"
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
    # Knife
    'weapon_knife': 'knife',
    'weapon_knifegg': 'knife',
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

    # 0: hp (normalized 0-1)
    state_vec[0] = player_state.get('health', 0) / 100.0
    # 1: armor (normalized 0-1)
    state_vec[1] = player_state.get('armor', 0) / 100.0
    # 2: helmet
    state_vec[2] = float(bool(player_state.get('helmet', False)))
    # 3: ammo (current clip, normalized 0-1; approximate max 100)
    #    We'll fill this from the active weapon below
    # 4: ct_alive count (from allplayers or map)
    map_data = data.get('map', {})
    ct_data = map_data.get('team_ct', {})
    t_data = map_data.get('team_t', {})
    # count alive players from allplayers if available
    allplayers = data.get('allplayers', {})
    if allplayers:
        ct_alive = sum(1 for p in allplayers.values()
                       if p.get('team') == 'CT' and p.get('state', {}).get('health', 0) > 0)
        t_alive = sum(1 for p in allplayers.values()
                      if p.get('team') == 'T' and p.get('state', {}).get('health', 0) > 0)
    else:
        ct_alive = ct_data.get('alive_count', ct_data.get('players_alive', 5))
        t_alive = t_data.get('alive_count', t_data.get('players_alive', 5))
    state_vec[4] = ct_alive / 5.0
    state_vec[5] = t_alive / 5.0
    # 6: round_time_left (normalized, assume max ~115 sec)
    round_data = data.get('round', {})
    phase = round_data.get('phase', '')
    # CS2 GSI doesn't directly give remaining time; approximate from phase
    state_vec[6] = 0.5  # placeholder — CS2 doesn't expose countdown via GSI
    # 7: bomb_planted
    state_vec[7] = float(round_data.get('bomb', '') == 'planted')
    # 8: freeze_time
    state_vec[8] = float(phase == 'freezetime')
    # 9: defuser
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

    # === Parse weapons ===
    weapons_data = player.get('weapons', {})

    active_weapon_name = None
    weapon_inventory = set()
    active_ammo = 0

    for slot_key, weapon_info in weapons_data.items():
        weapon_name_raw = weapon_info.get('name', '').lower()
        canonical = CS2_WEAPON_ALIASES.get(weapon_name_raw, None)
        if canonical is None:
            canonical = 'None'

        if weapon_info.get('state', '') == 'active':
            active_weapon_name = canonical
            active_ammo = weapon_info.get('ammo_clip', 0)
            ammo_max = weapon_info.get('ammo_clip_max', 1) or 1
            state_vec[3] = active_ammo / max(ammo_max, 1)

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
            state_vec = parse_gsi_json(data, self.server.auth_token)

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

    # Test server
    def test_callback(state_vec: np.ndarray):
        hp = state_vec[0] * 100
        armor = state_vec[1] * 100
        side = 'CT' if state_vec[12] > 0.5 else 'T'
        print(f"[GSI] HP={hp:.0f} Armor={armor:.0f} Side={side}")

    gsi = GSIServer(callback=test_callback)
    gsi.start()

    print("GSI server running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            if gsi.update_count > 0:
                print(f"  Updates: {gsi.update_count}, last: {gsi.seconds_since_update():.1f}s ago")
    except KeyboardInterrupt:
        gsi.stop()
