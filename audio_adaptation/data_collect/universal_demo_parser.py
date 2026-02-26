"""
Universal Demo Parser for CS2.

Parses demo files (.dem) into structured JSON format suitable for:
- Behavioral Cloning (BC): states, actions, mouse movements
- Offline RL: events (kills, deaths, damage), rewards, episode boundaries

Output JSON structure:
{
  "demos": [
    {
      "demo_name": "match_001",
      "demo_path": "path/to/demo.dem",
      "player_steam_id": 76561198012345678,
      "rounds": [
        {
          "round_id": 0,
          "start_tick": 1000,
          "end_tick": 8500,
          "winner": "CT",  # or "T"
          "player_team": "T",
          "score": [3, 2],  # [my_score, enemy_score]

          # Events for offline RL
          "events": [
            {"type": "kill", "tick": 1500, "victim_id": 123, "weapon": "ak47", "headshot": true},
            {"type": "death", "tick": 2000, "killer_id": 456, "weapon": "awp"},
            {"type": "damage_dealt", "tick": 1450, "victim_id": 123, "damage": 27, "hitgroup": "chest"},
            {"type": "damage_taken", "tick": 1980, "attacker_id": 456, "damage": 100, "hitgroup": "head"},
            {"type": "bomb_plant", "tick": 5000},
            {"type": "bomb_defuse", "tick": 7000},
            {"type": "round_end", "tick": 8500, "winner": "CT", "reason": "elimination"}
          ],

          # States for each tick
          "states": [
            {
              "tick": 1000,
              "keys": ["W", "MOUSE_LEFT"],
              "mouse": [45.2, -10.5],  # [yaw, pitch]
              "hp": 100,
              "armor": 100,
              "helmet": true,
              "defuser": false,
              "weapon": "ak-47",
              "ammo": 30,
              "weapon_list": ["ak-47", "glock-18", "knife"],
              "side": "T",
              "ct_alive": 5,
              "t_alive": 5,
              "round_time_left": 115,
              "bomb_planted": false,
              "freeze_time": false,
              "score": [3, 2]
            }
          ]
        }
      ]
    }
  ]
}
"""

from demoparser2 import DemoParser
import pandas as pd
import json
import math
import random as r
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import os


# Constants
TICKRATE = 64
ROUND_TIME = 115
BOMB_TIME = 40

# Key mapping from button bitflags
KEY_MAPPING = {
    "MOUSE_LEFT": 1 << 0,
    "SPACE": 1 << 1,
    "CTRL": 1 << 2,
    "W": 1 << 3,
    "S": 1 << 4,
    "E": 1 << 5,
    "A": 1 << 9,
    "D": 1 << 10,
    "MOUSE_RIGHT": 1 << 11,
    "R": 1 << 13,
    "TAB": 1 << 33,
    "F": 1 << 35,
}

# Weapon categories
PISTOLS = [
    'p2000', 'p250', 'five-seven', 'glock-18', 'tec-9',
    'cz75-auto', 'dual berettas', 'desert eagle', 'r8 revolver'
]
RIFLES = [
    'mp9', 'mac-10', 'pp-bizon', 'mp7', 'ump-45', 'p90', 'mp5-sd',
    'famas', 'galil ar', 'm4a4', 'm4a1-s', 'ak-47', 'aug', 'sg 553',
    'ssg 08', 'awp', 'scar-20', 'g3sg1', 'nova', 'xm1014', 'mag-7',
    'sawed-off', 'm249', 'negev'
]


def extract_buttons(buttons: int, is_walking: bool = False) -> List[str]:
    """Extract pressed keys from button bitflags."""
    all_keys = []

    if pd.isna(buttons):
        return all_keys

    # Extract standard keys
    for button_name, bit_slot in KEY_MAPPING.items():
        if (int(buttons) & bit_slot) != 0:
            all_keys.append(button_name)

    # Add SHIFT if walking
    if is_walking:
        all_keys.append('SHIFT')

    return all_keys


def infer_weapon_key(weapon_name: str, cnt: int) -> Optional[str]:
    """Infer weapon slot key (WEAPON1/2/3) from active weapon."""
    if not weapon_name or cnt == 0:
        return None

    aw = weapon_name.lower()

    # Knife
    if 'knife' in aw or 'bayonet' in aw or 'karambit' in aw or 'daggers' in aw:
        return 'WEAPON3'
    # Pistol
    elif aw in PISTOLS:
        return 'WEAPON2'
    # Rifle
    elif aw in RIFLES:
        return 'WEAPON1'
    # Grenades
    elif 'high explosive grenade' in aw:
        return 'HE'
    elif 'flashbang' in aw:
        return 'FLASH'
    elif 'smoke grenade' in aw:
        return 'SMOKE'
    elif 'decoy grenade' in aw:
        return 'DECOY'
    elif 'molotov' in aw or 'incendiary grenade' in aw:
        return 'MOLOTOV'
    elif 'c4' in aw:
        return 'C4'

    return None


class UniversalDemoParser:
    """
    Universal parser for CS2 demos.

    Extracts both BC and RL data from demo files.
    """

    def __init__(self, target_steam_id: int):
        """
        Args:
            target_steam_id: Steam ID of the player to track
        """
        self.target_steam_id = target_steam_id

    def parse_demo(
        self,
        demo_path: str,
        skip_first_rounds: int = 3
    ) -> Dict[str, Any]:
        """
        Parse a single demo file.

        Args:
            demo_path: Path to .dem file
            skip_first_rounds: Skip first N rounds (warmup/pistol)

        Returns:
            Dictionary with parsed data
        """
        print(f"\nParsing demo: {demo_path}")

        parser = DemoParser(demo_path)
        demo_name = os.path.basename(demo_path)[:-4]

        # Parse ticks
        tick_events = [
            'tick', 'inventory', 'has_helmet', 'has_defuser', 'armor_value',
            'balance', 'is_alive', 'team_num', 'buttons', 'active_weapon_name',
            'yaw', 'pitch', 'active_weapon_ammo', 'health', 'steamid', 'user_id',
            'is_walking', 'round_win_status'
        ]
        ticks_df = parser.parse_ticks(tick_events)
        ticks_df_target = ticks_df[ticks_df['steamid'] == self.target_steam_id]

        # Parse events for offline RL
        # NOTE: demoparser2 v0.39+ returns columns directly (attacker_steamid,
        # user_steamid, weapon, headshot, etc.) — do NOT use player=/other= params
        kills_df = parser.parse_event('player_death')
        # player_death columns: attacker_steamid, user_steamid (victim), weapon, headshot, ...
        damages_df = parser.parse_event('player_hurt')
        # player_hurt columns: attacker_steamid, user_steamid (victim), weapon, dmg_health, hitgroup, ...

        # Additional events for dense rewards
        weapon_fire_df = parser.parse_event('weapon_fire')
        # weapon_fire columns: user_steamid, weapon, ...
        player_blind_df = parser.parse_event('player_blind')
        # player_blind columns: user_steamid (blinded), attacker_steamid (flasher), blind_duration, ...

        # Grenade events (some may not exist in every demo — return empty DF)
        def _safe_parse_event(event_name):
            try:
                result = parser.parse_event(event_name)
                if isinstance(result, pd.DataFrame):
                    return result
                return pd.DataFrame()
            except Exception:
                return pd.DataFrame()

        grenade_thrown_df = _safe_parse_event('grenade_thrown')
        flashbang_detonate_df = _safe_parse_event('flashbang_detonate')
        hegrenade_detonate_df = _safe_parse_event('hegrenade_detonate')
        smokegrenade_detonate_df = _safe_parse_event('smokegrenade_detonate')
        molotov_detonate_df = _safe_parse_event('inferno_startburn')  # molotov_detonate doesn't exist

        # Bomb defuse attempts
        bomb_begindefuse_df = _safe_parse_event('bomb_begindefuse')
        bomb_abortdefuse_df = _safe_parse_event('bomb_abortdefuse')

        # Item purchases
        item_purchase_df = _safe_parse_event('item_purchase')
        # item_purchase columns: steamid (not user_steamid!), item_name, ...

        # Weapon reloads
        weapon_reload_df = _safe_parse_event('weapon_reload')
        # weapon_reload columns: user_steamid (no weapon column!)

        # Parse round events
        bomb_planted_ticks = list(parser.parse_event('bomb_planted')['tick'])
        bomb_defused_ticks = list(parser.parse_event('bomb_defused')['tick'])
        round_prestart = list(parser.parse_event('round_prestart')['tick'])[skip_first_rounds:]
        round_freeze_end = list(parser.parse_event('round_freeze_end')['tick'])[skip_first_rounds-1:]

        max_tick = ticks_df['tick'].max()

        # Compute time_left for each tick
        time_left_df = self._compute_time_left(
            round_prestart, round_freeze_end, bomb_planted_ticks, max_tick
        )

        # Extract buttons with weapon inference
        ticks_df_target = self._extract_buttons_with_weapons(ticks_df_target)

        # Build rounds
        rounds = self._build_rounds(
            ticks_df, ticks_df_target, time_left_df,
            kills_df, damages_df,
            weapon_fire_df, player_blind_df,
            grenade_thrown_df, flashbang_detonate_df, hegrenade_detonate_df,
            smokegrenade_detonate_df, molotov_detonate_df,
            bomb_begindefuse_df, bomb_abortdefuse_df,
            item_purchase_df, weapon_reload_df,
            round_prestart, round_freeze_end,
            bomb_planted_ticks, bomb_defused_ticks,
            max_tick,
            skip_first_rounds
        )

        return {
            'demo_name': demo_name,
            'demo_path': demo_path,
            'player_steam_id': self.target_steam_id,
            'rounds': rounds
        }

    def _compute_time_left(
        self,
        round_prestart: List[int],
        round_freeze_end: List[int],
        bomb_planted_ticks: List[int],
        max_tick: int
    ) -> pd.DataFrame:
        """Compute time_left, freeze, and bomb status for each tick."""
        rows = []

        for i in range(len(round_freeze_end)):
            freeze_start_tick = int(round_prestart[i])
            freeze_end_tick = int(round_freeze_end[i])
            freeze_duration = (freeze_end_tick - freeze_start_tick) / TICKRATE

            round_end_tick = round_prestart[i + 1] - 1 if i + 1 < len(round_prestart) else max_tick
            bomb_tick = next((bt for bt in bomb_planted_ticks if freeze_end_tick < bt < round_end_tick), None)

            # Freeze time
            for tick in range(freeze_start_tick, freeze_end_tick):
                sec_elapsed = (tick - freeze_start_tick) / TICKRATE
                time_left = max(freeze_duration - sec_elapsed, 0)
                rows.append([tick, math.ceil(time_left), True, False])

            # Normal round or bomb time
            for tick in range(freeze_end_tick, round_end_tick):
                if bomb_tick and tick >= bomb_tick:
                    sec_elapsed = (tick - bomb_tick) / TICKRATE
                    time_left = max(BOMB_TIME - sec_elapsed, 0)
                    rows.append([tick, math.ceil(time_left), False, True])
                    if time_left == 0:
                        break
                else:
                    sec_elapsed = (tick - freeze_end_tick) / TICKRATE
                    time_left = max(ROUND_TIME - sec_elapsed, 0)
                    rows.append([tick, math.ceil(time_left), False, False])
                    if time_left == 0:
                        break

        return pd.DataFrame(rows, columns=['tick', 'time_left', 'freeze', 'bomb'])

    def _extract_buttons_with_weapons(self, ticks_df_target: pd.DataFrame) -> pd.DataFrame:
        """Extract buttons and infer weapon keys."""
        buttons_list = []
        prev_weapon = ''
        weapon_switch_cooldown = 0

        for i, row in tqdm(ticks_df_target.iterrows(), total=len(ticks_df_target), desc="Extracting buttons"):
            # Extract standard keys
            keys = extract_buttons(row['buttons'], row['is_walking'])

            # Infer weapon key
            current_weapon = row['active_weapon_name'].lower() if row['active_weapon_name'] else ''

            if current_weapon != prev_weapon:
                prev_weapon = current_weapon
                weapon_switch_cooldown = r.randint(3, 7)

            if weapon_switch_cooldown > 0:
                weapon_key = infer_weapon_key(current_weapon, weapon_switch_cooldown)
                if weapon_key:
                    keys.append(weapon_key)
                weapon_switch_cooldown -= 1

            buttons_list.append(keys)

        ticks_df_target = ticks_df_target.copy()
        ticks_df_target['buttons_parsed'] = buttons_list
        return ticks_df_target

    def _build_rounds(
        self,
        ticks_df: pd.DataFrame,
        ticks_df_target: pd.DataFrame,
        time_left_df: pd.DataFrame,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        weapon_fire_df: pd.DataFrame,
        player_blind_df: pd.DataFrame,
        grenade_thrown_df: pd.DataFrame,
        flashbang_detonate_df: pd.DataFrame,
        hegrenade_detonate_df: pd.DataFrame,
        smokegrenade_detonate_df: pd.DataFrame,
        molotov_detonate_df: pd.DataFrame,
        bomb_begindefuse_df: pd.DataFrame,
        bomb_abortdefuse_df: pd.DataFrame,
        item_purchase_df: pd.DataFrame,
        weapon_reload_df: pd.DataFrame,
        round_prestart: List[int],
        round_freeze_end: List[int],
        bomb_planted_ticks: List[int],
        bomb_defused_ticks: List[int],
        max_tick: int,
        skip_first_rounds: int = 3
    ) -> List[Dict[str, Any]]:
        """Build rounds with states and events."""
        rounds = []

        # Get round win status changes
        round_win_status_df = ticks_df_target[['tick', 'round_win_status']].loc[
            ticks_df_target['round_win_status'].shift() != ticks_df_target['round_win_status']
        ]
        round_win_status_df = round_win_status_df[round_win_status_df['round_win_status'] != 0].reset_index(drop=True)

        # Track team
        first_tick = round_prestart[0]
        my_initial_team_num = ticks_df_target[ticks_df_target['tick'] == first_tick]['team_num'].iloc[0]
        my_initial_team = 'T' if my_initial_team_num == 2 else 'CT'

        score = [0, 0]  # [my_score, enemy_score]

        for round_idx, start_tick in enumerate(tqdm(round_prestart, desc="Building rounds")):
            # Actual round number in the game (accounting for skipped rounds)
            actual_round = round_idx + skip_first_rounds

            # Determine round boundaries
            end_tick = round_prestart[round_idx + 1] - 1 if round_idx + 1 < len(round_prestart) else max_tick

            # Get player's team for this round
            team_row = ticks_df_target[ticks_df_target['tick'] == start_tick]
            if team_row.empty:
                continue

            team_num = team_row['team_num'].iloc[0]
            team = 'T' if team_num == 2 else 'CT'

            # Update score using actual round number for correct indexing
            if round_idx > 0:
                prev_actual_round = actual_round - 1
                if prev_actual_round < len(round_win_status_df):
                    win_side = int(round_win_status_df['round_win_status'].iloc[prev_actual_round])

                    # Check for side swap using actual round number
                    swap_sides = ((actual_round - 1) // 12) % 2 == 1 or (actual_round > 25 and (actual_round - 1) % 3 == 0)
                    current_my_side = (
                        'CT' if (my_initial_team == 'T' and swap_sides)
                        else 'T' if (my_initial_team == 'CT' and swap_sides)
                        else my_initial_team
                    )

                    if current_my_side == 'T':
                        if win_side == 2:
                            score[0] += 1
                        elif win_side == 3:
                            score[1] += 1
                    else:
                        if win_side == 3:
                            score[0] += 1
                        elif win_side == 2:
                            score[1] += 1

            # Determine winner using actual round number
            if actual_round < len(round_win_status_df):
                win_side = int(round_win_status_df['round_win_status'].iloc[actual_round])
                winner = 'T' if win_side == 2 else 'CT'
            else:
                winner = None

            # Extract events for this round
            events = self._extract_round_events(
                round_idx, start_tick, end_tick,
                kills_df, damages_df,
                weapon_fire_df, player_blind_df,
                grenade_thrown_df, flashbang_detonate_df, hegrenade_detonate_df,
                smokegrenade_detonate_df, molotov_detonate_df,
                bomb_begindefuse_df, bomb_abortdefuse_df,
                item_purchase_df, weapon_reload_df,
                bomb_planted_ticks, bomb_defused_ticks,
                winner
            )

            # Extract states
            states = self._extract_round_states(
                start_tick, end_tick,
                ticks_df, ticks_df_target, time_left_df,
                team, score
            )

            rounds.append({
                'round_id': round_idx,
                'start_tick': start_tick,
                'end_tick': end_tick,
                'winner': winner,
                'player_team': team,
                'score': score.copy(),
                'events': events,
                'states': states
            })

        return rounds

    def _extract_round_events(
        self,
        round_idx: int,
        start_tick: int,
        end_tick: int,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        weapon_fire_df: pd.DataFrame,
        player_blind_df: pd.DataFrame,
        grenade_thrown_df: pd.DataFrame,
        flashbang_detonate_df: pd.DataFrame,
        hegrenade_detonate_df: pd.DataFrame,
        smokegrenade_detonate_df: pd.DataFrame,
        molotov_detonate_df: pd.DataFrame,
        bomb_begindefuse_df: pd.DataFrame,
        bomb_abortdefuse_df: pd.DataFrame,
        item_purchase_df: pd.DataFrame,
        weapon_reload_df: pd.DataFrame,
        bomb_planted_ticks: List[int],
        bomb_defused_ticks: List[int],
        winner: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Extract events (kills, deaths, damage, grenades, etc.) for a round."""
        events = []

        # Kills/Deaths
        # player_death: user_steamid=victim, attacker_steamid=killer
        round_kills = kills_df[(kills_df['tick'] >= start_tick) & (kills_df['tick'] <= end_tick)]
        for _, kill in round_kills.iterrows():
            if kill['attacker_steamid'] == self.target_steam_id:
                # Player killed someone
                events.append({
                    'type': 'kill',
                    'tick': int(kill['tick']),
                    'victim_id': int(kill['user_steamid']),
                    'weapon': str(kill['weapon']),
                    'headshot': bool(kill['headshot'])
                })
            elif kill['user_steamid'] == self.target_steam_id:
                # Player died
                events.append({
                    'type': 'death',
                    'tick': int(kill['tick']),
                    'killer_id': int(kill['attacker_steamid']),
                    'weapon': str(kill['weapon'])
                })

        # Damage dealt/taken
        # player_hurt: user_steamid=victim, attacker_steamid=attacker
        round_damages = damages_df[(damages_df['tick'] >= start_tick) & (damages_df['tick'] <= end_tick)]
        for _, dmg in round_damages.iterrows():
            if dmg['attacker_steamid'] == self.target_steam_id:
                # Player dealt damage
                events.append({
                    'type': 'damage_dealt',
                    'tick': int(dmg['tick']),
                    'victim_id': int(dmg['user_steamid']),
                    'damage': int(dmg['dmg_health']),
                    'hitgroup': str(dmg['hitgroup']),
                    'weapon': str(dmg['weapon'])
                })
            elif dmg['user_steamid'] == self.target_steam_id:
                # Player took damage
                events.append({
                    'type': 'damage_taken',
                    'tick': int(dmg['tick']),
                    'attacker_id': int(dmg['attacker_steamid']),
                    'damage': int(dmg['dmg_health']),
                    'hitgroup': str(dmg['hitgroup']),
                    'weapon': str(dmg['weapon'])
                })

        # Weapon fire (for accuracy tracking)
        round_weapon_fire = weapon_fire_df[(weapon_fire_df['tick'] >= start_tick) & (weapon_fire_df['tick'] <= end_tick)]
        round_weapon_fire = round_weapon_fire[round_weapon_fire['user_steamid'] == self.target_steam_id]
        for _, fire in round_weapon_fire.iterrows():
            events.append({
                'type': 'weapon_fire',
                'tick': int(fire['tick']),
                'weapon': str(fire['weapon'])
            })

        # Player blinded (flashbang hit)
        # player_blind: user_steamid=blinded player, attacker_steamid=flasher
        round_blind = player_blind_df[(player_blind_df['tick'] >= start_tick) & (player_blind_df['tick'] <= end_tick)]
        if 'user_steamid' in player_blind_df.columns:
            round_blind = round_blind[round_blind['user_steamid'] == self.target_steam_id]
            for _, blind in round_blind.iterrows():
                events.append({
                    'type': 'player_blind',
                    'tick': int(blind['tick']),
                    'attacker_id': int(blind['attacker_steamid']),
                    'blind_duration': float(blind['blind_duration'])
                })

        # Grenade thrown (utility usage)
        # Note: grenade_thrown may not exist in all demos
        if not grenade_thrown_df.empty and 'user_steamid' in grenade_thrown_df.columns:
            round_grenades = grenade_thrown_df[(grenade_thrown_df['tick'] >= start_tick) & (grenade_thrown_df['tick'] <= end_tick)]
            round_grenades = round_grenades[round_grenades['user_steamid'] == self.target_steam_id]
            weapon_col = 'weapon' if 'weapon' in grenade_thrown_df.columns else None
            for _, nade in round_grenades.iterrows():
                events.append({
                    'type': 'grenade_thrown',
                    'tick': int(nade['tick']),
                    'grenade_type': str(nade[weapon_col]) if weapon_col else 'unknown'
                })

        # Flashbang detonate (successful flash = reward)
        if not flashbang_detonate_df.empty and 'user_steamid' in flashbang_detonate_df.columns:
            round_flash_det = flashbang_detonate_df[(flashbang_detonate_df['tick'] >= start_tick) & (flashbang_detonate_df['tick'] <= end_tick)]
            round_flash_det = round_flash_det[round_flash_det['user_steamid'] == self.target_steam_id]
            for _, flash in round_flash_det.iterrows():
                events.append({
                    'type': 'flashbang_detonate',
                    'tick': int(flash['tick'])
                })

        # HE grenade detonate
        if not hegrenade_detonate_df.empty and 'user_steamid' in hegrenade_detonate_df.columns:
            round_he_det = hegrenade_detonate_df[(hegrenade_detonate_df['tick'] >= start_tick) & (hegrenade_detonate_df['tick'] <= end_tick)]
            round_he_det = round_he_det[round_he_det['user_steamid'] == self.target_steam_id]
            for _, he in round_he_det.iterrows():
                events.append({
                    'type': 'hegrenade_detonate',
                    'tick': int(he['tick'])
                })

        # Smoke grenade detonate
        if not smokegrenade_detonate_df.empty and 'user_steamid' in smokegrenade_detonate_df.columns:
            round_smoke_det = smokegrenade_detonate_df[(smokegrenade_detonate_df['tick'] >= start_tick) & (smokegrenade_detonate_df['tick'] <= end_tick)]
            round_smoke_det = round_smoke_det[round_smoke_det['user_steamid'] == self.target_steam_id]
            for _, smoke in round_smoke_det.iterrows():
                events.append({
                    'type': 'smokegrenade_detonate',
                    'tick': int(smoke['tick'])
                })

        # Molotov/incendiary detonate (inferno_startburn in CS2)
        if not molotov_detonate_df.empty and 'user_steamid' in molotov_detonate_df.columns:
            round_molotov_det = molotov_detonate_df[(molotov_detonate_df['tick'] >= start_tick) & (molotov_detonate_df['tick'] <= end_tick)]
            round_molotov_det = round_molotov_det[round_molotov_det['user_steamid'] == self.target_steam_id]
            for _, molotov in round_molotov_det.iterrows():
                events.append({
                    'type': 'molotov_detonate',
                    'tick': int(molotov['tick'])
                })

        # Bomb defuse attempts
        if not bomb_begindefuse_df.empty and 'user_steamid' in bomb_begindefuse_df.columns:
            round_begindefuse = bomb_begindefuse_df[(bomb_begindefuse_df['tick'] >= start_tick) & (bomb_begindefuse_df['tick'] <= end_tick)]
            round_begindefuse = round_begindefuse[round_begindefuse['user_steamid'] == self.target_steam_id]
            for _, defuse in round_begindefuse.iterrows():
                events.append({
                    'type': 'bomb_begindefuse',
                    'tick': int(defuse['tick'])
                })

        if not bomb_abortdefuse_df.empty and 'user_steamid' in bomb_abortdefuse_df.columns:
            round_abortdefuse = bomb_abortdefuse_df[(bomb_abortdefuse_df['tick'] >= start_tick) & (bomb_abortdefuse_df['tick'] <= end_tick)]
            round_abortdefuse = round_abortdefuse[round_abortdefuse['user_steamid'] == self.target_steam_id]
            for _, defuse in round_abortdefuse.iterrows():
                events.append({
                    'type': 'bomb_abortdefuse',
                    'tick': int(defuse['tick'])
                })

        # Item purchases (economic management)
        # item_purchase uses 'steamid' (not 'user_steamid') and 'item_name' (not 'weapon')
        if not item_purchase_df.empty:
            steamid_col = 'steamid' if 'steamid' in item_purchase_df.columns else 'user_steamid'
            if steamid_col in item_purchase_df.columns:
                round_purchases = item_purchase_df[(item_purchase_df['tick'] >= start_tick) & (item_purchase_df['tick'] <= end_tick)]
                round_purchases = round_purchases[round_purchases[steamid_col] == self.target_steam_id]
                item_col = 'item_name' if 'item_name' in item_purchase_df.columns else 'weapon'
                for _, purchase in round_purchases.iterrows():
                    events.append({
                        'type': 'item_purchase',
                        'tick': int(purchase['tick']),
                        'item': str(purchase[item_col]) if item_col in purchase.index else 'unknown'
                    })

        # Weapon reloads (tactical awareness)
        # weapon_reload has user_steamid but NO weapon column
        if not weapon_reload_df.empty and 'user_steamid' in weapon_reload_df.columns:
            round_reloads = weapon_reload_df[(weapon_reload_df['tick'] >= start_tick) & (weapon_reload_df['tick'] <= end_tick)]
            round_reloads = round_reloads[round_reloads['user_steamid'] == self.target_steam_id]
            for _, reload in round_reloads.iterrows():
                events.append({
                    'type': 'weapon_reload',
                    'tick': int(reload['tick']),
                })

        # Bomb events
        for bt in bomb_planted_ticks:
            if start_tick <= bt <= end_tick:
                events.append({
                    'type': 'bomb_plant',
                    'tick': int(bt)
                })

        for bt in bomb_defused_ticks:
            if start_tick <= bt <= end_tick:
                events.append({
                    'type': 'bomb_defuse',
                    'tick': int(bt)
                })

        # Round end
        if winner:
            events.append({
                'type': 'round_end',
                'tick': end_tick,
                'winner': winner,
                'reason': 'unknown'  # TODO: parse round end reason
            })

        # Sort by tick
        events = sorted(events, key=lambda e: e['tick'])

        return events

    def _extract_round_states(
        self,
        start_tick: int,
        end_tick: int,
        ticks_df: pd.DataFrame,
        ticks_df_target: pd.DataFrame,
        time_left_df: pd.DataFrame,
        team: str,
        score: List[int]
    ) -> List[Dict[str, Any]]:
        """Extract states for each tick in a round."""
        states = []

        # Get ticks where player is alive
        alive_ticks = ticks_df_target[
            (ticks_df_target['tick'] >= start_tick) &
            (ticks_df_target['tick'] <= end_tick) &
            (ticks_df_target['is_alive'] == True)
        ]

        for _, tick_row in alive_ticks.iterrows():
            tick = int(tick_row['tick'])

            # Get time info
            time_row = time_left_df[time_left_df['tick'] == tick]
            if time_row.empty:
                time_left, bomb, freeze = None, None, None
            else:
                time_left = int(time_row['time_left'].iloc[0])
                bomb = bool(time_row['bomb'].iloc[0])
                freeze = bool(time_row['freeze'].iloc[0])

            # Get alive counts
            alive_players = ticks_df[(ticks_df['tick'] == tick) & (ticks_df['is_alive'] == True)]
            alive_t = len(alive_players[alive_players['team_num'] == 2])
            alive_ct = len(alive_players[alive_players['team_num'] == 3])

            states.append({
                'tick': tick,
                'keys': tick_row['buttons_parsed'],
                'mouse': [float(tick_row['yaw']), float(tick_row['pitch'])],
                'hp': int(tick_row['health']),
                'armor': int(tick_row['armor_value']),
                'helmet': bool(tick_row['has_helmet']),
                'defuser': bool(tick_row['has_defuser']),
                'weapon': str(tick_row['active_weapon_name']) if tick_row['active_weapon_name'] else '',
                'ammo': int(tick_row['active_weapon_ammo']) if not pd.isna(tick_row['active_weapon_ammo']) else 0,
                'weapon_list': tick_row['inventory'] if tick_row['inventory'] else [],
                'side': team,
                'ct_alive': alive_ct,
                't_alive': alive_t,
                'round_time_left': time_left,
                'bomb_planted': bomb,
                'freeze_time': freeze,
                'score': score.copy()
            })

        return states

    def parse_multiple_demos(
        self,
        demo_paths: List[str],
        output_json: str,
        skip_first_rounds: int = 3
    ):
        """
        Parse multiple demos and save to JSON.

        Args:
            demo_paths: List of paths to .dem files
            output_json: Output JSON path
            skip_first_rounds: Skip first N rounds per demo
        """
        result = {'demos': []}

        for demo_path in demo_paths:
            try:
                demo_data = self.parse_demo(demo_path, skip_first_rounds)
                result['demos'].append(demo_data)
            except Exception as e:
                print(f"Error parsing {demo_path}: {e}")

        # Save to JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(result['demos'])} demos to {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal CS2 Demo Parser")
    parser.add_argument('--demo-dir', type=str, required=True, help='Directory containing .dem files')
    parser.add_argument('--output-json', type=str, required=True, help='Output JSON path')
    parser.add_argument('--steam-id', type=int, required=True, help='Target player Steam ID')
    parser.add_argument('--skip-rounds', type=int, default=3, help='Skip first N rounds')

    args = parser.parse_args()

    # Find all demos
    demo_paths = [
        os.path.join(args.demo_dir, f)
        for f in os.listdir(args.demo_dir)
        if f.endswith('.dem')
    ]

    print(f"Found {len(demo_paths)} demos")

    # Parse
    parser = UniversalDemoParser(target_steam_id=args.steam_id)
    parser.parse_multiple_demos(demo_paths, args.output_json, args.skip_rounds)
