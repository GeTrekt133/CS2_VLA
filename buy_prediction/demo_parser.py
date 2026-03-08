"""
Demo parser for buy prediction.
Extracts per-player per-round data using only features available via GSI at inference.

GSI-available features:
  - money (own)
  - round_num
  - team_score, enemy_score
  - equipment_value
  - round_wins pattern -> loss_streak
  - side (T/CT)

Ground truth: items_bought per player per round.
"""
from demoparser2 import DemoParser
import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path


def parse_buy_data(demo_path: str) -> pd.DataFrame:
    """
    Parse demo file, extract per-player per-round buy data.

    Returns DataFrame with columns:
        round_num, steamid, team_num, money, equipment_value,
        team_score, enemy_score, loss_streak, items_bought
    """
    parser = DemoParser(demo_path)

    # Round events
    try:
        round_freeze_end = parser.parse_event('round_freeze_end')
        round_end = parser.parse_event('round_end')
    except Exception as e:
        print(f"Error parsing round events: {e}")
        return pd.DataFrame()

    if len(round_freeze_end) == 0:
        return pd.DataFrame()

    round_freeze_end = round_freeze_end.sort_values('tick').reset_index(drop=True)
    round_end = round_end.sort_values('tick').reset_index(drop=True)

    # Player ticks
    tick_cols = ['tick', 'steamid', 'team_num', 'name']
    money_cols = ['balance', 'current_equip_value']
    try:
        ticks_df = parser.parse_ticks(tick_cols + money_cols)
    except Exception:
        try:
            ticks_df = parser.parse_ticks(tick_cols)
            ticks_df['balance'] = 0
            ticks_df['current_equip_value'] = 0
        except Exception as e:
            print(f"Error parsing ticks: {e}")
            return pd.DataFrame()

    # Item purchases
    try:
        item_purchase = parser.parse_event('item_purchase')
        rename = {}
        if 'user_steamid' in item_purchase.columns:
            rename['user_steamid'] = 'steamid'
        for col in ['item_name', 'item']:
            if col in item_purchase.columns:
                rename[col] = 'weapon'
                break
        if rename:
            item_purchase = item_purchase.rename(columns=rename)
    except Exception:
        item_purchase = pd.DataFrame(columns=['tick', 'steamid', 'weapon'])

    # Track scores and loss streaks per team
    t_score, ct_score = 0, 0
    # loss_streak[team_num] = consecutive losses
    loss_streak = {2: 0, 3: 0}

    rows = []

    for round_idx in range(len(round_freeze_end)):
        try:
            freeze_tick = int(round_freeze_end.iloc[round_idx]['tick'])

            # Find round_start tick (approx 2000 ticks before freeze_end)
            round_start_tick = max(0, freeze_tick - 2000)

            # Player states at freeze end (after buy)
            tick_diff = (ticks_df['tick'] - freeze_tick).abs()
            closest = ticks_df.loc[tick_diff.idxmin(), 'tick']
            players = ticks_df[ticks_df['tick'] == closest].copy()

            if len(players) == 0:
                continue

            # Items bought during freeze
            buys = item_purchase[
                (item_purchase['tick'] >= round_start_tick) &
                (item_purchase['tick'] <= freeze_tick)
            ]

            for _, p in players.iterrows():
                steamid = p['steamid']
                team_num = p['team_num']
                if pd.isna(team_num) or team_num not in [2, 3]:
                    continue
                team_num = int(team_num)

                money = int(p.get('balance', 0) or 0)
                equip = int(p.get('current_equip_value', 0) or 0)

                # Score from player's perspective
                if team_num == 2:
                    ts, es = t_score, ct_score
                else:
                    ts, es = ct_score, t_score

                # Items bought by this player
                player_buys = []
                if len(buys) > 0 and 'steamid' in buys.columns:
                    pb = buys[buys['steamid'] == steamid]
                    if 'weapon' in pb.columns and len(pb) > 0:
                        player_buys = pb['weapon'].tolist()

                rows.append({
                    'round_num': round_idx + 1,
                    'steamid': steamid,
                    'team_num': team_num,
                    'money': money,
                    'equipment_value': equip,
                    'team_score': ts,
                    'enemy_score': es,
                    'loss_streak': loss_streak.get(team_num, 0),
                    'items_bought': player_buys,
                })

            # Update scores and loss streaks from round_end
            if round_idx < len(round_end):
                winner = round_end.iloc[round_idx].get('winner', 0)
                if winner == 2:
                    t_score += 1
                    loss_streak[2] = 0
                    loss_streak[3] = loss_streak.get(3, 0) + 1
                elif winner == 3:
                    ct_score += 1
                    loss_streak[3] = 0
                    loss_streak[2] = loss_streak.get(2, 0) + 1

        except Exception as e:
            print(f"Error round {round_idx}: {e}")
            continue

    return pd.DataFrame(rows)


def parse_multiple_demos(demo_paths: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    """Parse multiple demos and concatenate."""
    all_data = []
    for i, path in enumerate(demo_paths):
        print(f"[{i+1}/{len(demo_paths)}] {Path(path).name}")
        try:
            df = parse_buy_data(path)
            if len(df) > 0:
                df['demo_id'] = i
                all_data.append(df)
                print(f"  -> {len(df)} samples")
        except Exception as e:
            print(f"  -> Error: {e}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"\nSaved {len(result)} samples to {output_path}")
    return result
