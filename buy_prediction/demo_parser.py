"""
Demo parser for extracting buy-related features from CS2 demo files.
Extracts: money, team money, round scores, equipment purchases.
"""
from demoparser2 import DemoParser
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def parse_buy_data(demo_path: str) -> pd.DataFrame:
    """
    Parse demo file and extract buy-related features per round.

    Returns DataFrame with columns:
    - round_num: round number
    - steamid: player steam id
    - team_num: 2=T, 3=CT
    - money_start: money at round start
    - money_spent: money spent on equipment
    - team_money_total: total team money at round start
    - team_money_avg: average team money
    - teammates_money: list of teammate money values
    - round_score_t: T side score
    - round_score_ct: CT side score
    - equipment_value: total equipment value after buy
    - items_bought: list of purchased items
    """
    parser = DemoParser(demo_path)

    # Parse round events
    try:
        round_start = parser.parse_event('round_start')
        round_freeze_end = parser.parse_event('round_freeze_end')
        round_end = parser.parse_event('round_end')
    except Exception as e:
        print(f"Error parsing round events: {e}")
        return pd.DataFrame()

    if len(round_freeze_end) == 0:
        print("No round_freeze_end events found")
        return pd.DataFrame()

    # Parse player data at ticks - check available columns
    base_cols = ['tick', 'steamid', 'team_num', 'name']
    money_cols = ['current_equip_value', 'cash_spent_this_round', 'balance']

    try:
        ticks_df = parser.parse_ticks(base_cols + money_cols)
    except Exception as e:
        print(f"Error with full columns, trying base only: {e}")
        try:
            ticks_df = parser.parse_ticks(base_cols)
            for col in money_cols:
                ticks_df[col] = 0
        except Exception as e2:
            print(f"Error parsing ticks: {e2}")
            return pd.DataFrame()

    # Parse item purchases
    try:
        item_purchase = parser.parse_event('item_purchase')
        # Rename columns to standard names
        rename_map = {}
        if 'user_steamid' in item_purchase.columns:
            rename_map['user_steamid'] = 'steamid'
        if 'item_name' in item_purchase.columns:
            rename_map['item_name'] = 'weapon'
        elif 'item' in item_purchase.columns:
            rename_map['item'] = 'weapon'
        if rename_map:
            item_purchase = item_purchase.rename(columns=rename_map)
    except Exception as e:
        print(f"Could not parse item_purchase: {e}")
        item_purchase = pd.DataFrame(columns=['tick', 'steamid', 'weapon'])

    # Get round boundaries
    rounds_data = []

    round_start = round_start.sort_values('tick').reset_index(drop=True)
    round_freeze_end = round_freeze_end.sort_values('tick').reset_index(drop=True)
    round_end = round_end.sort_values('tick').reset_index(drop=True)

    # Track scores
    t_score = 0
    ct_score = 0

    for round_idx in range(len(round_freeze_end)):
        try:
            freeze_end_tick = round_freeze_end.iloc[round_idx]['tick']

            # Find corresponding round_start (before freeze_end)
            round_starts_before = round_start[round_start['tick'] < freeze_end_tick]
            if len(round_starts_before) > 0:
                round_start_tick = round_starts_before.iloc[-1]['tick']
            else:
                round_start_tick = max(0, freeze_end_tick - 2000)

            # Get player states at freeze end (after buy phase)
            tick_diff = abs(ticks_df['tick'] - freeze_end_tick)
            closest_tick = ticks_df.loc[tick_diff.idxmin(), 'tick']
            players_at_freeze = ticks_df[ticks_df['tick'] == closest_tick].copy()

            if len(players_at_freeze) == 0:
                continue

            # Get player states at round start (before buy phase)
            tick_diff_start = abs(ticks_df['tick'] - round_start_tick)
            closest_tick_start = ticks_df.loc[tick_diff_start.idxmin(), 'tick']
            players_at_start = ticks_df[ticks_df['tick'] == closest_tick_start].copy()

            # Get items bought during freeze time
            buys_this_round = item_purchase[
                (item_purchase['tick'] >= round_start_tick) &
                (item_purchase['tick'] <= freeze_end_tick)
            ]

            # Update scores from previous round_end
            if round_idx > 0 and round_idx - 1 < len(round_end):
                prev_round_end = round_end.iloc[round_idx - 1]
                winner = prev_round_end.get('winner', 0)
                if winner == 2:
                    t_score += 1
                elif winner == 3:
                    ct_score += 1

            # Process each player
            for _, player in players_at_freeze.iterrows():
                steamid = player['steamid']
                team_num = player['team_num']

                if pd.isna(team_num) or team_num not in [2, 3]:
                    continue

                # Get money at round start
                player_start = players_at_start[players_at_start['steamid'] == steamid]
                if len(player_start) > 0 and 'balance' in player_start.columns:
                    money_start = player_start['balance'].values[0]
                else:
                    money_start = 800 if round_idx == 0 else 0

                # Get current balance and equipment
                balance = player.get('balance', 0) if 'balance' in player.index else 0
                equip_value = player.get('current_equip_value', 0) if 'current_equip_value' in player.index else 0
                cash_spent = player.get('cash_spent_this_round', 0) if 'cash_spent_this_round' in player.index else 0

                # Handle NaN values
                money_start = 0 if pd.isna(money_start) else int(money_start)
                balance = 0 if pd.isna(balance) else int(balance)
                equip_value = 0 if pd.isna(equip_value) else int(equip_value)
                cash_spent = 0 if pd.isna(cash_spent) else int(cash_spent)

                # Calculate team money
                teammates = players_at_start[
                    (players_at_start['team_num'] == team_num) &
                    (players_at_start['steamid'] != steamid)
                ]

                if 'balance' in teammates.columns and len(teammates) > 0:
                    teammates_money = [int(m) if not pd.isna(m) else 0 for m in teammates['balance'].tolist()]
                    team_money_total = sum(teammates_money) + money_start
                    team_money_avg = team_money_total / (len(teammates) + 1)
                else:
                    teammates_money = []
                    team_money_total = money_start
                    team_money_avg = money_start

                # Get items bought by this player
                if len(buys_this_round) > 0 and 'steamid' in buys_this_round.columns:
                    player_buys = buys_this_round[buys_this_round['steamid'] == steamid]
                    items_bought = player_buys['weapon'].tolist() if 'weapon' in player_buys.columns and len(player_buys) > 0 else []
                else:
                    items_bought = []

                rounds_data.append({
                    'round_num': round_idx + 1,
                    'steamid': steamid,
                    'team_num': int(team_num),
                    'money_start': money_start,
                    'money_spent': cash_spent,
                    'balance_after_buy': balance,
                    'team_money_total': team_money_total,
                    'team_money_avg': team_money_avg,
                    'teammates_money': teammates_money,
                    'round_score_t': t_score,
                    'round_score_ct': ct_score,
                    'equipment_value': equip_value,
                    'items_bought': items_bought,
                    'freeze_end_tick': freeze_end_tick
                })

        except Exception as e:
            print(f"Error processing round {round_idx}: {e}")
            continue

    return pd.DataFrame(rounds_data)


def parse_multiple_demos(demo_paths: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    """Parse multiple demo files and concatenate results."""
    all_data = []

    for i, demo_path in enumerate(demo_paths):
        print(f"Processing demo {i+1}/{len(demo_paths)}: {Path(demo_path).name}")
        try:
            df = parse_buy_data(demo_path)
            if len(df) > 0:
                df['demo_path'] = demo_path
                df['demo_id'] = i
                all_data.append(df)
                print(f"  -> {len(df)} samples")
            else:
                print(f"  -> No data extracted")
        except Exception as e:
            print(f"  -> Error: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"\nSaved {len(result)} total samples to {output_path}")

    return result


if __name__ == "__main__":
    # Example usage
    demo_path = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\1-03f67162-abf3-437e-b575-86538acdb399-1-1.dem"

    if Path(demo_path).exists():
        print(f"Parsing: {demo_path}")
        df = parse_buy_data(demo_path)
        print(f"\nParsed {len(df)} player-rounds")
        if len(df) > 0:
            print(df[['round_num', 'team_num', 'money_start', 'money_spent', 'items_bought']].head(20))
            df.to_csv('buy_data_test.csv', index=False)
    else:
        print(f"Demo file not found: {demo_path}")
