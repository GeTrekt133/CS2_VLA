"""
Demo Position Parser — extracts all player positions + bomb from demo file.

Port of legacy data_collect/demo_parser.py:parse_ticks_csv().
Used by RadarRenderer to generate minimap overlays.

Output DataFrame columns:
    tick, steamid, X, Y, is_alive, spotted, yaw, user_id, team_num,
    bomb_dropped, bomb_planted, owner, bombX, bombY
"""

from demoparser2 import DemoParser
import pandas as pd


def parse_positions(demo_path: str) -> pd.DataFrame:
    """
    Parse all player positions and bomb state from a demo file.

    Args:
        demo_path: Path to .dem file

    Returns:
        DataFrame with per-tick rows for ALL players (not just target).
    """
    parser = DemoParser(demo_path)

    ticks_df = parser.parse_ticks([
        'tick', 'steamid', 'X', 'Y', 'is_alive', 'spotted',
        'yaw', 'user_id', 'team_num',
    ])
    ticks_df = ticks_df.sort_values('tick')

    # Parse bomb events
    bomb_pick = parser.parse_event('bomb_pickup', player=['X', 'Y']).assign(
        bomb_dropped=False, bomb_planted=False)
    bomb_drop = parser.parse_event('bomb_dropped', player=['X', 'Y']).assign(
        bomb_dropped=True, bomb_planted=False)
    bomb_planted = parser.parse_event('bomb_planted', player=['X', 'Y']).assign(
        bomb_dropped=False, bomb_planted=True)
    bomb = pd.concat([bomb_pick, bomb_drop, bomb_planted], ignore_index=True).sort_values('tick')

    bomb['owner'] = bomb.apply(
        lambda r: None if (r['bomb_dropped'] or r['bomb_planted']) else r['user_steamid'],
        axis=1,
    )
    bomb = bomb[['tick', 'bomb_dropped', 'bomb_planted', 'owner', 'user_X', 'user_Y']]

    # Merge bomb state into ticks
    ticks_df = ticks_df.merge(bomb, how='left', on='tick')
    ticks_df = ticks_df.sort_values('tick').reset_index(drop=True)

    # Forward-fill bomb state
    ticks_df[['bomb_dropped', 'bomb_planted', 'owner', 'user_X', 'user_Y']] = \
        ticks_df[['bomb_dropped', 'bomb_planted', 'owner', 'user_X', 'user_Y']].ffill()

    # Resolve bomb carrier position
    owner_ticks = ticks_df[['tick', 'steamid', 'X', 'Y']].copy()
    ticks_df['owner'] = ticks_df['owner'].astype('Int64')
    owner_ticks['steamid'] = owner_ticks['steamid'].astype('Int64')
    ticks_df = ticks_df.merge(
        owner_ticks,
        left_on=['tick', 'owner'],
        right_on=['tick', 'steamid'],
        how='left',
        suffixes=('', '_owner'),
    )

    ticks_df['bombX'] = ticks_df.apply(
        lambda r: r['user_X'] if r['bomb_dropped'] or r['bomb_planted'] else r['X_owner'],
        axis=1,
    )
    ticks_df['bombY'] = ticks_df.apply(
        lambda r: r['user_Y'] if r['bomb_dropped'] or r['bomb_planted'] else r['Y_owner'],
        axis=1,
    )
    ticks_df = ticks_df.drop(columns=['user_X', 'user_Y', 'X_owner', 'Y_owner', 'steamid_owner'])

    return ticks_df
