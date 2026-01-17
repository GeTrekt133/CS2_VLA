from demoparser2 import DemoParser
import pandas as pd


def parse_ticks_csv(path):
    parser = DemoParser(path)

    ticks_df = parser.parse_ticks(['tick', 'steamid', 'X', 'Y', 'Z', 'is_alive', 'spotted', 'yaw', 'pitch', 'user_id', 'team_num'])
    ticks_df = ticks_df.sort_values('tick')

    bomb_pick = parser.parse_event('bomb_pickup', player=['X','Y']).assign(bomb_dropped=False, bomb_planted=False)
    bomb_drop = parser.parse_event('bomb_dropped', player=['X','Y']).assign(bomb_dropped=True, bomb_planted=False)
    bomb_planted = parser.parse_event('bomb_planted', player=['X','Y']).assign(bomb_dropped=False, bomb_planted=True)
    bomb = pd.concat([bomb_pick, bomb_drop, bomb_planted], ignore_index=True).sort_values('tick')

    bomb['owner'] = bomb.apply(lambda r: None if (r['bomb_dropped'] or r['bomb_planted']) else r['user_steamid'], axis=1)

    bomb = bomb[['tick','bomb_dropped','bomb_planted','owner','user_X','user_Y']]

    ticks_df = ticks_df.merge(bomb, how='left', on='tick')
    ticks_df = ticks_df.sort_values('tick').reset_index(drop=True)

    ticks_df[['bomb_dropped','bomb_planted','owner','user_X','user_Y']] = \
        ticks_df[['bomb_dropped','bomb_planted','owner','user_X','user_Y']].ffill()

    owner_ticks = ticks_df[['tick','steamid','X','Y']]
    ticks_df['owner'] = ticks_df['owner'].astype('Int64')
    owner_ticks['steamid'] = owner_ticks['steamid'].astype('Int64')
    ticks_df = ticks_df.merge(
        owner_ticks,
        left_on=['tick','owner'],
        right_on=['tick','steamid'],
        how='left',
        suffixes=('','_owner')
    )

    ticks_df['bombX'] = ticks_df.apply(lambda r: r['user_X'] if r['bomb_dropped'] or r['bomb_planted'] else r['X_owner'], axis=1)
    ticks_df['bombY'] = ticks_df.apply(lambda r: r['user_Y'] if r['bomb_dropped'] or r['bomb_planted'] else r['Y_owner'], axis=1)
    ticks_df = ticks_df.drop(columns=['user_X', 'user_Y', 'X_owner', 'Y_owner', 'steamid_owner'])

    return ticks_df


if __name__ == "__main__":
    ticks_df = parse_ticks_csv(r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\1-03f67162-abf3-437e-b575-86538acdb399-1-1.dem")
    ticks_df.to_csv('bbox.csv', index=False)