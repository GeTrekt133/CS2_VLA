"""
Feature engineering for buy prediction model.
Transforms raw parsed data into ML-ready features.
"""
import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Tuple, Optional


def parse_list_column(val):
    """Parse a list column that may be string or already a list."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == '' or val == '[]':
        return []
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


# CS2 weapon categories and prices
WEAPON_PRICES = {
    # Pistols
    'glock': 0, 'usp_silencer': 0, 'hkp2000': 0,
    'p250': 300, 'tec9': 500, 'fiveseven': 500,
    'cz75_auto': 500, 'deagle': 700, 'revolver': 600,
    'elite': 300,
    # SMGs
    'mac10': 1050, 'mp9': 1250, 'mp7': 1500,
    'ump45': 1200, 'p90': 2350, 'bizon': 1400,
    'mp5sd': 1500,
    # Rifles
    'famas': 2050, 'galilar': 1800,
    'ak47': 2700, 'm4a1': 3100, 'm4a1_silencer': 2900,
    'sg556': 3000, 'aug': 3300,
    'ssg08': 1700, 'awp': 4750,
    'scar20': 5000, 'g3sg1': 5000,
    # Heavy
    'nova': 1050, 'xm1014': 2000, 'sawedoff': 1100, 'mag7': 1300,
    'm249': 5200, 'negev': 1700,
    # Gear
    'vest': 650, 'vesthelm': 1000,
    'defuser': 400,
    'molotov': 400, 'incgrenade': 600,
    'decoy': 50, 'flashbang': 200,
    'hegrenade': 300, 'smokegrenade': 300,
    'taser': 200,
}

WEAPON_CATEGORIES = {
    'pistol': ['glock', 'usp_silencer', 'hkp2000', 'p250', 'tec9', 'fiveseven',
               'cz75_auto', 'deagle', 'revolver', 'elite'],
    'smg': ['mac10', 'mp9', 'mp7', 'ump45', 'p90', 'bizon', 'mp5sd'],
    'rifle': ['famas', 'galilar', 'ak47', 'm4a1', 'm4a1_silencer', 'sg556', 'aug', 'm4a4'],
    'sniper': ['ssg08', 'awp', 'scar20', 'g3sg1'],
    'heavy': ['nova', 'xm1014', 'sawedoff', 'mag7', 'm249', 'negev'],
    'grenade': ['molotov', 'incgrenade', 'decoy', 'flashbang', 'hegrenade', 'smokegrenade'],
    'gear': ['vest', 'vesthelm', 'defuser', 'taser'],
}

# Mapping from demo display names to internal names
ITEM_NAME_MAP = {
    # Rifles
    'ak-47': 'ak47',
    'm4a1-s': 'm4a1_silencer',
    'm4a4': 'm4a4',
    'galil ar': 'galilar',
    'famas': 'famas',
    'sg 553': 'sg556',
    'aug': 'aug',
    # Snipers
    'awp': 'awp',
    'ssg 08': 'ssg08',
    'scar-20': 'scar20',
    'g3sg1': 'g3sg1',
    # SMGs
    'mac-10': 'mac10',
    'mp9': 'mp9',
    'mp7': 'mp7',
    'ump-45': 'ump45',
    'p90': 'p90',
    'pp-bizon': 'bizon',
    'mp5-sd': 'mp5sd',
    # Pistols
    'glock-18': 'glock',
    'usp-s': 'usp_silencer',
    'p2000': 'hkp2000',
    'p250': 'p250',
    'tec-9': 'tec9',
    'five-seven': 'fiveseven',
    'cz75-auto': 'cz75_auto',
    'desert eagle': 'deagle',
    'r8 revolver': 'revolver',
    'dual berettas': 'elite',
    # Heavy
    'nova': 'nova',
    'xm1014': 'xm1014',
    'sawed-off': 'sawedoff',
    'mag-7': 'mag7',
    'm249': 'm249',
    'negev': 'negev',
    # Gear
    'kevlar vest': 'vest',
    'kevlar & helmet': 'vesthelm',
    'defuse kit': 'defuser',
    'zeus x27': 'taser',
    # Grenades
    'molotov': 'molotov',
    'incendiary grenade': 'incgrenade',
    'decoy grenade': 'decoy',
    'flashbang': 'flashbang',
    'high explosive grenade': 'hegrenade',
    'smoke grenade': 'smokegrenade',
}


def normalize_item_name(item: str) -> str:
    """Convert demo display name to internal name."""
    item_lower = item.lower().strip()
    return ITEM_NAME_MAP.get(item_lower, item_lower)


def get_weapon_category(weapon: str) -> str:
    """Get category for a weapon."""
    weapon = weapon.lower().replace('weapon_', '')
    for category, weapons in WEAPON_CATEGORIES.items():
        if weapon in weapons:
            return category
    return 'other'


def calculate_buy_type(items: List[str], money_spent: float) -> str:
    """
    Classify buy type:
    - full_buy: rifle + armor + utilities
    - force_buy: pistol upgrade or smg + some utility
    - eco: minimal spending
    - half_buy: smg/scout + armor
    """
    if not items:
        return 'eco'

    # Normalize item names from demo format to internal format
    items_normalized = [normalize_item_name(i) for i in items]

    has_rifle = any(w in items_normalized for w in WEAPON_CATEGORIES['rifle'])
    has_sniper = any(w in items_normalized for w in WEAPON_CATEGORIES['sniper'])
    has_smg = any(w in items_normalized for w in WEAPON_CATEGORIES['smg'])
    has_armor = any(w in items_normalized for w in ['vest', 'vesthelm'])

    if has_rifle or has_sniper:
        if has_armor:
            return 'full_buy'
        return 'force_buy'
    elif has_smg:
        if has_armor:
            return 'half_buy'
        return 'force_buy'
    elif money_spent < 500:
        return 'eco'
    else:
        return 'force_buy'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for buy prediction.

    Input columns: round_num, steamid, team_num, money_start, money_spent,
                   team_money_total, team_money_avg, teammates_money,
                   round_score_t, round_score_ct, equipment_value, items_bought

    Output: Feature matrix with engineered features.
    """
    features = pd.DataFrame()

    # Basic money features
    features['money'] = df['money_start']
    features['money_log'] = np.log1p(df['money_start'])
    features['team_money_total'] = df['team_money_total']
    features['team_money_avg'] = df['team_money_avg']

    # Money thresholds (common buy points)
    features['can_full_buy'] = (df['money_start'] >= 4100).astype(int)  # AK + armor + util
    features['can_awp'] = (df['money_start'] >= 5750).astype(int)  # AWP + armor
    features['can_half_buy'] = (df['money_start'] >= 2000).astype(int)  # SMG + armor
    features['money_for_save'] = (df['money_start'] < 1500).astype(int)

    # Team economy features
    features['team_can_full_buy'] = (df['team_money_avg'] >= 4100).astype(int)
    features['team_poor'] = (df['team_money_avg'] < 2000).astype(int)

    # Money difference from team avg
    features['money_diff_from_team'] = df['money_start'] - df['team_money_avg']
    features['money_ratio_to_team'] = df['money_start'] / (df['team_money_avg'] + 1)

    # Teammate money statistics
    def calc_teammate_stats(teammates_money):
        if not teammates_money or len(teammates_money) == 0:
            return 0, 0, 0, 0
        tm = np.array(teammates_money)
        return tm.min(), tm.max(), tm.std(), np.sum(tm >= 4100)

    teammate_stats = df['teammates_money'].apply(calc_teammate_stats)
    features['teammate_money_min'] = teammate_stats.apply(lambda x: x[0])
    features['teammate_money_max'] = teammate_stats.apply(lambda x: x[1])
    features['teammate_money_std'] = teammate_stats.apply(lambda x: x[2])
    features['teammates_can_buy'] = teammate_stats.apply(lambda x: x[3])

    # Round score features
    features['round_num'] = df['round_num']
    features['score_t'] = df['round_score_t']
    features['score_ct'] = df['round_score_ct']
    features['score_diff'] = df.apply(
        lambda r: r['round_score_t'] - r['round_score_ct'] if r['team_num'] == 2
                  else r['round_score_ct'] - r['round_score_t'], axis=1
    )
    features['total_rounds_played'] = df['round_score_t'] + df['round_score_ct']

    # Is losing / winning
    features['is_winning'] = (features['score_diff'] > 0).astype(int)
    features['is_losing'] = (features['score_diff'] < 0).astype(int)
    features['is_tied'] = (features['score_diff'] == 0).astype(int)

    # Half indicators (important for economy decisions)
    features['is_first_half'] = (df['round_num'] <= 12).astype(int)
    features['is_second_half'] = (df['round_num'] > 12).astype(int)
    features['rounds_until_half'] = 12 - df['round_num'].clip(upper=12)

    # Last round of half (often force buy)
    features['is_last_round_first_half'] = (df['round_num'] == 12).astype(int)
    features['is_last_round_second_half'] = (df['round_num'] == 24).astype(int)

    # Team side
    features['is_t_side'] = (df['team_num'] == 2).astype(int)
    features['is_ct_side'] = (df['team_num'] == 3).astype(int)

    # Economic loss bonus estimation (simplified)
    # In CS2: 1400/1900/2400/2900/3400 after 1/2/3/4/5 consecutive losses
    features['round_after_pistol'] = (df['round_num'].isin([2, 14])).astype(int)
    features['is_pistol_round'] = (df['round_num'].isin([1, 13])).astype(int)

    return features


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for buy prediction.

    Targets:
    - buy_type: categorical (eco, force, half, full)
    - buy_type_encoded: numeric encoding
    - should_buy_rifle: binary
    - should_buy_armor: binary
    - total_spend: regression target
    """
    targets = pd.DataFrame()

    # Calculate buy type
    targets['buy_type'] = df.apply(
        lambda r: calculate_buy_type(r['items_bought'], r['money_spent']), axis=1
    )

    # Encode buy type
    buy_type_map = {'eco': 0, 'force_buy': 1, 'half_buy': 2, 'full_buy': 3}
    targets['buy_type_encoded'] = targets['buy_type'].map(buy_type_map)

    # Binary targets for specific items
    def has_category(items, category):
        if not items:
            return 0
        items_normalized = [normalize_item_name(i) for i in items]
        return int(any(w in items_normalized for w in WEAPON_CATEGORIES.get(category, [])))

    targets['bought_rifle'] = df['items_bought'].apply(lambda x: has_category(x, 'rifle'))
    targets['bought_sniper'] = df['items_bought'].apply(lambda x: has_category(x, 'sniper'))
    targets['bought_smg'] = df['items_bought'].apply(lambda x: has_category(x, 'smg'))
    targets['bought_armor'] = df['items_bought'].apply(lambda x: has_category(x, 'gear'))
    targets['bought_grenade'] = df['items_bought'].apply(lambda x: has_category(x, 'grenade'))

    # Regression target
    targets['money_spent'] = df['money_spent']

    return targets


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare full dataset with features and targets.

    Returns: (features_df, targets_df)
    """
    # Parse list columns if they are strings (from CSV)
    if 'teammates_money' in df.columns:
        df = df.copy()
        df['teammates_money'] = df['teammates_money'].apply(parse_list_column)
    if 'items_bought' in df.columns:
        df['items_bought'] = df['items_bought'].apply(parse_list_column)

    features = engineer_features(df)
    targets = create_target(df)

    # Add identifiers
    features['steamid'] = df['steamid']
    features['demo_id'] = df.get('demo_id', 0)

    return features, targets


if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'round_num': [1, 2, 3],
        'steamid': [123, 123, 123],
        'team_num': [2, 2, 2],
        'money_start': [800, 3500, 5000],
        'money_spent': [0, 2700, 4500],
        'team_money_total': [4000, 17500, 25000],
        'team_money_avg': [800, 3500, 5000],
        'teammates_money': [[800, 800, 800, 800], [3500, 3500, 3500, 3500], [5000, 5000, 5000, 5000]],
        'round_score_t': [0, 1, 1],
        'round_score_ct': [0, 0, 1],
        'equipment_value': [200, 3000, 5000],
        'items_bought': [[], ['ak47', 'vest'], ['ak47', 'vesthelm', 'flashbang', 'smokegrenade']],
    })

    features, targets = prepare_dataset(sample_data)
    print("Features shape:", features.shape)
    print("Features columns:", features.columns.tolist())
    print("\nTargets shape:", targets.shape)
    print("Targets columns:", targets.columns.tolist())
    print("\nBuy types:", targets['buy_type'].value_counts().to_dict())
