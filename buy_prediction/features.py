"""
Feature engineering for buy prediction.
Only uses features available via GSI at inference time.

Features:
  - money, money_log
  - equipment_value, equipment_value_log
  - round_num
  - team_score, enemy_score, score_diff
  - loss_streak, loss_bonus (estimated $)
  - is_t_side, is_ct_side
  - is_pistol_round, round_after_pistol
  - can_full_buy, can_awp, can_half_buy, should_eco
  - is_first_half, is_last_round_half

Targets (multi-label for exact buy):
  - primary: none=0, pistol_upgrade=1, smg=2, rifle=3, awp=4, scout=5
  - armor: none=0, vest=1, vesthelm=2
  - buy_smoke, buy_flash, buy_he, buy_molotov, buy_defuser: binary
"""
import ast
import numpy as np
import pandas as pd
from typing import List, Tuple


# CS2 loss bonus table (after N consecutive losses)
LOSS_BONUS = {0: 1400, 1: 1900, 2: 2400, 3: 2900, 4: 3400}

# Item name normalization (demo format -> canonical)
ITEM_ALIASES = {
    'ak-47': 'ak47', 'ak47': 'ak47',
    'm4a1-s': 'm4a1_silencer', 'm4a1_silencer': 'm4a1_silencer',
    'm4a4': 'm4a1', 'm4a1': 'm4a1',
    'galil ar': 'galilar', 'galilar': 'galilar',
    'famas': 'famas',
    'sg 553': 'sg556', 'sg556': 'sg556',
    'aug': 'aug',
    'awp': 'awp',
    'ssg 08': 'ssg08', 'ssg08': 'ssg08',
    'scar-20': 'scar20', 'scar20': 'scar20',
    'g3sg1': 'g3sg1',
    'mac-10': 'mac10', 'mac10': 'mac10',
    'mp9': 'mp9',
    'mp7': 'mp7',
    'ump-45': 'ump45', 'ump45': 'ump45',
    'p90': 'p90',
    'pp-bizon': 'bizon', 'bizon': 'bizon',
    'mp5-sd': 'mp5sd', 'mp5sd': 'mp5sd',
    'glock-18': 'glock', 'glock': 'glock',
    'usp-s': 'usp_silencer', 'usp_silencer': 'usp_silencer',
    'p2000': 'hkp2000', 'hkp2000': 'hkp2000',
    'p250': 'p250',
    'tec-9': 'tec9', 'tec9': 'tec9',
    'five-seven': 'fiveseven', 'fiveseven': 'fiveseven',
    'cz75-auto': 'cz75_auto', 'cz75_auto': 'cz75_auto',
    'desert eagle': 'deagle', 'deagle': 'deagle',
    'r8 revolver': 'revolver', 'revolver': 'revolver',
    'dual berettas': 'elite', 'elite': 'elite',
    'nova': 'nova', 'xm1014': 'xm1014',
    'sawed-off': 'sawedoff', 'sawedoff': 'sawedoff',
    'mag-7': 'mag7', 'mag7': 'mag7',
    'm249': 'm249', 'negev': 'negev',
    'kevlar vest': 'vest', 'vest': 'vest',
    'kevlar & helmet': 'vesthelm', 'vesthelm': 'vesthelm',
    'kevlar + helmet': 'vesthelm',
    'defuse kit': 'defuser', 'defuser': 'defuser',
    'zeus x27': 'taser', 'taser': 'taser',
    'molotov': 'molotov',
    'incendiary grenade': 'incgrenade', 'incgrenade': 'incgrenade',
    'decoy grenade': 'decoy', 'decoy': 'decoy',
    'flashbang': 'flashbang',
    'high explosive grenade': 'hegrenade', 'hegrenade': 'hegrenade',
    'smoke grenade': 'smokegrenade', 'smokegrenade': 'smokegrenade',
}

PISTOL_UPGRADES = {'p250', 'tec9', 'fiveseven', 'cz75_auto', 'deagle', 'revolver', 'elite'}
SMGS = {'mac10', 'mp9', 'mp7', 'ump45', 'p90', 'bizon', 'mp5sd'}
RIFLES = {'ak47', 'm4a1', 'm4a1_silencer', 'famas', 'galilar', 'sg556', 'aug'}
SNIPERS_AWP = {'awp', 'scar20', 'g3sg1'}
SNIPERS_SCOUT = {'ssg08'}
HEAVIES = {'nova', 'xm1014', 'sawedoff', 'mag7', 'm249', 'negev'}

# Weapon prices for cost estimation
WEAPON_PRICES = {
    'glock': 0, 'usp_silencer': 0, 'hkp2000': 0,
    'p250': 300, 'tec9': 500, 'fiveseven': 500,
    'cz75_auto': 500, 'deagle': 700, 'revolver': 600, 'elite': 300,
    'mac10': 1050, 'mp9': 1250, 'mp7': 1500,
    'ump45': 1200, 'p90': 2350, 'bizon': 1400, 'mp5sd': 1500,
    'famas': 2050, 'galilar': 1800,
    'ak47': 2700, 'm4a1': 3100, 'm4a1_silencer': 2900,
    'sg556': 3000, 'aug': 3300,
    'ssg08': 1700, 'awp': 4750, 'scar20': 5000, 'g3sg1': 5000,
    'nova': 1050, 'xm1014': 2000, 'sawedoff': 1100, 'mag7': 1300,
    'm249': 5200, 'negev': 1700,
    'vest': 650, 'vesthelm': 1000, 'defuser': 400, 'taser': 200,
    'molotov': 400, 'incgrenade': 600, 'decoy': 50,
    'flashbang': 200, 'hegrenade': 300, 'smokegrenade': 300,
}


def normalize_item(item: str) -> str:
    return ITEM_ALIASES.get(item.lower().strip(), item.lower().strip())


def parse_list_col(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or val in ('', '[]'):
        return []
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from raw parsed data. Only GSI-available features."""
    f = pd.DataFrame()

    # Money
    f['money'] = df['money'].fillna(0).astype(int)
    f['money_log'] = np.log1p(f['money'])

    # Equipment value
    f['equipment_value'] = df['equipment_value'].fillna(0).astype(int)
    f['equipment_value_log'] = np.log1p(f['equipment_value'])
    f['has_equipment'] = (f['equipment_value'] > 1000).astype(int)

    # Round
    f['round_num'] = df['round_num']
    f['is_pistol_round'] = df['round_num'].isin([1, 13]).astype(int)
    f['round_after_pistol'] = df['round_num'].isin([2, 14]).astype(int)

    # Score
    f['team_score'] = df['team_score']
    f['enemy_score'] = df['enemy_score']
    f['score_diff'] = df['team_score'] - df['enemy_score']
    f['total_rounds'] = df['team_score'] + df['enemy_score']
    f['is_winning'] = (f['score_diff'] > 0).astype(int)
    f['is_losing'] = (f['score_diff'] < 0).astype(int)

    # Loss streak and bonus
    f['loss_streak'] = df['loss_streak'].fillna(0).astype(int).clip(0, 4)
    f['loss_bonus'] = f['loss_streak'].map(LOSS_BONUS)

    # Side
    f['is_t_side'] = (df['team_num'] == 2).astype(int)
    f['is_ct_side'] = (df['team_num'] == 3).astype(int)

    # Half
    f['is_first_half'] = (df['round_num'] <= 12).astype(int)
    f['is_last_round_half'] = df['round_num'].isin([12, 24]).astype(int)
    f['rounds_in_half'] = df['round_num'].apply(
        lambda r: r if r <= 12 else r - 12
    )

    # Buy thresholds
    f['can_full_buy'] = (f['money'] >= 4100).astype(int)
    f['can_awp'] = (f['money'] >= 5750).astype(int)
    f['can_half_buy'] = (f['money'] >= 2000).astype(int)
    f['should_eco'] = (f['money'] < 1500).astype(int)

    # Combined economic signals
    f['money_after_loss_bonus'] = f['money'] + f['loss_bonus']

    return f


def classify_primary(items: List[str]) -> int:
    """Classify primary weapon purchase: 0=none, 1=pistol_upgrade, 2=smg, 3=rifle, 4=awp, 5=scout."""
    normalized = [normalize_item(i) for i in items]
    for w in normalized:
        if w in SNIPERS_AWP:
            return 4
        if w in RIFLES:
            return 3
        if w in SNIPERS_SCOUT:
            return 5
        if w in SMGS:
            return 2
        if w in HEAVIES:
            return 2  # treat heavy as smg tier
        if w in PISTOL_UPGRADES:
            return 1
    return 0


def classify_armor(items: List[str]) -> int:
    """Classify armor: 0=none, 1=vest, 2=vesthelm."""
    normalized = [normalize_item(i) for i in items]
    if 'vesthelm' in normalized:
        return 2
    if 'vest' in normalized:
        return 1
    return 0


def has_item(items: List[str], target: str) -> int:
    normalized = [normalize_item(i) for i in items]
    return int(target in normalized)


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-label targets for exact buy prediction."""
    items_col = df['items_bought'].apply(parse_list_col)

    t = pd.DataFrame()
    t['primary'] = items_col.apply(classify_primary)
    t['armor'] = items_col.apply(classify_armor)
    t['buy_smoke'] = items_col.apply(lambda x: has_item(x, 'smokegrenade'))
    t['buy_flash'] = items_col.apply(lambda x: has_item(x, 'flashbang'))
    t['buy_he'] = items_col.apply(lambda x: has_item(x, 'hegrenade'))
    t['buy_molotov'] = items_col.apply(
        lambda x: has_item(x, 'molotov') or has_item(x, 'incgrenade')
    )
    t['buy_defuser'] = items_col.apply(lambda x: has_item(x, 'defuser'))

    return t


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: raw data -> (features, targets)."""
    df = df.copy()
    if 'items_bought' in df.columns:
        df['items_bought'] = df['items_bought'].apply(parse_list_col)

    features = engineer_features(df)
    targets = create_targets(df)
    return features, targets


# Human-readable label maps
PRIMARY_LABELS = {0: 'none', 1: 'pistol_upgrade', 2: 'smg', 3: 'rifle', 4: 'awp', 5: 'scout'}
ARMOR_LABELS = {0: 'none', 1: 'vest', 2: 'vesthelm'}
