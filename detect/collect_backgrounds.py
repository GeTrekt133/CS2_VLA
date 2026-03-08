"""
Автоматический сбор фонов (пустых карт) для синтетического датасета.

Версия 2.0:
- Автоматический парсинг всех .dem файлов из папки CS2
- Смена оружия между сбором фонов для разнообразия
- Поддержка как CSV, так и автоматического парсинга

Требования:
- CS2 должна быть открыта
- Консоль доступна через клавишу `
- demoparser2 установлен (pip install demoparser2)

Использование:
  # Автоматический парсинг всех демок + сбор фонов
  python detect/collect_backgrounds.py --auto-parse --map de_dust2

  # Использование готового CSV (старый способ)
  python detect/collect_backgrounds.py --csv spotted_alive_players.csv --map de_dust2
"""

import pyautogui
import pandas as pd
import numpy as np
import time
import os
import threading
from pathlib import Path
from typing import List, Optional
import glob


# ==================== Stop Signal (global hotkey) ====================

_stop_event = threading.Event()

# F10 — не используется CS2, работает даже в fullscreen
STOP_KEY = 'f10'


def _start_stop_listener():
    """Запускает глобальный хук на F10 для остановки сбора (работает даже когда CS2 в фокусе)."""
    import keyboard  # pip install keyboard
    keyboard.on_press_key(STOP_KEY, lambda _: _stop_event.set())
    print(f"[Hotkey] Press {STOP_KEY.upper()} at any time to stop collection (works in CS2)")


# ==================== Weapon Management ====================

CS2_WEAPONS = [
    'weapon_ak47',
    'weapon_m4a1',
    'weapon_awp',
    'weapon_deagle',
    'weapon_glock',
    'weapon_usp_silencer',
    'weapon_famas',
    'weapon_galil',
    'weapon_aug',
    'weapon_sg556',
    'weapon_mp9',
    'weapon_mac10',
    'weapon_ump45',
    'weapon_p90',
    'weapon_bizon',
    'weapon_nova',
    'weapon_xm1014',
    'weapon_mag7',
    'weapon_sawedoff',
    'weapon_m249',
    'weapon_negev',
]


def give_weapon(weapon_name):
    """Выдаёт оружие через консоль."""
    enter_console_command(f'give {weapon_name}', delay=0.15)


def switch_to_random_weapon():
    """Меняет оружие на случайное из списка."""
    weapon = np.random.choice(CS2_WEAPONS)
    print(f"  Switching weapon: {weapon}")
    give_weapon(weapon)
    time.sleep(0.3)


# ==================== Demo Parsing ====================

def find_demo_files(csgo_dir: str = None) -> List[str]:
    """
    Находит все .dem файлы в папке CS2.

    Args:
        csgo_dir: путь к папке с демками (опционально)

    Returns:
        Список путей к .dem файлам
    """
    if csgo_dir is None:
        csgo_dir = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"

    demo_files = glob.glob(os.path.join(csgo_dir, "*.dem"))
    print(f"Found {len(demo_files)} demo files in {csgo_dir}")
    return demo_files


def parse_demo_for_positions(demo_path: str, tick_step: int = 16) -> pd.DataFrame:
    """
    Парсит демку и возвращает DataFrame с позициями игроков.
    Аналогично parse_demo.py, но возвращает результат напрямую.

    Args:
        demo_path: путь к .dem файлу
        tick_step: шаг между тиками (default: 16)

    Returns:
        DataFrame с колонками: tick, steamid, name, X, Y, Z, is_alive, spotted, yaw, pitch
    """
    try:
        from demoparser2 import DemoParser
    except ImportError:
        print("ERROR: demoparser2 not installed! Run: pip install demoparser2")
        return pd.DataFrame()

    print(f"  Parsing demo: {os.path.basename(demo_path)}...")

    try:
        parser = DemoParser(demo_path)

        # Парсим тики
        ticks_df = parser.parse_ticks([
            'tick',
            'steamid',
            'user_id',
            'name',
            'X',
            'Y',
            'Z',
            'is_alive',
            'spotted',
            'yaw',
            'pitch'
        ])

        # Фильтруем: жив И видим
        filtered_df = ticks_df[
            (ticks_df['is_alive'] == True) &
            (ticks_df['spotted'] == True)
        ].copy()

        # Сортируем и применяем шаг
        filtered_df = filtered_df.sort_values('tick').reset_index(drop=True)
        unique_ticks = filtered_df['tick'].unique()
        ticks_with_step = unique_ticks[::tick_step]
        filtered_df = filtered_df[filtered_df['tick'].isin(ticks_with_step)].reset_index(drop=True)

        print(f"    ✓ Extracted {len(filtered_df)} positions")
        return filtered_df

    except Exception as e:
        print(f"    ✗ Failed to parse demo: {e}")
        return pd.DataFrame()


def parse_all_demos(csgo_dir: str = None, tick_step: int = 16) -> pd.DataFrame:
    """
    Парсит все демки в папке CS2 и объединяет результаты.

    Args:
        csgo_dir: путь к папке с демками (опционально)
        tick_step: шаг между тиками

    Returns:
        Объединённый DataFrame со всеми позициями
    """
    demo_files = find_demo_files(csgo_dir)

    if not demo_files:
        print("No demo files found!")
        return pd.DataFrame()

    all_positions = []

    for demo_path in demo_files:
        df = parse_demo_for_positions(demo_path, tick_step)
        if not df.empty:
            all_positions.append(df)

    if not all_positions:
        print("No positions extracted from any demo!")
        return pd.DataFrame()

    # Объединяем все DataFrame
    combined_df = pd.concat(all_positions, ignore_index=True)
    print(f"\n✓ Total positions from all demos: {len(combined_df)}")

    return combined_df


# ==================== Console Commands ====================

def enter_console_command(command, delay=0.3):
    """Вводит команду в консоль CS2."""
    pyautogui.press("`")
    time.sleep(delay)
    pyautogui.typewrite(f'{command}\n')
    time.sleep(delay)
    pyautogui.press("`")
    time.sleep(delay)


def setup_empty_map(map_name):
    """
    Загружает карту offline без ботов, включает noclip.
    HUD и viewmodel остаются видимыми (консистентность с геймплеем).
    """
    print(f"Loading map: {map_name}")

    pyautogui.press("`")
    time.sleep(0.5)
    pyautogui.typewrite(f'map {map_name}\n')
    time.sleep(20)
    pyautogui.press("`")
    time.sleep(0.5)

    commands = [
        'sv_cheats 1',
        'bot_kick',
        'mp_autoteambalance 0',
        'mp_limitteams 0',
        'mp_warmup_end',
        'noclip',
    ]

    for cmd in commands:
        enter_console_command(cmd)
        time.sleep(0.3)

    # Убираем курсор из зоны UI
    screen_width, screen_height = pyautogui.size()
    pyautogui.moveTo(screen_width - 10, screen_height - 10)

    time.sleep(2)
    print(f"Map {map_name} ready (noclip on, bots kicked)")


def extract_positions_from_csv(csv_path, min_distance=200):
    """
    Извлекает уникальные позиции из CSV (parse_demo.py output).
    Greedy фильтрация: оставляет только позиции на расстоянии >= min_distance.

    Args:
        csv_path: путь к CSV (tick, steamid, user_id, name, X, Y, Z, is_alive, spotted, yaw, pitch)
        min_distance: минимальное расстояние между позициями (game units)

    Returns:
        List of [x, y, z, yaw, pitch]
    """
    df = pd.read_csv(csv_path)

    if 'is_alive' in df.columns:
        df = df[df['is_alive'] == True]

    all_positions = df[['X', 'Y', 'Z', 'yaw', 'pitch']].values.tolist()

    if len(all_positions) == 0:
        print("No positions found in CSV!")
        return []

    # Greedy distance filter
    filtered = [all_positions[0]]
    for pos in all_positions[1:]:
        min_dist = min(
            np.sqrt((pos[0] - f[0])**2 + (pos[1] - f[1])**2 + (pos[2] - f[2])**2)
            for f in filtered
        )
        if min_dist >= min_distance:
            filtered.append(pos)

    print(f"Positions: {len(all_positions)} total -> {len(filtered)} unique (min_distance={min_distance})")
    return filtered


def collect_backgrounds(
    positions,
    output_dir,
    map_name,
    angles_per_position=3,
    angle_variation=45.0,
    delay=0.5,
    setup_map=True,
    weapon_change_interval=75,
    random_weapons=True
):
    """
    Телепортируется на позиции и делает скриншоты.

    Args:
        positions: список [x, y, z, yaw, pitch]
        output_dir: директория для сохранения
        map_name: название карты (для именования файлов)
        angles_per_position: кол-во снимков на позицию (с вариацией углов)
        angle_variation: максимальное отклонение yaw (градусы)
        delay: задержка после телепортации (секунды, для рендеринга)
        setup_map: загружать ли карту (False если уже загружена)
        weapon_change_interval: менять оружие каждые N фонов (0 = не менять)
        random_weapons: использовать случайное оружие (True) или циклически (False)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Считаем существующие файлы для resumability
    existing = len(list(output_path.glob(f'bg_{map_name}_*.jpg')))

    if setup_map:
        setup_empty_map(map_name)

    count = existing
    total_new = len(positions) * angles_per_position
    weapon_idx = 0  # Для циклической смены оружия

    print(f"\nCollecting {total_new} backgrounds "
          f"({len(positions)} positions x {angles_per_position} angles)")
    if existing > 0:
        print(f"Resuming from {existing} existing backgrounds")

    if weapon_change_interval > 0:
        print(f"Weapon change: every {weapon_change_interval} backgrounds "
              f"({'random' if random_weapons else 'cyclic'})")
        # Выдаём первое оружие сразу
        if random_weapons:
            switch_to_random_weapon()
        else:
            give_weapon(CS2_WEAPONS[weapon_idx])
            weapon_idx += 1

    _stop_event.clear()
    _start_stop_listener()

    try:
        for i, (x, y, z, yaw, pitch) in enumerate(positions):
            if _stop_event.is_set():
                print("\n[ESC] Stop requested!")
                break

            for angle_idx in range(angles_per_position):
                if _stop_event.is_set():
                    break

                # Смена оружия через интервал
                if weapon_change_interval > 0 and count > 0 and count % weapon_change_interval == 0:
                    if random_weapons:
                        switch_to_random_weapon()
                    else:
                        give_weapon(CS2_WEAPONS[weapon_idx % len(CS2_WEAPONS)])
                        weapon_idx += 1
                # Первый снимок — оригинальный угол, остальные — с вариацией
                if angle_idx == 0:
                    cur_yaw = yaw
                    cur_pitch = pitch
                else:
                    cur_yaw = yaw + np.random.uniform(-angle_variation, angle_variation)
                    cur_pitch = np.clip(
                        pitch + np.random.uniform(-angle_variation / 3, angle_variation / 3),
                        -89, 89
                    )

                # Телепортация
                # setang: pitch yaw roll (pitch первым!)
                enter_console_command(f'setpos {x:.1f} {y:.1f} {z:.1f}', delay=0.1)
                enter_console_command(f'setang {cur_pitch:.1f} {cur_yaw:.1f} 0', delay=0.1)
                time.sleep(delay)

                # Скриншот
                filename = f"bg_{map_name}_{count:06d}.jpg"
                screenshot = pyautogui.screenshot()
                screenshot.save(str(output_path / filename))
                count += 1

                if (count - existing) % 50 == 0:
                    print(f"  Progress: {count - existing}/{total_new} "
                          f"(position {i+1}/{len(positions)})")

    except KeyboardInterrupt:
        print(f"\n[Ctrl+C] Interrupted!")

    # Cleanup keyboard listener
    try:
        import keyboard
        keyboard.unhook_all()
    except Exception:
        pass

    stopped_by = "ESC" if _stop_event.is_set() else "completed"
    print(f"\nDone ({stopped_by})! Collected {count - existing} new backgrounds "
          f"(total in folder: {count})")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect empty map backgrounds for synthetic YOLO dataset (v2.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-parse all demos and collect backgrounds
  python detect/collect_backgrounds.py --auto-parse --map de_dust2

  # Use existing CSV
  python detect/collect_backgrounds.py --csv positions.csv --map de_mirage

  # Custom weapon change interval
  python detect/collect_backgrounds.py --auto-parse --map de_inferno --weapon-interval 100
        """
    )

    # Режим работы: автопарсинг или CSV
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--auto-parse', action='store_true',
                             help='Automatically parse all .dem files from CS2 directory')
    input_group.add_argument('--csv', type=str,
                             help='CSV from parse_demo.py with player positions')

    parser.add_argument('--csgo-dir', type=str,
                        default=r'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo',
                        help='CS2 directory with demo files (for --auto-parse)')
    parser.add_argument('--map', type=str, required=True,
                        help='Map name (de_dust2, de_mirage, de_inferno, etc.)')
    parser.add_argument('--output', type=str, default=r'D:\backgrounds',
                        help='Output directory for backgrounds (default: D:\\backgrounds)')
    parser.add_argument('--min-distance', type=float, default=200,
                        help='Min distance between positions in game units (default: 200)')
    parser.add_argument('--angles', type=int, default=3,
                        help='Screenshots per position with angle variation (default: 3)')
    parser.add_argument('--angle-variation', type=float, default=45.0,
                        help='Max yaw variation in degrees (default: 45)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay after teleport in seconds (default: 0.5)')
    parser.add_argument('--no-setup', action='store_true',
                        help='Skip map loading (map already loaded and ready)')
    parser.add_argument('--weapon-interval', type=int, default=75,
                        help='Change weapon every N backgrounds (0=no change, default: 75)')
    parser.add_argument('--no-random-weapons', action='store_true',
                        help='Use cyclic weapon rotation instead of random')
    parser.add_argument('--start-delay', type=int, default=5,
                        help='Seconds to wait before starting (default: 5)')
    parser.add_argument('--tick-step', type=int, default=16,
                        help='Tick step for demo parsing (default: 16, for --auto-parse)')
    args = parser.parse_args()

    print("=" * 60)
    print("Background Collection v2.0 - with Auto-Parse & Weapon Rotation")
    print("=" * 60)

    print(f"  Map:             {args.map}")
    print(f"  Output:          {args.output}")
    print(f"  Min distance:    {args.min_distance}")
    print(f"  Angles/position: {args.angles}")
    print(f"  Angle variation: {args.angle_variation}°")
    print(f"  Delay:           {args.delay}s")
    print(f"  Weapon interval: {args.weapon_interval} "
          f"({'cyclic' if args.no_random_weapons else 'random'})")
    print("=" * 60)

    # Ждём переключения в CS2
    print(f"\nSwitch to CS2 window! Starting in {args.start_delay} seconds...")
    for i in range(args.start_delay, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    if args.auto_parse:
        # Последовательный режим: демка → скриншоты → следующая демка → ...
        demo_files = find_demo_files(args.csgo_dir)

        if not demo_files:
            print("No demo files found!")
            exit(1)

        print(f"\nProcessing {len(demo_files)} demos sequentially...")
        total_collected = 0
        setup_done = False

        for demo_idx, demo_path in enumerate(demo_files):
            print(f"\n{'='*60}")
            print(f"Demo {demo_idx + 1}/{len(demo_files)}: {os.path.basename(demo_path)}")
            print(f"{'='*60}")

            # Парсим одну демку
            df = parse_demo_for_positions(demo_path, args.tick_step)
            if df.empty:
                print("  No positions, skipping.")
                continue

            # Извлекаем и фильтруем позиции
            all_pos = df[['X', 'Y', 'Z', 'yaw', 'pitch']].values.tolist()
            positions = [all_pos[0]]
            for pos in all_pos[1:]:
                min_dist = min(
                    np.sqrt((pos[0] - f[0])**2 + (pos[1] - f[1])**2 + (pos[2] - f[2])**2)
                    for f in positions
                )
                if min_dist >= args.min_distance:
                    positions.append(pos)

            print(f"  {len(all_pos)} raw -> {len(positions)} unique positions")

            if not positions:
                continue

            # Карту загружаем только один раз (первая демка)
            need_setup = (not args.no_setup) and (not setup_done)

            count = collect_backgrounds(
                positions=positions,
                output_dir=args.output,
                map_name=args.map,
                angles_per_position=args.angles,
                angle_variation=args.angle_variation,
                delay=args.delay,
                setup_map=need_setup,
                weapon_change_interval=args.weapon_interval,
                random_weapons=not args.no_random_weapons
            )
            setup_done = True
            total_collected = count

        print(f"\n{'='*60}")
        print(f"All demos processed! Total backgrounds: {total_collected}")
        print(f"{'='*60}")

    else:
        # CSV режим — как раньше
        print(f"  Mode:            CSV input")
        print(f"  CSV:             {args.csv}")

        positions = extract_positions_from_csv(args.csv, args.min_distance)

        if not positions:
            print("No positions extracted!")
            exit(1)

        estimated_time = len(positions) * args.angles * (args.delay + 0.8)
        print(f"\nEstimated time: ~{estimated_time/60:.1f} min "
              f"({len(positions)} positions x {args.angles} angles)")

        collect_backgrounds(
            positions=positions,
            output_dir=args.output,
            map_name=args.map,
            angles_per_position=args.angles,
            angle_variation=args.angle_variation,
            delay=args.delay,
            setup_map=not args.no_setup,
            weapon_change_interval=args.weapon_interval,
            random_weapons=not args.no_random_weapons
        )
