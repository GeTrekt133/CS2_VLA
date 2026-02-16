#!/usr/bin/env python3
"""
GSI Monitor - просмотр данных из CS2 в реальном времени.

Просто показывает что приходит от CS2, без записи и без модели.

Usage:
    python gsi/run_monitor.py
    python gsi/run_monitor.py --raw       # показать сырой JSON
    python gsi/run_monitor.py --verbose   # подробный вывод
    python gsi/run_monitor.py --save log.json  # сохранить в файл
"""
import sys
import json
import time
import argparse
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsi.server.http_server import GSIServer, GSIPayload
from gsi.adapter.state_adapter import GSIStateAdapter, GameStateSnapshot

# Global flag for shutdown
_shutdown = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C signal."""
    global _shutdown
    _shutdown = True


class GSIMonitor:
    """Монитор GSI данных."""

    def __init__(
        self,
        raw_mode: bool = False,
        verbose: bool = False,
        save_path: Optional[str] = None
    ):
        self.raw_mode = raw_mode
        self.verbose = verbose
        self.save_path = save_path
        self.adapter = GSIStateAdapter()
        self.update_count = 0
        self.last_snapshot: GameStateSnapshot = None

        # Для сохранения в файл
        self.log_entries: List[dict] = []

    def on_update(self, payload: GSIPayload):
        """Обработка GSI обновления."""
        self.update_count += 1

        # Сохранение в лог если включено
        if self.save_path:
            self._save_entry(payload)

        # Очистка экрана (Windows)
        print("\033[H\033[J", end="")

        print("=" * 70)
        status = f"Updates: {self.update_count}"
        if self.save_path:
            status += f"  |  Saving to: {self.save_path}"
        print(f"  CS2 GSI Monitor  |  {status}  |  Ctrl+C to stop")
        print("=" * 70)

        if self.raw_mode:
            self._print_raw(payload)
        else:
            self._print_parsed(payload)

    def _save_entry(self, payload: GSIPayload):
        """Сохранить запись в лог."""
        snapshot = self.adapter.parse(payload)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "update_num": self.update_count,
            "raw": payload.raw_data,
        }

        # Добавляем распарсенные данные если есть
        if snapshot:
            entry["parsed"] = {
                "side": snapshot.side,
                "hp": snapshot.hp,
                "armor": snapshot.armor,
                "helmet": snapshot.helmet,
                "weapon": snapshot.weapon,
                "ammo": snapshot.ammo,
                "weapon_list": snapshot.weapon_list,
                "round_num": snapshot.round_num,
                "phase": snapshot.phase,
                "round_time_left": snapshot.round_time_left,
                "bomb_planted": snapshot.bomb_planted,
                "freeze_time": snapshot.freeze_time,
                "ct_alive": snapshot.ct_alive,
                "t_alive": snapshot.t_alive,
                "score_ct": snapshot.score_ct,
                "score_t": snapshot.score_t,
                "money": snapshot.money,
                "xp_level": snapshot.xp_level,
            }

            # Добавляем state_vec
            state_vec = self.adapter.to_state_vec(snapshot)
            entry["state_vec"] = state_vec.tolist()

        self.log_entries.append(entry)

    def save_log(self):
        """Сохранить весь лог в файл."""
        if not self.save_path or not self.log_entries:
            return

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.log_entries, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Сохранено {len(self.log_entries)} записей в {self.save_path}")

    def _print_raw(self, payload: GSIPayload):
        """Вывод сырого JSON."""
        print("\n[RAW JSON]")
        print("-" * 70)

        # Убираем слишком большие секции для читаемости
        data = payload.raw_data.copy()

        # Форматируем JSON
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])

        if len(json.dumps(data)) > 3000:
            print("\n... (truncated)")

    def _print_parsed(self, payload: GSIPayload):
        """Вывод распарсенных данных."""

        if not payload.is_in_game():
            print("\n  [!] Не в игре (меню/загрузка)")
            print("\n  Ожидание подключения к матчу...")
            return

        snapshot = self.adapter.parse(payload)
        if snapshot is None:
            print("\n  [!] Не удалось распарсить данные")
            return

        self.last_snapshot = snapshot

        # === Основная информация ===
        print("\n  ИГРОК")
        print("  " + "-" * 40)
        print(f"  Сторона:    {snapshot.side}")
        print(f"  HP:         {snapshot.hp} / 100")
        print(f"  Броня:      {snapshot.armor} / 100  {'[ШЛЕМ]' if snapshot.helmet else ''}")
        print(f"  Деньги:     ${snapshot.money}")
        if snapshot.xp_level > 0:
            print(f"  XP Level:   {snapshot.xp_level}")

        # === Оружие ===
        print("\n  ОРУЖИЕ")
        print("  " + "-" * 40)
        print(f"  Активное:   {snapshot.weapon}")
        print(f"  Патроны:    {snapshot.ammo}")
        print(f"  Инвентарь:  {', '.join(snapshot.weapon_list)}")

        # === Раунд ===
        print("\n  РАУНД")
        print("  " + "-" * 40)
        print(f"  Номер:      {snapshot.round_num}")
        print(f"  Фаза:       {snapshot.phase}")
        print(f"  Счёт:       CT {snapshot.score_ct} : {snapshot.score_t} T")
        print(f"  Время:      {snapshot.round_time_left:.1f} сек")
        print(f"  Freeze:     {'Да' if snapshot.freeze_time else 'Нет'}")
        print(f"  Бомба:      {'УСТАНОВЛЕНА!' if snapshot.bomb_planted else 'нет'}")

        # === Игроки ===
        print("\n  ИГРОКИ")
        print("  " + "-" * 40)
        print(f"  CT живых:   {snapshot.ct_alive} / 5")
        print(f"  T живых:    {snapshot.t_alive} / 5")

        if self.verbose:
            self._print_verbose(payload, snapshot)

        # === state_vec ===
        print("\n  STATE_VEC (для модели)")
        print("  " + "-" * 40)
        state_vec = self.adapter.to_state_vec(snapshot)
        print(f"  Shape:      {state_vec.shape}")
        print(f"  Scalars:    hp={state_vec[0]:.2f} armor={state_vec[1]:.2f} "
              f"helmet={state_vec[2]:.0f} ammo={state_vec[3]:.2f}")
        print(f"              ct_alive={state_vec[4]:.2f} t_alive={state_vec[5]:.2f} "
              f"time={state_vec[6]:.2f}")
        print(f"              bomb={state_vec[7]:.0f} freeze={state_vec[8]:.0f}")
        print(f"  Side:       {list(state_vec[9:11])}")

    def _print_verbose(self, payload: GSIPayload, snapshot: GameStateSnapshot):
        """Подробный вывод."""
        print("\n  ПОДРОБНО")
        print("  " + "-" * 40)
        print(f"  Timestamp:  {snapshot.timestamp:.3f}")
        print(f"  Est. tick:  {snapshot.tick}")

        # Все игроки
        if payload.allplayers:
            print("\n  ВСЕ ИГРОКИ:")
            for pid, pdata in payload.allplayers.items():
                team = pdata.get("team", "?")
                state = pdata.get("state", {})
                hp = state.get("health", 0)
                name = pdata.get("name", pid[:8])
                status = "ALIVE" if hp > 0 else "DEAD"
                print(f"    [{team}] {name:15s} HP:{hp:3d} {status}")


def main():
    parser = argparse.ArgumentParser(description="CS2 GSI Monitor")
    parser.add_argument(
        "--raw", "-r",
        action="store_true",
        help="Показать сырой JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3000,
        help="Порт GSI сервера (default: 3000)"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Сохранить все данные в JSON файл"
    )

    args = parser.parse_args()

    monitor = GSIMonitor(
        raw_mode=args.raw,
        verbose=args.verbose,
        save_path=args.save
    )

    print("=" * 70)
    print("  CS2 GSI Monitor")
    print("=" * 70)
    print()
    print("  1. Убедитесь, что gamestate_integration_cs2nn.cfg установлен в CS2")
    print("  2. Запустите CS2 и зайдите в матч")
    print("  3. Данные появятся автоматически")
    print()
    print(f"  Сервер: http://127.0.0.1:{args.port}")
    print("  Режим:  " + ("RAW JSON" if args.raw else "Parsed") +
          (" + Verbose" if args.verbose else ""))
    if args.save:
        print(f"  Запись: {args.save}")
    print()
    print("=" * 70)
    print("\n  Ожидание данных от CS2...")

    server = GSIServer(port=args.port)
    server.register_callback(monitor.on_update)

    # Регистрируем обработчики для корректного завершения
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Сохранение при любом выходе
    def cleanup():
        server.stop()
        monitor.save_log()
        print(f"[OK] Получено обновлений: {monitor.update_count}")

    atexit.register(cleanup)

    server.start()

    global _shutdown
    try:
        while not _shutdown:
            time.sleep(0.5)
            # Периодическое сохранение каждые 50 обновлений
            if args.save and monitor.update_count > 0 and monitor.update_count % 50 == 0:
                monitor.save_log()
    except KeyboardInterrupt:
        pass

    print("\n\n[!] Остановка...")
    # cleanup вызовется автоматически через atexit


if __name__ == "__main__":
    main()
