import pyautogui
import pandas as pd
import time
import os
from pathlib import Path
from datetime import datetime


def load_ticks_from_csv(csv_path):
    """Загружает список тиков из CSV"""
    df = pd.read_csv(csv_path)
    return df['tick'].unique()


def enter_console_command(command, delay=0.3):
    """Вводит команду в консоль CS2"""
    pyautogui.press("`")
    time.sleep(delay)
    pyautogui.typewrite(f'{command}\n')
    time.sleep(delay)
    pyautogui.press("`")
    time.sleep(delay)


def setup_demo(demo_path):
    """Инициализирует демку в CS2"""
    print(f"📂 Загружаю демку: {demo_path}")
    time.sleep(5)
    
    # Открываем консоль и загружаем демку
    pyautogui.press("`")
    time.sleep(0.5)
    pyautogui.typewrite(f'playdemo {demo_path}\n')
    time.sleep(15)  # Ждём загрузки демки
    
    # Закрываем консоль
    pyautogui.press("`")
    time.sleep(0.5)
    
    # Открываем демо-интерфейс
    enter_console_command('demoui')
    time.sleep(0.5)

    # Пауза демки
    enter_console_command('demo_pause')
    time.sleep(0.5)
    
    # Включаем bbox для всех игроков
    print("🎯 Включаю bbox для игроков...")
    for i in range(4, 14):
        enter_console_command(f'spec_player {i}')
        enter_console_command('cl_ent_bbox')
        time.sleep(0.2)
    time.sleep(0.5)

    screen_width, screen_height = pyautogui.size()
    pyautogui.moveTo(screen_width - 10, screen_height - 10)
    
    print("✅ Демка готова")


def take_screenshot(tick, user_id, output_dir):
    """Делает скриншот и сохраняет его"""
    screenshot_path = os.path.join(output_dir, f"tick_{tick}_{user_id}.jpg")
    
    # Делаем скриншот
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    
    print(f"📸 Скриншот сохранён: {screenshot_path}")
    
    return screenshot_path


def process_ticks(demo_path, csv_path, output_dir=None):
    """
    Основная логика обработки тиков
    
    Args:
        demo_path: путь к демке
        csv_path: путь к CSV файлу с тиками
        output_dir: директория для сохранения скриншотов
    """
    # Создаём директорию для скриншотов
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), 'screenshots')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Скриншоты будут сохранены в: {output_dir}")
    
    # Загружаем тики из CSV
    ticks = pd.read_csv(csv_path)
    print(f"📊 Загружено {len(ticks)} уникальных тиков")
    
    # Инициализируем демку
    setup_demo(demo_path)
    time.sleep(2)
    
    # Итеративно обрабатываем каждый тик
    for i in range(len(ticks)):
        try:
            print(f"\n⏱️  Обработка тика {i+1}/{len(ticks)}:")

            # Переходим на нужный тик
            print(ticks["tick"][i], ticks["user_id"][i])
            enter_console_command(f'demo_gototick {int(ticks["tick"][i])}')
            enter_console_command(f'spec_player {int(ticks["user_id"][i]) + 1}')
            time.sleep(1.5)  # Ждём загрузки кадра
            
            # Делаем скриншот
            take_screenshot(ticks["tick"][i], int(ticks["user_id"][i]) + 1, output_dir)
            
        except KeyboardInterrupt:
            print("\n⚠️  Прервано пользователем")
            break
        except Exception as e:
            print(f"❌ Ошибка при обработке тика {ticks["tick"][i]}: {e}")
            continue
    
    print(f"\n✅ Готово! Все скриншоты сохранены в {output_dir}")


if __name__ == "__main__":
    # Параметры
    demo_file = "1-03f67162-abf3-437e-b575-86538acdb399-1-1.dem"
    csv_file = r"C:\Users\misas\CS2_NN\spotted_alive_players.csv"
    output_directory = r"C:\Users\misas\CS2_NN\detect\screenshots"
    
    # Проверяем наличие CSV файла
    if not os.path.exists(csv_file):
        print(f"❌ CSV файл не найден: {csv_file}")
        exit(1)
    
    print("🚀 Начинаю обработку демки с bounding box'ами")
    print("⏱️  Убедись, что CS2 открыт!")
    time.sleep(3)
    
    process_ticks(demo_file, csv_file, output_directory)
