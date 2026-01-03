import obsws_python as obs
from demoparser2 import DemoParser
import subprocess
import time
import os
import pyautogui
import threading
from tqdm import tqdm


def spam_f1():
    while True:
        pyautogui.press('f1')
        time.sleep(1)

def start_both():
    t1 = threading.Thread(target=lambda: pyautogui.press('f5'))
    t2 = threading.Thread(target=client.start_record)
    t1.start(); t2.start()
    t1.join(); t2.join()

OBS_PATH = r"C:\Program Files\obs-studio\bin\64bit\obs64.exe"
OBS_DIR = os.path.dirname(OBS_PATH)
DEMO_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"
DEMOS = [f for f in os.listdir(DEMO_DIR) if f.endswith('.dem')]
RECORD_DIR = r"D:\RecordDemos"
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 4455
WEBSOCKET_PASSWORD = "secret"
STEAM_ID = 76561198386265483

subprocess.Popen(
    [OBS_PATH, "--minimize-to-tray"],
    cwd=OBS_DIR
)
print("🚀 OBS запущен, ждём инициализации...")
time.sleep(7)
print(len(DEMOS))
for i, demo in tqdm(enumerate(DEMOS)):
    try:
        if i <= 6:
            continue
        demo_path = os.path.join(DEMO_DIR, demo)
        parser = DemoParser(demo_path)
        start_df = parser.parse_event('round_prestart')
        round_start_ticks = list(start_df['tick'][3:])
        ticks_df = parser.parse_ticks(['user_id', 'steamid'])
        max_tick = int(ticks_df.iloc[-1]['tick'])
        if len(round_start_ticks) >= 13:
            tmp_df = ticks_df[ticks_df['tick'] == 300]
            user_id = int(tmp_df[tmp_df['steamid'] == STEAM_ID]['user_id']) + 1
            pyautogui.press("`")
            time.sleep(0.5)
            pyautogui.typewrite(f'playdemo {demo}\n')
            time.sleep(15)
            pyautogui.press("`")
            time.sleep(0.5)
            pyautogui.press("`")
            time.sleep(0.5)

            x, y = pyautogui.position()
            pyautogui.moveTo(x, y - 1000)
            time.sleep(0.5)

            pyautogui.typewrite(f'demoui\n')
            time.sleep(0.5)

            pyautogui.typewrite('demo_timescale 4\n')
            time.sleep(0.5)

            pyautogui.typewrite(f'bind "F1" "spec_player {user_id}"\n')
            time.sleep(0.5)

            threading.Thread(target=spam_f1, daemon=True).start()

            pyautogui.typewrite('spec_show_xray 0\n')
            time.sleep(1)
            pyautogui.press("`")
            time.sleep(0.5)

            client = obs.ReqClient(host=WEBSOCKET_HOST, port=WEBSOCKET_PORT, password=WEBSOCKET_PASSWORD)
            save_dir = os.path.join(RECORD_DIR, demo[:-4])
            os.makedirs(save_dir, exist_ok=True)
            time.sleep(1)
            client.set_record_directory(save_dir)
            print("🔗 Подключено к OBS WebSocket")
            for i, start_tick in enumerate(round_start_ticks):
                if i == len(round_start_ticks) - 1:
                    round_dur = int(max_tick - start_tick) // 256 - 25
                    if round_dur < 0: round_dur += 25
                else:
                    round_dur = int(round_start_ticks[i + 1] - start_tick) // 256
                pyautogui.press("`")
                time.sleep(0.5)

                pyautogui.typewrite('demo_pause\n')
                time.sleep(0.5)

                pyautogui.typewrite(f'demo_gototick {start_tick}\n')
                time.sleep(0.5)

                pyautogui.press("`")
                time.sleep(0.5)
                
                client.start_record()
                time_st = time.time()
                time.sleep(0.5)
                pyautogui.press('f5')
                # print(i, time.time() - time_st)

                time.sleep(round_dur)

                client.stop_record()
                print("✅ Запись завершена.")
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"Произошла ошибка: {e}")
