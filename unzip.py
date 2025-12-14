import os
import subprocess

DEMO_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"
TOOL_PATH = r"C:\Users\misas\Downloads\zstd-v1.5.7-win64\zstd.exe"
DOWNLOAD_DIR = r"C:\Users\misas\Downloads"

DEMOS = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.zst')]

for demo in DEMOS:
    demo_path = os.path.join(DOWNLOAD_DIR, demo)
    demo_res_path = os.path.join(DEMO_DIR, demo[:-4])  # Убираем ".zst"

    print(f"🗜️ Распаковка: {demo_path} → {demo_res_path}")

    # subprocess автоматически экранирует аргументы
    result = subprocess.run(
        [TOOL_PATH, "-d", demo_path, "-o", demo_res_path],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ Успех: {demo}")
    else:
        print(f"❌ Ошибка при распаковке {demo}")
        print("STDERR:", result.stderr)
