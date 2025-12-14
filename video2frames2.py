import subprocess


cmd = [
    r"C:\Users\misas\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
    "-hwaccel",
    "cuda",
    "-hwaccel_output_format",
    "cuda",
    "-y",
    "-i",
    r"C:\Users\misas\Videos\2025-12-14 16-25-44.mkv",
    "-vf",
    "fps=1",
    "-q:v",
    "2",
    r"frames\match2\frame_%07d.jpg",
]
subprocess.run(cmd, check=True)