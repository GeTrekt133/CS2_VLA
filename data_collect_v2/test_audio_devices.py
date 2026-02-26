"""Quick test: which audio device actually captures system sound?
Play any music/video through speakers before running this.
"""
import sounddevice as sd
import numpy as np

DURATION = 3  # seconds
CANDIDATES = [16, 18, 28, 31, 37]

print(f"Recording {DURATION}s from each candidate device...")
print("Make sure some audio is playing (YouTube, music, etc.)\n")

for device_id in CANDIDATES:
    try:
        dev = sd.query_devices(device_id)
    except Exception:
        continue

    inp_ch = dev['max_input_channels']
    out_ch = dev['max_output_channels']
    sr = int(dev['default_samplerate'])
    is_output_only = out_ch > 0 and inp_ch == 0
    ch = min(out_ch if is_output_only else inp_ch, 2)

    if ch == 0:
        print(f"[{device_id:2d}] {dev['name']}: SKIP (0 usable channels)")
        continue

    try:
        kwargs = dict(
            samplerate=sr,
            channels=ch,
            device=device_id,
            dtype='float32',
        )
        if is_output_only:
            # WASAPI loopback strategies
            strategies = []
            try:
                strategies.append(("auto", sd.WasapiSettings(exclusive=False, auto_convert=True)))
            except TypeError:
                pass
            strategies.append(("shared", sd.WasapiSettings(exclusive=False)))
            strategies.append(("exclusive", sd.WasapiSettings(exclusive=True)))

            opened = False
            for mode, settings in strategies:
                try:
                    recording = sd.rec(
                        int(DURATION * sr), extra_settings=settings, **kwargs
                    )
                    sd.wait()
                    opened = True
                    break
                except Exception:
                    continue

            if not opened:
                print(f"[{device_id:2d}] {dev['name']}: WASAPI FAILED (all modes)")
                continue
        else:
            recording = sd.rec(int(DURATION * sr), **kwargs)
            sd.wait()

        # Analyze
        if recording.ndim > 1:
            mono = recording.mean(axis=1)
        else:
            mono = recording
        peak = np.abs(mono).max()
        rms = np.sqrt(np.mean(mono ** 2))

        if peak > 0.01:
            status = "** HAS AUDIO **"
        elif peak > 0.001:
            status = "very quiet"
        else:
            status = "SILENT"

        print(f"[{device_id:2d}] {dev['name'][:50]}: peak={peak:.4f} rms={rms:.6f} {status}")

    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"[{device_id:2d}] {dev['name'][:50]}: ERROR - {err}")

print("\nUse --audio-device <ID> with the device that shows '** HAS AUDIO **'")
