import io
import time

import serial

try:
    import winsound
    BEEP = lambda: winsound.Beep(1000, 300)
except ImportError:
    BEEP = lambda: print('\a', end='', flush=True)

PORT = "COM15"
BAUD = 921600
OUT = "csi_data.csv"

PHASES = [
    ("empty",     30, "Sit still, hands away from ESP32. Establishing baseline."),
    ("walking",   30, "Walk around between the ESP32 and your laptop."),
    ("waving",    30, "Wave your hand vigorously between ESP32 and laptop."),
    ("breathing", 60, "Sit still and breathe normally. Don't move otherwise."),
]

ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 1
ser.dtr = False
ser.rts = False
ser.open()
sio = io.TextIOWrapper(io.BufferedRWPair(ser, ser), encoding='utf-8', errors='ignore')

print(f"Connected to {PORT} at {BAUD} baud")
print("Waiting for CSI data...\n")

header_line = None
deadline = time.time() + 30
lines_shown = 0
csi_seen = False

while time.time() < deadline:
    raw = sio.readline().strip()
    if not raw:
        continue
    if lines_shown < 20:
        print(f"  | {raw[:100]}")
        lines_shown += 1
    if raw.startswith("type,"):
        header_line = raw
    elif raw.startswith("CSI_DATA,"):
        csi_seen = True
        break

if not csi_seen:
    print("\nERROR: No CSI data after 30s. Check what the ESP32 is doing.")
    ser.close()
    exit(1)

print("\nCSI data flowing. Starting collection.\n")
total = sum(p[1] for p in PHASES)
print(f"Total duration: {total}s ({total // 60}m{total % 60:02d}s)")
print(f"Plan ({len(PHASES)} phases):")
for i, (lbl, d, inst) in enumerate(PHASES, 1):
    print(f"  {i}. [{lbl}] {d}s \u2014 {inst}")
print()
input("Press Enter when you're in position to start...\n")

count = 0
phase_idx = 0
phase_start = time.time()
label, duration, instruction = PHASES[0]

print(f">>> Phase 1/{len(PHASES)}: [{label}]")
print(f">>> {instruction}\n")
BEEP()

try:
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        if header_line:
            f.write(header_line + ",label\n")
        else:
            f.write("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,"
                    "aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,"
                    "secondary_channel,local_timestamp,ant,sig_len,rx_format,len,"
                    "first_word,data,label\n")

        while True:
            now = time.time()
            if now - phase_start >= duration:
                phase_idx += 1
                if phase_idx >= len(PHASES):
                    break
                label, duration, instruction = PHASES[phase_idx]
                phase_start = now
                _phase_start_count = count
                print(f"\n\n>>> Phase {phase_idx + 1}/{len(PHASES)}: [{label}]")
                print(f">>> {instruction}\n")
                BEEP()

            line = sio.readline().strip()
            if not line:
                continue
            if line.startswith("CSI_DATA,"):
                bracket_open = line.find('[')
                bracket_close = line.rfind(']')
                if bracket_open > 0 and bracket_close > bracket_open:
                    before_data = line[:bracket_open]
                    if before_data.count(',') == 24:
                        f.write(f"{line},{label}\n")
                        count += 1
                        if not hasattr(f, '_last_update'):
                            f._last_update = now
                        if now - f._last_update >= 0.5:
                            remaining = duration - (now - phase_start)
                            rows_this_phase = count - _phase_start_count
                            rate = rows_this_phase / max(0.1, now - phase_start)
                            print(f"\r[{label}] {remaining:5.1f}s left | rows: {count} "
                                  f"| rate: {rate:.1f}/s   ",
                                  end="", flush=True)
                            f._last_update = now

    BEEP()
    time.sleep(0.1)
    BEEP()
    print(f"\n\nDone! Captured {count} rows to {OUT}")

except KeyboardInterrupt:
    print(f"\n\nStopped early. Saved {count} rows to {OUT}")
finally:
    try:
        sio.close()
    except Exception:
        pass
    ser.close()
