import io
import json
import time
from collections import deque

import numpy as np
import joblib
import serial

PORT = "COM15"
BAUD = 921600
MODEL = "csi_model.joblib"
PREDICT_EVERY = 25


print("Loading model...")
bundle = joblib.load(MODEL)
clf = bundle['classifier']
scaler = bundle['scaler']
active = bundle['active_mask']
WINDOW = bundle['window']
print(f"Classes: {list(clf.classes_)}")
print(f"Window size: {WINDOW} packets\n")

ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 1
ser.dtr = False
ser.rts = False
ser.open()
sio = io.TextIOWrapper(io.BufferedRWPair(ser, ser), encoding='utf-8', errors='ignore')
print(f"Connected to {PORT}. Filling buffer...")


def parse_amp(line):
    if not line.startswith("CSI_DATA,"):
        return None
    bo = line.find('[')
    bc = line.rfind(']')
    if bo <= 0 or bc <= bo:
        return None
    fields = line[:bo].split(',')
    if len(fields) < 24:
        return None
    try:
        declared_len = int(fields[22])
        vals = np.array(json.loads(line[bo:bc + 1]), dtype=np.int8)
    except (ValueError, json.JSONDecodeError):
        return None
    if len(vals) >= declared_len:
        vals = vals[:declared_len]
    else:
        vals = np.concatenate([vals, np.zeros(declared_len - len(vals), dtype=np.int8)])
    vals = vals[4:]
    if len(vals) % 2 == 1:
        vals = vals[:-1]
    iq = vals.astype(np.float32)
    return np.abs(iq[0::2] + 1j * iq[1::2])


buffer = deque(maxlen=WINDOW)
packets_since_predict = 0
n_predictions = 0
last_pred_time = time.time()
history = deque(maxlen=40)

try:
    while True:
        raw = sio.readline().strip()
        if not raw:
            continue
        amp = parse_amp(raw)
        if amp is None or len(amp) < len(active):
            continue

        amp_active = amp[:len(active)][active]
        buffer.append(amp_active)
        packets_since_predict += 1

        if len(buffer) == WINDOW and packets_since_predict >= PREDICT_EVERY:
            packets_since_predict = 0
            win = np.stack(buffer)
            feat = np.concatenate([
                win.mean(axis=0),
                win.std(axis=0),
                win.max(axis=0) - win.min(axis=0),
            ]).reshape(1, -1)

            probs = clf.predict_proba(scaler.transform(feat))[0]
            pred_idx = int(np.argmax(probs))
            pred = clf.classes_[pred_idx]
            confidence = probs[pred_idx]

            now = time.time()
            pred_rate = 1.0 / max(0.001, now - last_pred_time)
            last_pred_time = now
            n_predictions += 1
            history.append(pred[0].upper())

            symbols = {
                'empty':     ('\u2014', '\033[90m'),
                'breathing': ('\u00b7', '\033[92m'),
                'walking':   ('W', '\033[91m'),
                'waving':    ('w', '\033[93m'),
            }
            sym, color = symbols.get(pred, ('?', '\033[0m'))
            reset = '\033[0m'

            bar_width = 30
            filled = int(bar_width * confidence)
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

            tape = "".join(history)
            probs_str = " ".join(f"{c[:4]}:{p:.0%}" for c, p in zip(clf.classes_, probs))

            print(f"\r{color}{pred:>10s}{reset} [{bar}] {confidence:5.1%}  "
                  f"{probs_str}  | tape: {tape}  ",
                  end="", flush=True)

except KeyboardInterrupt:
    print(f"\n\nStopped. Made {n_predictions} predictions.")
finally:
    try:
        sio.close()
    except Exception:
        pass
    ser.close()
