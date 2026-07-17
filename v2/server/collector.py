import argparse
import os
import socket
import struct
import threading
import time
from collections import defaultdict, deque

import numpy as np

CSI_PORT = 5566
PING_PORT = 4444
HDR_FMT = '<HBBIqbBH'
HDR_SIZE = struct.calcsize(HDR_FMT)
PING_FMT = '<HBBIq'
PING_SIZE = struct.calcsize(PING_FMT)
CSI_MAGIC = 0xC51D
PING_MAGIC = 0xB117
LLTF_BYTES = 128

stop_flag = threading.Event()

offsets = defaultdict(lambda: deque(maxlen=200))
store = defaultdict(lambda: {'t': [], 'csi': [], 'rssi': [], 'seq': []})
store_lock = threading.Lock()


def clock_offset(sensor_id: int) -> float | None:
    d = offsets[sensor_id]
    return float(np.median(d)) if len(d) >= 5 else None


def echo_thread():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", PING_PORT))
    s.settimeout(0.5)
    while not stop_flag.is_set():
        try:
            data, addr = s.recvfrom(256)
        except socket.timeout:
            continue
        recv_t = time.time()
        s.sendto(data, addr)
        if len(data) >= PING_SIZE:
            magic, sensor_id, _, seq, esp_us = struct.unpack(PING_FMT, data[:PING_SIZE])
            if magic == PING_MAGIC:
                offsets[sensor_id].append(recv_t - esp_us * 1e-6)


def csi_thread():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", CSI_PORT))
    s.settimeout(0.5)
    counts = defaultdict(int)
    last_report = time.time()
    while not stop_flag.is_set():
        try:
            data, _ = s.recvfrom(1024)
        except socket.timeout:
            continue
        if len(data) < HDR_SIZE:
            continue
        magic, sensor_id, chan, seq, esp_us, rssi, flags, length = \
            struct.unpack(HDR_FMT, data[:HDR_SIZE])
        if magic != CSI_MAGIC:
            continue
        payload = data[HDR_SIZE:HDR_SIZE + length]
        if len(payload) < LLTF_BYTES:
            continue
        off = clock_offset(sensor_id)
        if off is None:
            continue
        t = esp_us * 1e-6 + off
        with store_lock:
            st = store[sensor_id]
            st['t'].append(t)
            st['csi'].append(np.frombuffer(payload[:LLTF_BYTES], dtype=np.int8).copy())
            st['rssi'].append(rssi)
            st['seq'].append(seq)
        counts[sensor_id] += 1

        now = time.time()
        if now - last_report > 5.0:
            rates = {sid: c / (now - last_report) for sid, c in sorted(counts.items())}
            print("CSI Hz:", {k: f"{v:.1f}" for k, v in rates.items()})
            counts.clear()
            last_report = now


def cam_thread(outdir: str):
    import cv2
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    writer = None
    frame_ts = []
    while not stop_flag.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        frame_ts.append(time.time())
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(os.path.join(outdir, 'video.mp4'),
                                     fourcc, 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer:
        writer.release()
    cap.release()
    np.save(os.path.join(outdir, 'frame_ts.npy'), np.array(frame_ts))
    print(f"camera: {len(frame_ts)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--no-cam', action='store_true')
    args = ap.parse_args()

    outdir = os.path.join('sessions', args.session)
    os.makedirs(outdir, exist_ok=True)

    threads = [threading.Thread(target=echo_thread, daemon=True),
               threading.Thread(target=csi_thread, daemon=True)]
    if not args.no_cam:
        threads.append(threading.Thread(target=cam_thread, args=(outdir,), daemon=True))
    for t in threads:
        t.start()

    print("recording... Ctrl+C to stop")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_flag.set()
        for t in threads:
            t.join(timeout=3)

    arrays = {}
    with store_lock:
        for sid, st in store.items():
            arrays[f't_{sid}'] = np.array(st['t'])
            arrays[f'csi_{sid}'] = np.stack(st['csi'])
            arrays[f'rssi_{sid}'] = np.array(st['rssi'], dtype=np.int8)
            arrays[f'seq_{sid}'] = np.array(st['seq'], dtype=np.uint32)
            n = len(st['seq'])
            gaps = int(st['seq'][-1] - st['seq'][0] + 1 - n) if n else 0
            print(f"sensor {sid}: {n} frames, {gaps} lost in air/queue")
    np.savez_compressed(os.path.join(outdir, 'csi.npz'), **arrays)
    print(f"saved {outdir}/csi.npz")


if __name__ == '__main__':
    main()
