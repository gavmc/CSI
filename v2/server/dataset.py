import argparse
import os

import numpy as np

SENSORS = [1, 2, 3, 4]
GRID_HZ = 50
WINDOW = 50
STRIDE = 10
LLTF_COMPLEX = 64
VALID_SC = np.r_[1:27, 38:64]


def amplitude(csi_int8: np.ndarray) -> np.ndarray:
    c = csi_int8.astype(np.float32).reshape(-1, LLTF_COMPLEX, 2)
    amp = np.sqrt(c[..., 0] ** 2 + c[..., 1] ** 2)
    return amp[:, VALID_SC]


def load_session(session_dir: str):
    d = np.load(os.path.join(session_dir, 'csi.npz'))
    streams = {}
    for sid in SENSORS:
        if f't_{sid}' not in d:
            raise SystemExit(f"{session_dir}: sensor {sid} missing from recording")
        t = d[f't_{sid}']
        order = np.argsort(t)
        streams[sid] = (t[order], amplitude(d[f'csi_{sid}'][order]))
    return streams


def build_windows(streams, kp=None):
    t0 = max(s[0][0] for s in streams.values())
    t1 = min(s[0][-1] for s in streams.values())
    if kp is not None:
        t0 = max(t0, kp['frame_ts'][0])
        t1 = min(t1, kp['frame_ts'][-1])
    grid = np.arange(t0, t1, 1.0 / GRID_HZ)
    if len(grid) < WINDOW:
        return None

    MIN_DENSITY = 0.6
    starts = np.arange(0, len(grid) - WINDOW, STRIDE)
    keep = []
    for st in starts:
        w0, w1 = grid[st], grid[st + WINDOW - 1]
        ok = all(((t >= w0) & (t <= w1)).sum() >= MIN_DENSITY * WINDOW
                 for t, _ in streams.values())
        keep.append(ok)
    starts = starts[np.array(keep)]
    if len(starts) == 0:
        return None

    R = np.empty((len(SENSORS), len(grid), len(VALID_SC)), dtype=np.float32)
    for i, sid in enumerate(SENSORS):
        t, amp = streams[sid]
        for s in range(amp.shape[1]):
            R[i, :, s] = np.interp(grid, t, amp[:, s])

    starts = np.arange(0, len(grid) - WINDOW, STRIDE)
    X = np.stack([R[:, st:st + WINDOW] for st in starts])

    if kp is None:
        return X, None, None

    ends = grid[starts + WINDOW - 1]
    ft, kpts = kp['frame_ts'], kp['kpts']
    Y = np.empty((len(ends), kpts.shape[1], 2), dtype=np.float32)
    V = np.empty((len(ends), kpts.shape[1]), dtype=bool)
    for j in range(kpts.shape[1]):
        Y[:, j, 0] = np.interp(ends, ft, kpts[:, j, 0])
        Y[:, j, 1] = np.interp(ends, ft, kpts[:, j, 1])
        V[:, j] = np.interp(ends, ft, kpts[:, j, 2]) > 0.5
    return X, Y, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('-o', '--out', default='dataset.npz')
    ap.add_argument('--unlabeled', action='store_true')
    args = ap.parse_args()

    Xs, Ys, Vs = [], [], []
    for sd in args.sessions:
        streams = load_session(sd)
        kp = None
        if not args.unlabeled:
            kp = np.load(os.path.join(sd, 'keypoints.npz'))
        out = build_windows(streams, kp)
        if out is None:
            print(f"{sd}: too short, skipped")
            continue
        X, Y, V = out
        Xs.append(X)
        if Y is not None:
            Ys.append(Y)
            Vs.append(V)
        print(f"{sd}: {len(X)} windows")

    X = np.concatenate(Xs)
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X = (X - mean) / std

    save = {'X': X, 'mean': mean, 'std': std}
    if Ys:
        save['Y'] = np.concatenate(Ys)
        save['V'] = np.concatenate(Vs)
    np.savez_compressed(args.out, **save)
    print(f"saved {args.out}: X{X.shape}" + (f" Y{save['Y'].shape}" if Ys else " (unlabeled)"))


if __name__ == '__main__':
    main()
