import os
import sys

import numpy as np
from ultralytics import YOLO

JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
CONF_MIN = 0.3


def main(session_dir: str):
    frame_ts = np.load(os.path.join(session_dir, 'frame_ts.npy'))
    video_path = os.path.join(session_dir, 'video.mp4')

    model = YOLO('yolov8s-pose.pt')
    kpts = []

    for res in model(video_path, stream=True, verbose=False):
        if res.keypoints is None or len(res.keypoints) == 0 or res.keypoints.conf is None:
            kpts.append(np.zeros((len(JOINTS), 3), dtype=np.float32))
            continue
        boxes_conf = res.boxes.conf.cpu().numpy()
        i = int(boxes_conf.argmax())
        xyn = res.keypoints.xyn[i].cpu().numpy()
        conf = res.keypoints.conf[i].cpu().numpy()
        k = np.concatenate([xyn[JOINTS], conf[JOINTS, None]], axis=1).astype(np.float32)
        k[k[:, 2] < CONF_MIN] = 0.0
        kpts.append(k)

        if len(kpts) % 300 == 0:
            print(f"{len(kpts)} frames...")

    kpts = np.stack(kpts)
    m = min(len(kpts), len(frame_ts))
    np.savez_compressed(os.path.join(session_dir, 'keypoints.npz'),
                        frame_ts=frame_ts[:m], kpts=kpts[:m])
    detected = (kpts[:m, :, 2] > CONF_MIN).any(axis=1).mean()
    print(f"saved keypoints.npz: {m} frames, person detected in {detected:.0%}")


if __name__ == '__main__':
    main(sys.argv[1])
