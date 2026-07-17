import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from csi_pose_transformer import CSIPoseTransformer, MaskedCSIPretrainer, keypoint_loss


def pck(pred_xy, target_xy, vis, thresh=0.1):
    d = torch.linalg.norm(pred_xy - target_xy, dim=-1)
    return ((d < thresh) & vis).sum().item() / max(vis.sum().item(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--pretrain', action='store_true')
    ap.add_argument('--init', default=None)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = np.load(args.data)
    X = torch.from_numpy(d['X'])
    model = CSIPoseTransformer().to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev), strict=False)
        print(f"initialized encoder from {args.init}")

    if args.pretrain:
        net = MaskedCSIPretrainer(model).to(dev)
        ds = TensorDataset(X)
        out_path = args.out or 'pretrained.pt'
    else:
        net = model
        ds = TensorDataset(X, torch.from_numpy(d['Y']), torch.from_numpy(d['V']))
        out_path = args.out or 'model.pt'

    n_val = max(1, int(0.15 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.bs)
    print(f"train {len(tr)}  val {len(va)}  device {dev}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = float('inf')

    for ep in range(args.epochs):
        net.train()
        tr_loss = 0.0
        for batch in tl:
            batch = [b.to(dev) for b in batch]
            if args.pretrain:
                loss = net(batch[0])
            else:
                pred = net(batch[0])
                loss = keypoint_loss(pred, batch[1], batch[2])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(batch[0])
        sched.step()
        tr_loss /= len(tr)

        net.eval()
        va_loss, va_pck, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in vl:
                batch = [b.to(dev) for b in batch]
                if args.pretrain:
                    loss = net(batch[0])
                else:
                    pred = net(batch[0])
                    loss = keypoint_loss(pred, batch[1], batch[2])
                    va_pck += pck(pred[..., :2], batch[1], batch[2]) * len(batch[0])
                va_loss += loss.item() * len(batch[0])
                n += len(batch[0])
        va_loss /= n
        msg = f"ep {ep:3d}  train {tr_loss:.4f}  val {va_loss:.4f}"
        if not args.pretrain:
            msg += f"  PCK@0.1 {va_pck / n:.3f}"
        print(msg)

        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), out_path)
    print(f"best val {best:.4f} -> {out_path}")


if __name__ == '__main__':
    main()
