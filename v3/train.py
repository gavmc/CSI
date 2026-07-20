from dataset import CSIDataset
from model import SimpleLoss, DETR

import torch
from torch.utils.data import DataLoader, ConcatDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SESSIONS = 5
EPOCHS = 3


def get_loader():
    datasets = [CSIDataset(f"session_{session_num:03d}.h5") for session_num in range(SESSIONS)]
    dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return loader



def train(model, loss_fn, optimizer):
    model.train()
    loader = get_loader()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for csi, target in loader:
            csi = csi.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(csi)
            loss = loss_fn(pred, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(loader)
        print(
            f"Epoch: {epoch} | "
            f"Loss: {running_loss:.4f}"
        )

        
if __name__ == "__main__":

    model = DETR().to(device)

    loss_fn = SimpleLoss(
        pos_weight=1,
        conf_weight=1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 1e-4,
        weight_decay=1e-4
    )

    train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer
    )

    torch.save(model.state_dict(), "DETR_weights.pth")
