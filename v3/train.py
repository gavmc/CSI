import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import h5py

from model import CSI_DETR, Simple_Loss



EPOCHS = 20
BATCH_SIZE = 24
WINDOW_LEN = 50



def load_data(path):

    with h5py.File(path, "r") as f:

        raw_csi = torch.from_numpy(f["CSI"][...]).float()
        pos_data = torch.from_numpy(f["POS"][...]).float()[WINDOW_LEN-1:]

        csi_data = raw_csi.unfold(dimension=0, size=WINDOW_LEN, step=1).permute(0, 3, 1, 2)

        print(csi_data.shape)
        print(pos_data.shape)

        dataset = TensorDataset(csi_data, pos_data)

        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(model, loss_fn, optimizer, data_loader, epochs):

    model.train()

    for epoch in range(epochs):
        running_loss = 0

        for x, y in data_loader:

            optimizer.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch} | Loss {running_loss / len(data_loader)}")



if __name__ == "__main__":
    model = CSI_DETR()
    loss_fn = Simple_Loss(5, 1)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    data_loader = load_data("./csi_data.h5")
    train(model, loss_fn, optimizer, data_loader, EPOCHS)

    torch.save(model.state_dict(), "csi_detr.pth")
