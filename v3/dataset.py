import h5py
import torch
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    def __init__(self, file_path, window_length = 50, stride = 10):

        self.file_path = file_path
        self.window_length = window_length
        self.stride = stride

        with h5py.File(file_path, "r") as f:
            num_packets = len(f["csi/amplitude"])

        self.num_windows = ((num_packets - window_length) // stride) + 1

        self.h5_file = None
        
    def _load_data(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, "r")

    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, index):
        self._load_data()

        start = index * self.stride
        end = start + self.window_length
        csi = self.h5_file["csi/amplitude"][start:end]

        pose_index = end - 1
        pose = self.h5_file["pose/coords"][pose_index]
        confidence = self.h5_file["pose/confidence"][pose_index]

        csi = torch.from_numpy(csi).float()
        pose = torch.from_numpy(pose).float()
        confidence = torch.from_numpy(confidence).float()

        target = torch.cat([pose, confidence.unsqueeze(1)], dim=-1)

        return csi, target

    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

