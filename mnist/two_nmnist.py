

import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import lava.lib.dl.slayer as slayer

class TWONMNISTDataset(Dataset):
    """TWONMNIST dataset method

    Parameters
    ----------
    path : str, optional
        path of dataset root, by default 'data'
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 1
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default Noney.
    max_digit: int
        max digit in mnist (useful for testing)
    """
    def __init__(
        self, path='data',
        train=True,
        sampling_time=1, sample_length=300,
        transform=None, data_path = "2nmnist/data"
    ):
        super(TWONMNISTDataset, self).__init__()
        self.path = path
        self.train = train
        
        if self.train:
            data_path = "2nmnist/data/Train"
        else:
            data_path = "2nmnist/data/Test"
        self.samples = glob.glob(f'{data_path}/*.bin')
        self.samples = [i.replace("\\", "/") for i in self.samples]
        self.labels = [i.split("_")[-1][:-4].split("-") for i in self.samples]
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length/sampling_time)
        self.transform = transform

    def __getitem__(self, i):
        filename = self.samples[i]
        label = torch.zeros(10)
        for l in self.labels[i]:
            label[int(l)] = 1
        event = slayer.io.read_2d_spikes(filename)
        if self.transform is not None:
            event = self.transform(event)
        spike = event.to_tensor(self.sampling_time)
        padded_spike = np.zeros((2, 34, 34, self.num_time_bins))
        padded_spike[:, :spike.shape[1], :spike.shape[2], :min(spike.shape[-1], self.num_time_bins)]  = spike[:, :, :, :min(spike.shape[-1], self.num_time_bins)]
        return torch.from_numpy(padded_spike).float(), label

    def __len__(self):
        return len(self.samples)
