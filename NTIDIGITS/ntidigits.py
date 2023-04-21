from lava.lib.dl.slayer.io import Event
from torch.utils.data import Dataset
from scipy.cluster.vq import kmeans2
from collections import OrderedDict
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as pp
import numpy as np
import scipy
import torch
import sys
import os


def pad_spike_data(spike_data:np.array, pad_timesteps:int, fill_value = 0):
    """
    Adds zeros to the end of spike data so that the number of timesteps matches pad_timesteps

    Args:
        spike_data (np.array): np.array of shape (channels, input_size, timesteps)
        pad_timesteps (int): number of timesteps of the return array

    Returns:
        np.array: array of shape (channels, input_size, pad_timesteps)
    """
    try:
        if pad_timesteps is None:
            return spike_data
        curr_timesteps = spike_data.shape[-1]
        #assert pad_timesteps >= curr_timesteps, "Cannot pad the current spike data since it has more timesteps = %d than pad timesteps %d" % (curr_timesteps, pad_timesteps)
        if len(spike_data.shape) == 3:
            padded_spike_data = np.full((spike_data.shape[0], spike_data.shape[1], pad_timesteps), fill_value)
            padded_spike_data[:, :, :min(curr_timesteps, pad_timesteps)] = spike_data[:, :, :min(curr_timesteps, pad_timesteps)]
        else:
            padded_spike_data = np.full((spike_data.shape[0], pad_timesteps), fill_value)
            padded_spike_data[:, :min(curr_timesteps, pad_timesteps)] = spike_data[:, :min(curr_timesteps, pad_timesteps)]
        return padded_spike_data
    except Exception as e:
        import pdb; pdb.set_trace()
    

def convert_sample_into_events(sample_name, h5py_file, train = True):
    if train:
        timestep_key = "train_timestamps"
        address_key = 'train_addresses'
        cluster_dict_key = "train_" + sample_name.decode()
    else:
        timestep_key = "test_timestamps"
        address_key = 'test_addresses'
        cluster_dict_key = "test_" + sample_name.decode()
    # print(sample_name)

    timestamps = h5py_file[timestep_key][sample_name][:]
    addresses = h5py_file[address_key][sample_name][:]
    mask = (np.abs(timestamps[0] - timestamps) > .1) * (np.abs(timestamps[-1] - timestamps) > .1)  # ensures we don't include artificats

    if not np.any(mask):
        mask = np.array([True] * timestamps.size)
    filt_timesteps = timestamps[mask]
    filt_addresses = addresses[mask]
    ground_truth = sample_name.decode().split("-")[-1]
    for i in range(1, 5):
        try:
            cluster_centers, cluster_decisions = kmeans2(np.expand_dims(filt_timesteps, -1), len(ground_truth),iter = 100, missing = "raise", minit = "++")
        except scipy.cluster.vq.ClusterError as e:
            print("Failed to form cluster for sample")
            pp.Figure()
            pp.plot(timestamps, addresses, ".")
            pp.savefig(sample_name.decode() + ".jpg")

    cluster_dict = OrderedDict()  # get order of clusters
    for i in cluster_decisions:
        if i not in cluster_dict:
            cluster_dict[i] = None
    unique_ordered_clusters = cluster_dict.keys()

    # 1, 1, 64, timesteps
    truth_list = []
    data_list = []
    index = 0
    #pp.Figure()
    # pp.plot(timestamps, addresses, ".")
    # pp.show()
    for cluster_label, digit in zip(unique_ordered_clusters, ground_truth):
        y_event = None
        if digit == 'o':
            digit=10
        if digit == 'z':
            digit=0
        else:
            digit=int(digit)

        cluster_active = np.argwhere(cluster_decisions==cluster_label).flatten()
        start_label_time = filt_timesteps[cluster_active[0]]
        end_label_time = filt_timesteps[cluster_active[-1]]
        digit_time_mask = (timestamps >= start_label_time) * (timestamps <= end_label_time)
        adjusted_timestamp = (timestamps[digit_time_mask] - start_label_time)
        # pp.Figure()
        # pp.plot(adjusted_timestamp,addresses[digit_time_mask], ".")
        # pp.show()
        data_event = Event(addresses[digit_time_mask], y_event, np.zeros(adjusted_timestamp.shape), adjusted_timestamp)
        truth_list.append(digit)
        data_list.append(data_event)
        index += 1

    return data_list, truth_list


def convert_sample_into_event(sample_name, h5py_file, train = True):
    if train:
        timestep_key = "train_timestamps"
        address_key = 'train_addresses'
        cluster_dict_key = "train_" + sample_name.decode()
    else:
        timestep_key = "test_timestamps"
        address_key = 'test_addresses'
        cluster_dict_key = "test_" + sample_name.decode()


    timestamps = h5py_file[timestep_key][sample_name][:]
    addresses = h5py_file[address_key][sample_name][:]
    y_event = None
    c_event = np.zeros(addresses.shape)
    data_event = Event(addresses, y_event, c_event, timestamps)
    digit = sample_name.decode().split("-")[-1]
    if digit == 'o':
        digit=10
    if digit == 'z':
        digit=0
    else:
        digit=int(digit)
    return data_event, digit

def augment_(event, factor=.1, shift=100):

    # # Add between 1 and 5% random spikes
    # add_events = int(event.x.size * np.random.uniform(.01, .05))
    # t_events_add = np.random.uniform(0, max(event.t), add_events)
    # x_events_add = np.random.randint(0, 64, add_events)
    # new_x = np.concatenate([event.x, x_events_add])
    # new_t = np.concatenate([event.t, t_events_add])
    # y_event = np.zeros(new_x.shape)
    # c_event = np.zeros(new_x.shape)
    # event = Event(new_x, y_event, c_event, new_t)
    event.t *= 1 + 2 * (np.random.random() - .5) * factor
    event.t += np.random.random() * shift
    return event


def convert_sample_into_lava_format(sample_name, h5py_file, sample_time = 1000, scale_time = 1e6, train = True):
    if train:
        timestep_key = "train_timestamps"
        address_key = 'train_addresses'
        cluster_dict_key = "train_" + sample_name.decode()
    else:
        timestep_key = "test_timestamps"
        address_key = 'test_addresses'
        cluster_dict_key = "test_" + sample_name.decode()


    timestamps = h5py_file[timestep_key][sample_name][:]
    addresses = h5py_file[address_key][sample_name][:]
    y_event = np.zeros(addresses.shape)
    c_event = np.zeros(addresses.shape)
    data_event = Event(addresses, y_event, c_event, timestamps*scale_time)
    if train:
        data_event = augment_(data_event)

    spike_data = (data_event.to_tensor(sample_time) > 0)
    # 1, 1, 64, timesteps
    label_data = np.zeros((1, 1, 11, spike_data.shape[-1]))
    label_addresses = np.zeros(timestamps.shape)
    y_event = np.zeros(addresses.shape)
    c_event = np.zeros(addresses.shape)
    ground_truth = sample_name.decode().split("-")[-1]

    for digit in ground_truth:
        if digit == 'o':
            digit=10
        if digit == 'z':
            digit=0
        else:
            digit=int(digit)
        label_data[0, 0, digit, :] = 1

    padded_spike_data = np.zeros((1, 1, 64, label_data.shape[-1]))
    padded_spike_data[:, :, :spike_data.shape[2], :] = spike_data
    return torch.from_numpy(padded_spike_data).float().squeeze(0).squeeze(0), torch.from_numpy(label_data).float().squeeze(0).squeeze(0)

def get_smallest_time_sample(dataset):
    """Isolate smallest sample from dataset.

    Parameters
    ----------
    dataset : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    labels = dataset["train_labels"]
    min_time_duration = sys.maxsize
    min_time_duration_sample = None
    for label in labels:
        timestamps = dataset["train_timestamps"][label]

        # Remove artificts from data using mask
        mask = (np.abs(timestamps[0] - timestamps) > .1) * (np.abs(timestamps[-1] - timestamps) > .1) 
        if not np.any(mask):
            mask = np.array([True] * len(timestamps))
        filt_timesteps = timestamps[mask]

        # Determine time duration of sample
        time_length = filt_timesteps[-1] - filt_timesteps[0]
        if time_length < min_time_duration:
            min_time_duration = time_length
            min_time_duration_sample = label
    return label, time_length

class NTIDIGITSDetection(Dataset):
    def __init__(
        self,
        h5py_file,
        scale_timesteps,
        sample_time,
        train=True,
        only_singleton = True,
        pad_timesteps = None,
    ):  
        self.sample_time = sample_time
        self.scale_timesteps = scale_timesteps
        self.h5py_file = h5py_file
        self.train = train
        self.only_singleton = only_singleton
        self.cache_dictionary = {}
        self.pad_timesteps = pad_timesteps
        if self.train:
            labels_key = "train_labels"
        else:
            labels_key = "test_labels"

        # filter out singleton labels
        if self.only_singleton:
             self.labels = [l for l in self.h5py_file[labels_key] if len(l.decode().split("-")[-1]) == 1]
        else:
            self.labels = self.h5py_file[labels_key][:]

    def __getitem__(self, i):
        if self.train:
            sample = self.labels[i]
        else:
            sample = self.labels[i]

        #if self.only_singleton:
        x_train, y_train =  convert_sample_into_lava_format(sample, self.h5py_file, sample_time = self.sample_time, scale_time = self.scale_timesteps, train = self.train)
        x_train = pad_spike_data(x_train, self.pad_timesteps, 0)
        y_train = pad_spike_data(y_train, self.pad_timesteps, 0)
        return x_train, y_train

    def __len__(self):
        return len(self.labels)

class NTIDIGITSAugmented(Dataset):
    def __init__(
        self,
        scale_timesteps,
        sample_time,
        train=True,
        pad_timesteps = None,
    ):  
        self.sample_time = sample_time
        self.scale_timesteps = scale_timesteps
        self.train = train
        self.pad_timesteps = pad_timesteps
        if self.train:
            base_path = "augmented_data/Train/"
        else:
            base_path = "augmented_data/Test/"
        self.file_names = [base_path + i for i in os.listdir(base_path)]

    def __getitem__(self, i):
        file_name = self.file_names[i]
        label = int(file_name[:-4].split("_")[-1])
        event = slayer.io.read_1d_spikes(file_name)
        event.t *= self.scale_timesteps
        
        x_train = event.to_tensor(self.sample_time)[0]

        label_padded_x_train = np.zeros((64, x_train.shape[-1]))
        label_padded_x_train[:x_train.shape[0], :] = x_train
        x_train = pad_spike_data(label_padded_x_train, self.pad_timesteps, 0)
    
        return x_train, label

    def __len__(self):
        return len(self.file_names)

def process_n_tidigits(h5py_file, train = True):
    if train:
        labels_key = "train_labels"
    else:
        labels_key = "test_labels"
    labels = h5py_file[labels_key][:]
    event_list = []
    gt_list = []
    for label in labels:
        if len(label.decode().split("-")[-1]) == 1:
            event, label = convert_sample_into_event(label, h5py_file, train)
            event_list.append(event)
            gt_list.append(label)
        elif train:
            events, labels = convert_sample_into_events(label, h5py_file, train)
            event_list.extend(events)
            gt_list.extend(labels)
    index = 0
    for event, gt in zip(event_list, gt_list):
        if train:
            save_path = "augmented_data/Train/%d_%d.bin" % (index, gt)
        else:
            save_path = "augmented_data/Test/%d_%d.bin" % (index, gt)
        index += 1
        slayer.io.encode_1d_spikes(save_path, event)



if __name__ == "__main__":
    scale_timesteps = 1e6  # scale timesteps by this value
    sample_time = 1000
    dataset = NTIDIGITSAugmented(1e3, 1, pad_timesteps=1000)
    for x, y in dataset:
        print(x.shape)
