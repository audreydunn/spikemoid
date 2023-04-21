import os
import numpy as np
import glob
import zipfile
import torch
import lava.lib.dl.slayer as slayer

warning_message = """Warning: This script should not be run since every
version of numpy will generate a different dataset split. 
Rather download the knmnist.zip file. This script is released so the community can 
see how the dataset was created as well as modify it to 
superimpose an arbitrary number of images in a single frame if desired"""

attribution_text = '''
# NMNIST dataset is freely available here:
# https://www.garrickorchard.com/datasets/n-mnist

# (c) Creative Commons:
#     Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
#     "Converting Static Image Datasets to Spiking Neuromorphic Datasets Using
#     Saccades",
#     Frontiers in Neuroscience, vol.9, no.437, Oct. 2015
#             '''.replace(' '*12, '')

def generate_knmnist(k = 2):
    print(warning_message)
    splits = ["Train", "Test"]
    path = "data/"
    if not os.path.isdir(path):
        os.mkdir(path)
    max_digit = 9
    sampling_time = 1
    sample_length = 300
    for split in splits:

        data_path = path +  split
        if split == "Train":
            source = ('https://www.dropbox.com/sh/tg2ljlbmtzygrag/'+
                'AABlMOuR15ugeOxMCX0Pvoxga/Train.zip')
            save_dir = "%dnmnist/data/Train/"% k
        elif split == "Test":
            source = ('https://www.dropbox.com/sh/tg2ljlbmtzygrag/'+
                'AADSKgJ2CjaBWh75HnTNZyhca/Test.zip')
            save_dir = "%dnmnist/data/Test/"% k

        if len(glob.glob(f'{data_path}/')) == 0:  # dataset does not exist
            print(
                f'NMNIST {"training" if split=="Train" else "testing"} '
                'dataset is not available locally.'
            )
            print('Attempting download (This will take a while) ...')
            print(attribution_text)
            os.system(f'wget {source} -P {path}/ -q --show-progress')
            print('Extracting files ...')
            with zipfile.ZipFile(data_path + '.zip') as zip_file:
                for member in zip_file.namelist():
                    zip_file.extract(member, path)
            print('Download complete.')
        os.makedirs(save_dir, exist_ok=True)
        samples = glob.glob(f'{data_path}/*/*.bin')
        samples = [i.replace("\\", "/") for i in samples]
        samples = [i for i in samples if int(i.split("/")[-2]) <= max_digit]
        knmnist_samples = []

        np.random.seed(42)
        np.random.shuffle(samples)
        max_digit = max_digit
        im_idx = 0
        while im_idx < len(samples):
            num_samples = np.random.randint(1, k+1)
    
            if num_samples == 2:  # no duplicate samples
                gt1 = int(int(samples[im_idx].split('/')[-2]))
                try:
                    gt2 = int(int(samples[im_idx+1].split('/')[-2]))
                except Exception as IndexError:
                    num_samples = 1
                else:
                    if gt1 == gt2:
                        num_samples = 1    
            if k > 2:
                raise(Exception("When K == 2 this code is not verified"))
                time_flip = np.random.randint(0, 2, size=num_samples, dtype=bool)
            else:
                time_flip = [0, 1]
            superimposed_sample = time_flip, samples[im_idx: im_idx + num_samples] 
            knmnist_samples.append(superimposed_sample)
            im_idx += num_samples

        for i, sample in enumerate(knmnist_samples):
            label = torch.zeros(max_digit + 1)
            event = None
            label_list = []
            for time_flip, filename in zip(sample[0], sample[1]):  # may iterate one or two times
                detect_labal = int(filename.split('/')[-2])
                label_list.append(str(detect_labal))
                label[detect_labal] = 1
                if event is None:
                    event = slayer.io.read_2d_spikes(filename)
                else:
                    new_event = slayer.io.read_2d_spikes(filename)
                    if time_flip:
                        new_event.t = np.flip(max(new_event.t) - new_event.t)  # just flip the time dimension!
                        new_event.c = np.flip(new_event.c)
                        new_event.x = np.flip(new_event.x)
                        new_event.y = np.flip(new_event.y)

                    event.x = np.concatenate([new_event.x, event.x])
                    event.y = np.concatenate([new_event.y, event.y])
                    event.t = np.concatenate([new_event.t, event.t])
                    event.c = np.concatenate([new_event.c, event.c])

            label_string = "-".join(label_list)
            save_name = "%d_%s.bin" % (i, label_string)
            event = slayer.io.Event(event.x, event.y, event.c, event.t)
            
            save_name = save_dir + save_name
            slayer.io.encode_2d_spikes(save_name, event)

if __name__ == "__main__":
    generate_knmnist(2)