# Overview
This is the public codebase for the Spikemoid publication.

# Setup Instructions 
1. Clone this repo and cd into it
2. Clone the [lava-dl fork](https://github.com/audreydunn/lava-dl/tree/spikemoid_public) and cd into it
3. checkout the 'spikemoid_public' branch.
4. Reinstall lava-dl. You can use the default poetry install in the lava-dl-fork Readme 
5. If you get poetry not recognized error on windows run `set PATH=%PATH%;%USERPROFILE%\AppData\Roaming\pypoetry\venv\Scripts`
6. Install cuda pytorch version 1.12.1 using [this guide](https://pytorch.org/get-started/previous-versions/)
7. cd ..
8. pip install pandas 
9. pip install tonic
10. Download [2nmnist.zip](https://doi.org/10.5281/zenodo.7847750) and extract in the mnist folder
11. Download [n-tidigits.hdf5](https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=0) and place it in the NTIDIGITS folder. Further information about the publicly available dataset can be found [here](https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit).

Model Weights from the paper available for download [here](https://doi.org/10.5281/zenodo.7854046)

# NMNIST (TABLE 1)
1. `cd mnist`
2. `python train.py --count_scale 1 --global_alpha_theta --epochs 25` <-- spikemax
3. `python train.py --count_scale 2 --global_alpha_theta --epochs 25`
4. `python train.py --count_scale 4 --global_alpha_theta --epochs 25`
5. `python train.py --count_scale 8 --global_alpha_theta --epochs 25`

# Gestures (Table 1)
1. `cd gestures`
2. `python train.py --counts --count_scale 1 --global_alpha_theta --epochs 100` <-- spikemax
3. `python train.py --counts --count_scale 2 --global_alpha_theta --epochs 100`
4. `python train.py --counts --count_scale 4 --global_alpha_theta --epochs 100`
5. `python train.py --counts --count_scale 8 --global_alpha_theta --epochs 100`

# NTIDIGITS (Table 1)
1. `cd NTIDIGITS`
2. `python classification_train.py --counts --count_scale 1 --epochs 800` <-- spikemax
3. `python classification_train.py --counts --count_scale 2 --epochs 800`
4. `python classification_train.py --counts --count_scale 4 --epochs 800`
5. `python classification_train.py --counts --count_scale 8 --epochs 800`

# IV. Experiments - B. 2NMNIST - Spikemoid
1. `cd mnist`
2. `python train_two_nmnist.py --pretrain --global_alpha_theta --epochs 50`
3. `python train_two_nmnist.py --pretrain --epochs 50`

#  IV. Experiments - C. N-TIDIGITS18 Spikemoid
1. `cd NTIDIGITS`
2. `python classification_train.py --stop_updating_alpha_theta 10 --epochs 3000 --global_alpha_theta` <- Train Scaled-Spikemax model to convergence
3. `python detection_train.py --pretrain --pretrain_folder bamsumit_params0_global` Note: --pretrain requires the --pretrain_folder argument to be set to the output files of your classification_train.py run
4. `python detection_train.py --pretrain --global_alpha_theta --pretrain_folder bamsumit_params0_global`


# Spikemoid Model (Table II)
1. `cd NTIDIGITS`
2. `python classification_train.py --epochs 15 --pretrain --stop_updating_alpha_theta 5`  <-- fine tune detection model on classification task 

# Generate 2NMNIST dataset
The global version of this dataset is available [here](https://doi.org/10.5281/zenodo.7847750).
The code to generate the dataset can be found here: 
Regenerating the dataset on different machines will yield different mixtures of digits, 
so it is not recommended to actually run this. However, for reproducibility concerns we release the code here. 

# Additional Info
The implementation of spikemoid and scaled spikemax can be found in this [file](https://github.com/audreydunn/lava-dl/blob/spikemoid_public/src/lava/lib/dl/slayer/loss.py). We plan to soon contact the lava team about incorporating spikemoid into their main repository

This research was conducted at the Georgia Institute of Technology