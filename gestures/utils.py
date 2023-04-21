import torchvision
import tonic
import torch
"The purpose of this script is to show transformations functioning in tonic"
def apply_heavy_augmentations(train = True):
    to_torch = lambda x: torch.from_numpy(x)
    if train:
        rc_transform = tonic.transforms.RandomCrop(
            sensor_size=(128, 128, 2), target_size=(64, 64)
        )
        frame_transform = tonic.transforms.ToFrame(sensor_size=(64, 64, 2), n_time_bins=1500)
        # TODO DO NOT KEEP THIS SMALL
        rotate_transform = lambda x: torchvision.transforms.functional.rotate(x, 180)

        transform = tonic.transforms.Compose([rc_transform,
                                            frame_transform,
                                            to_torch,
                                            rotate_transform
                                            ])

    else:

        downsample_transform = tonic.transforms.Downsample(spatial_factor=0.25)
        frame_transform = tonic.transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=1500)

        transform = tonic.transforms.Compose([downsample_transform, frame_transform, to_torch])
    return transform

def apply_light_augmentations(train = True):
    if train:
        pass # TODO
    else:
        transform = None
    return transform



if __name__ == "__main__":
    heavy_augmentations = True
    if heavy_augmentations:
        train_transform = apply_heavy_augmentations(train = True)
        test_transform = apply_heavy_augmentations(train = False)
    else:
        train_transform = apply_light_augmentations(train = True)
        test_transform = apply_light_augmentations(train = False)
    print(train_transform)
    training_set = tonic.datasets.DVSGesture(
        save_to="./data", train=True, transform=test_transform
    )
    training_set_augment = tonic.datasets.DVSGesture(
        save_to="./data", train=True, transform = train_transform
    )
    for (x, y), (x_aug, y_aug) in zip(training_set, training_set_augment):
        print("Experiment With Ground Truth", y)
        x_aug, y_aug = training_set_augment[0]
        ani = tonic.utils.plot_animation(x.numpy())

        ani = tonic.utils.plot_animation(x_aug.numpy())