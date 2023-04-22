from torch.utils.data import DataLoader
import lava.lib.dl.slayer as slayer
import torchvision
import pandas as pd
import numpy as np
import argparse
import os, sys
import torch
import tonic
import json

class Network(torch.nn.Module):
    def __init__(self, num_classes = 10):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : False,   
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.1),}
        if num_classes == 2:
            num_logits = 1  # Use spikemoid loss
        else:
            num_logits = num_classes
        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Pool(neuron_params_drop, 4),
                slayer.block.cuba.Conv(neuron_params_drop,2, 16, 5, padding = 1, weight_norm = True, delay = True),
                slayer.block.cuba.Pool(neuron_params_drop, 2),
                slayer.block.cuba.Conv(neuron_params_drop,16, 32, 3, padding = 1,weight_norm = True, delay = True),
                slayer.block.cuba.Pool(neuron_params_drop, 2),
                slayer.block.cuba.Conv(neuron_params_drop,32, 64, 3, padding = 1, weight_norm = True, delay = True),
                slayer.block.cuba.Flatten(),
                slayer.block.cuba.Dense(neuron_params_drop, 4096, 512, weight_norm=True, delay = True), #16384
                slayer.block.cuba.Dense(neuron_params, 512, num_logits, weight_norm=True),
            ])
    
    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

def save_stat_dict(stats, trained_folder):
    dict_list= []
    for train_loss, train_acc, test_loss, test_acc in zip(stats.training.loss_log,stats.training.accuracy_log,stats.testing.loss_log,stats.testing.accuracy_log):
        dict = {}
        dict["train_loss"] = train_loss
        dict["train_acc"] = train_acc
        dict["test_loss"] = test_loss
        dict["test_acc"] = test_acc
        dict_list.append(dict)
    pd.DataFrame(dict_list).to_csv(trained_folder + "/stats.csv")

def random_crop(img, padding_size=8, crop_size=64):
    """
    This breaks if crop_size is not evenly divisible by 2
    """

    # select x and y center of the random crop
    x = int(np.random.randint((crop_size//2)-1, high=img.shape[1]+padding_size-(crop_size//2)))
    y = int(np.random.randint((crop_size//2)-1, high=img.shape[2]+padding_size-(crop_size//2)))

    smol_imgs = []
    for channel in range(img.shape[0]):
        smol_img = img[channel, :, :]
        # create zero-padded larger img
        padded = np.zeros((smol_img.shape[0]+padding_size, smol_img.shape[1]+padding_size))
        # set original image into the center of the zero img
        padded[padding_size:smol_img.shape[0]+padding_size, padding_size:smol_img.shape[1]+padding_size] = smol_img

        # crop out image
        smol_imgs.append(padded[x-((crop_size//2)-1):x+((crop_size//2)-1), y-((crop_size//2)-1):y+((crop_size//2)-1)])

    return np.vstack(smol_imgs)

def random_crop_transform(event):
    cropped_imgs = []
    for i in range(len(event)):
        cropped_imgs.append(random_crop(event[i, :, :, :], padding_size=8, crop_size=64))
    return np.vstack(cropped_imgs)

def to_device_transform(input_):
    return input_.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    Experimental options
    --loss_fn spikemax_counts --global_alpha_theta (Alpha and theta will not be used)
    --loss_fn spikemax_rates --global_alpha_theta (Alpha and theta will not be used)
    --loss_fn spikemax_alpha --global_alpha_theta (Just alpha is updated)
    --loss_fn spikemax_alpha     (Just alpha is updated)
    --loss_fn spikemax_alpha_theta  (Alpha and theta are updated)
    """
    parser.add_argument("--loss_fn", "--lfn", default = "spikemax_counts", type = str, help = "Options are spikemax_counts, spikemax_rates, spikemax_alpha, spikemax_alpha_theta, spikemoid (spikemoid for max_digit = 2)")
    parser.add_argument("--global_alpha_theta", action= "store_true", default = False)
    parser.add_argument("--max_digit", "--md", default = 10, type = int)
    parser.add_argument("--device_id", "-d_id", default = 0, type = int)
    parser.add_argument("--run_name", "-r_n", default = "0", type = str)
    parser.add_argument("--resume",action = "store_true", default = False)
    parser.add_argument("--counts",action = "store_true", default = False)
    parser.add_argument("--count_scale", type = float, default = 1)
    parser.add_argument("--epochs", "-ep", type = int, default = 100)

    args = parser.parse_args()
    counts = args.counts
    count_scale = args.count_scale
    print(args, flush = True)

    max_digit = args.max_digit
    loss_fn = args.loss_fn
    global_alpha_theta = args.global_alpha_theta
    run_name = args.run_name
    resume = args.resume

    num_classes = max_digit + 1
    device = torch.device("cuda:%d" %args.device_id)
    net = Network(num_classes=num_classes).to(device)
    batch_size = 32

    sensor_size = tonic.datasets.DVSGesture.sensor_size
    timesteps = 1500
    rotation_transform = torchvision.transforms.RandomRotation([-10,10])
    jitter_transform = tonic.transforms.SpatialJitter(sensor_size=(128, 128, 2), var_x=16, var_y=16, clip_outliers=True)
    frame_transform = tonic.transforms.ToFrame(sensor_size=(128, 128, 2), n_time_bins=timesteps)
    cast_to_torch_gpu = lambda x:  torch.from_numpy(x).to(device)
    def sample_transform(image):
        start_sample = np.random.randint(0, image.shape[0] - 300)
        end_sample = start_sample + 300
        return image[start_sample:end_sample]

    safe_transform = tonic.transforms.Compose([
                                                jitter_transform,
                                               frame_transform,
                                               sample_transform,
                                               cast_to_torch_gpu,
                                               rotation_transform
                                               ])

    test_transform = tonic.transforms.Compose([
                                               frame_transform,
                                               cast_to_torch_gpu
                                               ])
    training_set = tonic.datasets.DVSGesture(
        save_to="./data", train=True, transform=safe_transform
    )
    testing_set = tonic.datasets.DVSGesture(
        save_to="./data", train=False, transform=test_transform
    )

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=testing_set , batch_size=batch_size, shuffle=False)
    if counts or loss_fn == "spikemax_counts":
        trained_folder = "spikemax_counts-scale{}_{}".format(count_scale, run_name)
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([1/300 * count_scale]).to(device=device), requires_grad=False)  # this essentially undoes the mean operation in the default spikemax function. (Each sample is 300 timesteps)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
        if not global_alpha_theta:
            raise(Exception("Using spikemax counts requires global alpha theta. Alpha and theta will be shared for all classes"))
    elif loss_fn == "spikemax_rates":
        if not global_alpha_theta:
            raise(Exception("Using spikemax rates requires global alpha theta. Alpha and theta will be shared for all classes"))
        trained_folder = "spikemax_rates-{}class{}".format(num_classes, run_name)
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([1]).to(device=device), requires_grad=False)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
    elif loss_fn == "spikemax_alpha":
        
        if not global_alpha_theta:
            trained_folder = "spikemax_alpha-{}class{}".format(num_classes, run_name)
            alpha = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.1).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),0.0).to(device=device), requires_grad=False)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha, theta = theta).to(device)
        else:
            trained_folder = "spikemax_alpha-{}class_global_alpha{}".format(num_classes, run_name)
            alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
    elif loss_fn == "spikemax_alpha_theta":
        theta_penalty = 1e-3
        if not global_alpha_theta:
            trained_folder = "spikemax_alpha_theta-{}class{}".format(num_classes, run_name)
            alpha = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.1).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),0.0).to(device=device), requires_grad=True)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha, theta = theta).to(device)
        else:
            trained_folder = "spikemax_alpha_theta-{}class_global_alpha_theta{}".format(num_classes, run_name)
            alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.Tensor([0.0]).to(device=device), requires_grad=True)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
            raise(Exception("This test is proabbly useless"))
    elif loss_fn == "spikemoid":
        theta_penalty = 1e-3
        if max_digit != 1:
            raise(Exception("Cannot use spikemoid with more than two classes"))
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([1e-3]).to(device=device), requires_grad=True)
        error = slayer.loss.SpikeMoid(alpha = alpha[0], theta=theta[0])
        trained_folder = "spikemoid-{}class{}".format(num_classes, run_name)

    params = list(net.parameters())
    params.append(alpha)
    params.append(theta)

    optimizer = torch.optim.Adam(params, lr=0.001)
    os.makedirs(trained_folder, exist_ok=True)

    if loss_fn == "spikemoid":
        def classifier(spikes):
           return (torch.sigmoid((spikes.mean(-1) - theta)/alpha) > .5).long()
    elif global_alpha_theta:
        classifier = slayer.classifier.Rate.predict  # just choose the class with most votes
    else:
        def classifier(spikes):
            return torch.argmax(torch.softmax((spikes.mean(-1) - theta.squeeze(-1))/alpha.squeeze(-1), -1), dim = -1).long() # every class has its own special theta and alpha meaning we need to manually compute probs

    epochs = args.epochs
    if resume:
        stats_df = pd.read_csv(trained_folder + '/stats.csv')
        del(stats_df["Unnamed: 0"])  # removes useless column
        metrics_dict_list = stats_df.to_dict('records')
        best_acc = max(stats_df["test_acc"])

        start_idx = np.argmax(stats_df["test_acc"]) + 1
        metrics_dict_list = metrics_dict_list[:start_idx]

        with open(trained_folder + "/params.json", 'r') as fp:
            params = json.load(fp)
            theta.data[:] = torch.from_numpy(np.array(params["theta"])).reshape(theta.shape)
            alpha.data[:] = torch.from_numpy(np.array(params["alpha"]).reshape(alpha.shape))
        net(torch.zeros(batch_size, 2, 128, 128, 1500).to(device))
        net.load_state_dict(torch.load(trained_folder + '/network.pt'), strict = False)
        
    else:
        start_idx = 0
        metrics_dict_list = []
        best_acc = -sys.maxsize
    for epoch in range(start_idx, epochs):
        metrics_dict = {}
        if epoch >= 25:  # at this point stop updating alpha and theta
            alpha.requires_grad = False
            theta.requires_grad = False
            for g in optimizer.param_groups:
                g['lr'] = 1e-4  # decay learning rate

        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        net.train()
        import time
        start = time.time()
        for i, (input_, label) in enumerate(train_loader): # training loop
            input_ = (input_ > 0).moveaxis(1, -1).float()
            if i % 32 == 0:
                print(i, flush = True)
            output = net(input_)
            loss = error(output, label.to(device))
            optimizer.zero_grad()
            if loss_fn in ["spikemoid"]:  # no need to add penalty in this case
                (loss + theta.abs().mean()[0]).backward()
            else:
                loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()

            correct_predictions += torch.sum(classifier(output).flatten() == label.to(device)).detach().cpu().numpy()
            total_predictions += output.shape[0]
        metrics_dict["train_loss"] = total_loss/total_predictions
        metrics_dict["train_acc"] = correct_predictions/total_predictions
        metrics_dict["alpha"] = alpha.detach().cpu().numpy().tolist()
        metrics_dict["theta"] = theta.detach().cpu().numpy().tolist()
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        net.eval()
        with torch.no_grad():
            for i, (input_, label) in enumerate(test_loader): # training loop
                if i % 32 == 0:
                    print(i, flush = True)
                input_ = (input_ > 0).moveaxis(1, -1).float()
                output = net(input_)
                val_loss = error(output, label.to(device)).detach().cpu().numpy()
                total_loss += val_loss
                correct_predictions += torch.sum(classifier(output).flatten() == label.to(device)).detach().cpu().numpy()
                total_predictions += output.shape[0]
        print("Time for 1 epoch:", time.time() - start)
        metrics_dict["test_loss"] = total_loss/total_predictions
        metrics_dict["test_acc"] = correct_predictions/total_predictions
        if metrics_dict["test_acc"] > best_acc:
            best_acc = metrics_dict["test_acc"]
            print("Best acc", best_acc, flush = True)
            torch.save(net.state_dict(), trained_folder + '/network.pt')
            with open(trained_folder + "/params.json", 'w') as fp:
                json.dump({"alpha": alpha.detach().cpu().numpy().tolist(), "theta":theta.detach().cpu().numpy().tolist()}, fp)
            
        print(f'[Epoch {epoch:2d}/{epochs}] {metrics_dict}', flush = True)

        
        metrics_dict_list.append(metrics_dict)
        pd.DataFrame(metrics_dict_list).to_csv(trained_folder + "/stats.csv")
