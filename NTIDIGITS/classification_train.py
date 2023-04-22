from ntidigits import NTIDIGITSDetection
from torch.utils.data import DataLoader
import lava.lib.dl.slayer as slayer
import pandas as pd
import numpy as np
import argparse
import torch
import h5py
import json
import sys
import os

class Network(torch.nn.Module):
    def __init__(self, num_classes = 11):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,
                'voltage_decay' : 0.25,
                'tau_grad'      : 0.1,
                'scale_grad'    : 1,
                'requires_grad' : False,
                'shared_param'  : False,
                "graded_spike"  : False,
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.1),}

        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(neuron_params_drop, 64, 256, weight_scale = 2, delay = True),
                slayer.block.cuba.Dense(neuron_params_drop, 256, 256,weight_scale = 2,  delay = True),
                slayer.block.cuba.Dense(neuron_params_drop, 256, num_classes, weight_scale = 2),
            ])
    
    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

def combine_stats(stats, sets = ["training", "testing"]):
    """Method to compute statistics for multi class detection
    """
    for set in sets:
        stats["%s_accuracy"%set] = stats["%s_correct_predictions"%set] / stats["%s_total_predictions"%set]

def get_correct_predictions(x:np.array, y:np.array, device):
    correct_predictions = (x.sum(-1).argmax(-1) == y).sum()
    return {"correct_predictions": correct_predictions, "total_predictions": x.shape[0]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_alpha_theta", action="store_true", default = False)
    parser.add_argument("--run_name", "-r_n", default = "0", type = str)
    parser.add_argument("--resume",action = "store_true", default = False)
    parser.add_argument("--counts",action = "store_true", default = False)
    parser.add_argument("--count_scale", type = float, default = 1)
    parser.add_argument("--device_id", "-d_id", default = 0, type = int)
    parser.add_argument("--pretrain", action= "store_true", default = False)
    parser.add_argument("--epochs", "-ep", type = int, default = 400)
    parser.add_argument("--stop_updating_alpha_theta", default = 200, type = int )
    args = parser.parse_args()
    print(args)
    epochs = args.epochs
    count_scale = args.count_scale
    pretrain = args.pretrain
    device = torch.device("cuda:%d" % args.device_id)
    global_alpha_theta = args.global_alpha_theta
    resume = args.resume
    run_name = args.run_name
    counts = args.counts

    batch_size = 4
    pad_timesteps = 1000
    lr = 1e-3
    trained_folder = "bamsumit_params{}".format(run_name)
    if pretrain:
        trained_folder += "_pretrain"
    elif counts:
        trained_folder += "_counts%.2f" % count_scale

    net = Network().to(device)
    h5py_file = h5py.File('n-tidigits.hdf5', 'r', swmr=True, libver="latest")
    if pretrain:
        pad_timesteps = 3000
        lr = 5e-4
        pretrain_folder = "ntidigits_detection_results"
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([1]).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0], theta=theta[0]).to(device)
        net.forward(torch.zeros((8, 64, 300)).to(device))
        net.load_state_dict(torch.load(pretrain_folder + '/network.pt'), strict = False)
    elif counts:
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([1/pad_timesteps * count_scale]).to(device=device), requires_grad=False)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0], theta = theta[0]).to(device)
    elif global_alpha_theta:
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([0.01]).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMax(alpha = alpha[0], theta=theta[0], reduction='mean', mode="softmax").to(device)
        trained_folder += "_global"
    else:
        alpha = torch.nn.parameter.Parameter(data=torch.full((1, 11, 1),.01).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.full((1, 11, 1),.001).to(device = device), requires_grad=True)
        error = slayer.loss.SpikeMax(alpha = alpha[0], theta=theta[0], reduction='mean', mode="softmax").to(device)
    print("trained folder: ", trained_folder)
    if not os.path.exists(trained_folder):
        os.mkdir(trained_folder)

    training_set = NTIDIGITSDetection(h5py_file, 1e3, 1, train = True, pad_timesteps=pad_timesteps)
    testing_set =  NTIDIGITSDetection(h5py_file, 1e3, 1, train = False, pad_timesteps=pad_timesteps)
    if resume:
        stats_df = pd.read_csv(trained_folder + '/stats.csv')
        del(stats_df["Unnamed: 0"])  # removes useless column
        metrics_dict_list = stats_df.to_dict('records')
        best_acc = max(stats_df["testing_accuracy"])
        start_idx = np.argmax(stats_df["testing_accuracy"]) + 1
        metrics_dict_list = metrics_dict_list[:start_idx]
        with open(trained_folder + "/params.json", 'r') as fp:
            json_params = json.load(fp)
            theta.data[:] = torch.from_numpy(np.array(json_params["theta"])).reshape(theta.shape)
            alpha.data[:] = torch.from_numpy(np.array(json_params["alpha"]).reshape(alpha.shape))
        error = slayer.loss.SpikeMax(alpha = alpha[0], theta=theta[0], reduction='mean', mode="softmax").to(device)
        net.forward(torch.zeros((8, 64, 300)).to(device))
        net.load_state_dict(torch.load(trained_folder + '/network.pt'), strict = False)
    else:
        start_idx = 0
        metrics_dict_list = []
        best_acc = -sys.maxsize
    evaluator = lambda x, y:  get_correct_predictions(x, y, device)
    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=testing_set , batch_size=batch_size * 4, shuffle=False)
    params = list(net.parameters())
    params += [theta, alpha]
    optimizer = torch.optim.Adam(params, lr=lr)
    print("starting training", flush = True)
    print("starting alpha", alpha)
    print("starting theta", theta)
    for epoch in range(start_idx, epochs):
        if epoch > args.stop_updating_alpha_theta:
            alpha.requires_grad = False
            theta.requires_grad = False
        stats = {}
        net.train()
        stats["training_loss"] = 0
        for i, (input, label) in enumerate(train_loader): # training loop
            if (i+1) % 32 == 0:
                print(i, flush = True)
            optimizer.zero_grad()
            output = net(input.float().to(device))
            label = label.sum(-1) != 0
            labels = torch.argwhere(label)[:, -1]
            loss = error(output, labels.to(device))
            stats["training_loss"] += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            class_dict = evaluator(output.detach().cpu().numpy(), labels.detach().cpu().numpy())
            for key, val in class_dict.items(): # iterate over batch dict and update stats dict
                if "training_%s" % key not in stats:
                    stats["training_%s" % key] = 0
                stats["training_%s" % key] += val
        stats["training_loss"] /= (i + 1)  # average testing loss

        net.eval()
        stats["testing_loss"] = 0
        with torch.no_grad():
            for i, (input, label) in enumerate(test_loader): # training loop
                if (i+1) % 32 == 0:
                    print(i, flush = True)
                output = net(input.float().to(device))
                label = label.sum(-1) != 0
                labels = torch.argwhere(label)[:, -1]
                loss = error(output, labels.to(device))
                stats["testing_loss"] += loss.detach().cpu().numpy()
                class_dict = evaluator(output.detach().cpu().numpy(), labels.detach().cpu().numpy())
                for key, val in class_dict.items(): # iterate over batch dict and update stats dict
                    if "testing_%s" % key not in stats:
                        stats["testing_%s" % key] = 0
                    stats["testing_%s" % key] += val
            stats["testing_loss"] /= (i + 1)  # average testing loss

        stats["alpha"] = alpha.detach().cpu().numpy().flatten()
        stats["theta"] = theta.detach().cpu().numpy().flatten()
        combine_stats(stats)
        print("Epoch", epoch, stats)
        if stats["testing_accuracy"] > best_acc:
            best_acc = stats["testing_accuracy"]
            print("Best acc :", best_acc, "epoch: ", epoch)
            torch.save(net.state_dict(),  trained_folder + "/network.pt")  
            with open(trained_folder + "/params.json", 'w') as fp:
                json.dump({"alpha": alpha.detach().cpu().numpy().flatten().tolist(), "theta": theta.detach().cpu().numpy().flatten().tolist()}, fp)
        metrics_dict_list.append(stats)
        pd.DataFrame(metrics_dict_list).to_csv(trained_folder + "/stats.csv")
