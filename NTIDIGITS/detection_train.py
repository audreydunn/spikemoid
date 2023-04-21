from ntidigits import NTIDIGITSDetection, get_smallest_time_sample
from torch.utils.data import DataLoader
import lava.lib.dl.slayer as slayer
from scipy.special import expit
import pandas as pd
import numpy as np
import argparse
import torch
import h5py
import json
import sys
import os

from classification_train import Network

def combine_stats(stats, sets = ["training", "testing"]):
    """Method to compute statistics for multi class detection
    """
    for set in sets:
        stats["%s_true_positive_rate"%set] = stats["%s_true_positives"%set] / (stats["%s_true_positives"%set] + stats["%s_false_negatives"%set])
        stats["%s_false_positive_rate"%set] = stats["%s_false_positives"%set] / (stats["%s_false_positives"%set] + stats["%s_true_negatives"%set])
        stats["%s_true_negative_rate"%set] = stats["%s_true_negatives"%set] / (stats["%s_true_negatives"%set] + stats["%s_false_positives"%set])
        stats["%s_f1_score"%set] = (2 * stats["%s_true_positives"%set]) / ( 2 * stats["%s_true_positives"%set] + stats["%s_false_positives"%set]  + stats["%s_false_negatives"%set])

def spikemoid_detection_scores(x, y, alpha, theta, p_cutoff = .5):
    probs = expit((x.mean(-1) - theta)/alpha)
    decisions = probs >= p_cutoff
    digits_present = y.sum(-1) != 0
    present_mask = digits_present == 1
    non_present_mask = digits_present == 0
    true_positives = np.sum(decisions[present_mask] == 1)
    false_positives = np.sum(decisions[non_present_mask])

    true_negatives = np.sum(decisions[non_present_mask] == 0)
    false_negatives = np.sum(decisions[present_mask] == 0)

    classification_score_dict = {"true_positives": true_positives,
                                 "false_positives": false_positives,
                                 "true_negatives": true_negatives,
                                 "false_negatives": false_negatives}
    return classification_score_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_alpha_theta", action="store_true", default = False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--device_id", "-d_id", default = 0, type = int)
    parser.add_argument("--run_name", "-r_n", default = "", type = str)
    parser.add_argument("--fixed_alpha_theta", action="store_true", default = False)
    parser.add_argument("--pretrain", action="store_true", default = False)
    parser.add_argument("--epochs", "-ep", type = int, default = 400)
    parser.add_argument("--pretrain_folder", default = ".", type = str)
    args = parser.parse_args()
    print(args, flush = True)
    epochs = args.epochs
    pretrain = args.pretrain
    pretrain_folder = args.pretrain_folder
    fixed_alpha_theta = args.fixed_alpha_theta

    resume = args.resume
    global_alpha_theta = args.global_alpha_theta
    run_name = args.run_name

    scale_timesteps = 1e3  # scale timesteps by this value
    sample_time = 1
    batch_size = 16
    lr = 1e-3
    device = torch.device("cuda:%d" % args.device_id)
    if fixed_alpha_theta:
        trained_folder = "ntidigits_detection_results_fixed{}".format(run_name) 
    elif global_alpha_theta:
        trained_folder = "ntidigits_detection_results_global{}".format(run_name)
    else:
        trained_folder = "ntidigits_detection_results{}".format(run_name)

    print(trained_folder)
    if not os.path.exists(trained_folder):
        os.mkdir(trained_folder)

    h5py_file = h5py.File('n-tidigits.hdf5', 'r', swmr=True, libver="latest")
    # # Obtain the smallest singleton digit
    smallest_sample_name, smallest_time_length = get_smallest_time_sample(h5py_file)

    padding_size = 3000
    num_classes = 11
    if fixed_alpha_theta:
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.02]).to(device=device), requires_grad=False)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMoid(alpha=alpha[0], theta=theta[0], reduction='mean').to(device)       
    elif global_alpha_theta:
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([.001]).to(device=device), requires_grad=True)
        error = slayer.loss.SpikeMoid(alpha=alpha[0], theta=theta[0], reduction='mean').to(device)
    else:
        alpha = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.1).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.001).to(device=device), requires_grad=True)
        error = slayer.loss.SpikeMoid(alpha = alpha, theta=theta, reduction="mean")

    training_set = NTIDIGITSDetection(h5py_file, scale_timesteps, sample_time, train=True, only_singleton = False, pad_timesteps = padding_size)
    testing_set = NTIDIGITSDetection(h5py_file, scale_timesteps, sample_time, train=False, only_singleton = False, pad_timesteps = padding_size)

    net = Network().to(device)

    # load pretrained
    if pretrain:
        stats_df = pd.read_csv(pretrain_folder + '/stats.csv')
        del(stats_df["Unnamed: 0"])  # removes useless column
        metrics_dict_list = stats_df.to_dict('records')
        start_idx = len(metrics_dict_list)
        best_f1 = -sys.maxsize
        with open(pretrain_folder + "/params.json", 'r') as fp:
            params = json.load(fp)
            if fixed_alpha_theta:
                pass
            elif global_alpha_theta:
                alpha.data[:] = params["alpha"][0]
                theta.data[:] = 1e-3
            else:
                alpha.data[:] = torch.full((1, num_classes, 1), params["alpha"][0]) 
                theta.data[:] = 1e-3

        if fixed_alpha_theta:
            error = slayer.loss.SpikeMoid(alpha=alpha[0], theta=theta[0], reduction='mean').to(device)   
        elif global_alpha_theta:
            error = slayer.loss.SpikeMoid(alpha=alpha[0], theta=theta[0], reduction='mean').to(device)
        else:
            error = slayer.loss.SpikeMoid(alpha = alpha, theta=theta, reduction="mean").to(device)
        
        net(torch.zeros(batch_size, 64, 300).to(device)).to(device)
        net.load_state_dict(torch.load(pretrain_folder + '/network.pt'), strict=True)

    if resume:
        stats_df = pd.read_csv(trained_folder + '/stats.csv')
        del(stats_df["Unnamed: 0"])  # removes useless column
        metrics_dict_list = stats_df.to_dict('records')
        start_idx = len(metrics_dict_list)
        best_f1 = min(stats_df["testing_f1_score"])
        with open(trained_folder + "/spikemoid_params.json", 'r') as fp:
            params = json.load(fp)
            theta.data[:] = torch.from_numpy(np.array(params["theta"])).reshape(theta.shape)
            alpha.data[:] = torch.from_numpy(np.array(params["alpha"]).reshape(alpha.shape))
        net(torch.zeros(batch_size, 64, 300).to(device))
        net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    else:
        start_idx = 0
        metrics_dict_list = []
        best_f1 = -sys.maxsize
    epoch = start_idx
    evaluator = lambda x, y:  spikemoid_detection_scores(x, y, alpha = alpha.detach().cpu().squeeze(-1).numpy(), 
        theta = theta.detach().cpu().squeeze(-1).numpy(), p_cutoff=.50)
    
    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=testing_set , batch_size=batch_size * 4, shuffle=False)
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    optimizer2 = torch.optim.Adam([theta, alpha],lr = 1e-4)
    for epoch in range(epochs):
        if epoch >= 200:
            theta.requires_grad = False
            alpha.requires_grad = False

        stats = {}
        net.train()
        stats["training_loss"] = 0
        for i, (input, label) in enumerate(train_loader): # training loop
            if (i+1) % 100 == 0:
                print(i, flush = True)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            output = net(input.float().to(device))
            loss = error(output, (label.sum(-1) != 0).float().to(device))  
            flat_label = label.flatten()
            stats["training_loss"] += loss.detach().cpu().numpy()
            loss.backward()

            optimizer.step()
            optimizer2.step()


            class_dict = evaluator(output.detach().cpu().numpy(), label.detach().cpu().numpy())
            for key, val in class_dict.items(): # iterate over batch dict and update stats dict
                if "training_%s" % key not in stats:
                    stats["training_%s" % key] = 0
                stats["training_%s" % key] += val
        stats["training_loss"] /= (i + 1)  # average testing loss

        net.eval()
        stats["testing_loss"] = 0
        with torch.no_grad():
            for i, (input, label) in enumerate(test_loader): # training loop
                if (i+1) % 100 == 0:
                    print(i, flush = True)
                output = net(input.float().to(device))
                loss = error(output, (label.sum(-1) != 0).float().to(device))  
                flat_label = label.flatten()
                stats["testing_loss"] += loss.detach().cpu().numpy()
                class_dict = evaluator(output.detach().cpu().numpy(), label.detach().cpu().numpy())
                for key, val in class_dict.items(): # iterate over batch dict and update stats dict
                    if "testing_%s" % key not in stats:
                        stats["testing_%s" % key] = 0
                    stats["testing_%s" % key] += val
            stats["testing_loss"] /= (i + 1)  # average testing loss

        stats["alpha"] = alpha.detach().cpu().numpy().flatten()
        stats["theta"] = theta.detach().cpu().numpy().flatten()
        combine_stats(stats)
        print("Epoch", epoch, stats)
        if stats["testing_f1_score"] > best_f1:
            best_f1 = stats["testing_f1_score"]
            print("Best f1 :", best_f1, "epoch: ", epoch)
            torch.save(net.state_dict(),  trained_folder + "/network.pt")  
            with open(trained_folder + "/spikemoid_params.json", 'w') as fp:
                json.dump({"alpha": alpha.detach().cpu().numpy().flatten().tolist(), "theta": theta.detach().cpu().numpy().flatten().tolist() 
                            }, fp)
        metrics_dict_list.append(stats)
        pd.DataFrame(metrics_dict_list).to_csv(trained_folder + "/stats.csv")
