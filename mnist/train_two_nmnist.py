from two_nmnist import TWONMNISTDataset
from torch.utils.data import DataLoader
import lava.lib.dl.slayer as slayer
from train import Network
import pandas as pd
import numpy as np
import argparse
import os, sys
import torch
import json
import time

def calculate_statistics(pred_detections, truth_detections, pred_negatives, truth_negatives):
    """Calculates true_positives, false_positives, true_negatives, and false_negatives.

    Current Implementation is coded inefficiently with O(n^2 complexity)

    Args:
        pred_detections (_type_): List of predicted detections. Each element of this list contains a list of predicted integer emitters for a given dwell. 
        truth_detections (_type_): List of ground truth detections. Each element of this list contains a list of ground truth integer emitters for a given dwell
        pred_negatives (_type_): List of prediction negatives. Each predicted negative means the SNN does not think the emitter is present in the dwell.
        truth_negatives (_type_): List of ground truth negatives. Each ground truth negatives means there is no emitter present in the dwell
    """

    true_positives = np.sum([sum(x in t for x in p) for (p, t) in  zip(pred_detections, truth_detections)])
    false_positives = np.sum([sum(x not in t for x in p) for (p, t) in  zip(pred_detections, truth_detections)])
    true_negatives = np.sum([sum(x in t for x in p) for (p, t) in  zip(pred_negatives, truth_negatives)])
    false_negatives = np.sum([sum(x not in t for x in p) for (p, t) in  zip(pred_negatives, truth_negatives)])

    return true_positives, false_positives, true_negatives, false_negatives

def combine_stats(stats, sets = ["training", "testing"]):
    """Method to compute statistics for multi class detection
    """
    for set in sets:
        stats["%s_true_positive_rate"%set] = stats["%s_true_positives"%set] / (stats["%s_true_positives"%set] + stats["%s_false_negatives"%set])
        stats["%s_false_positive_rate"%set] = stats["%s_false_positives"%set] / (stats["%s_false_positives"%set] + stats["%s_true_negatives"%set])
        stats["%s_true_negative_rate"%set] = stats["%s_true_negatives"%set] / (stats["%s_true_negatives"%set] + stats["%s_false_positives"%set])
        stats["%s_accuracy"%set] = (stats["%s_true_positives"%set] + stats["%s_true_negatives"%set]) / (stats["%s_true_positives"%set] + stats["%s_true_negatives"%set] + stats["%s_false_positives"%set]  + stats["%s_false_negatives"%set])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_digit", "--md", default = 9, type = int)
    parser.add_argument("--global_alpha_theta", action= "store_true", default = False)
    parser.add_argument("--resume", action = "store_true", default = False)
    parser.add_argument("--device_id", "-d_id", default = 0, type = int)
    parser.add_argument("--fixed_alpha_theta", action="store_true", default = False)
    parser.add_argument("--pretrain", action="store_true", default = False)
    parser.add_argument("--batch_size", default = 132, type = int)
    parser.add_argument("--epochs", "-ep", type = int, default = 50)
    args = parser.parse_args()
    fixed_alpha_theta = args.fixed_alpha_theta
    epochs = args.epochs
    print(args, flush = True)
    pretrain = args.pretrain
    batch_size = args.batch_size
    k = 2
    max_digit = args.max_digit
    resume = args.resume
    global_alpha_theta = args.global_alpha_theta
    if max_digit == 1:
        raise(Exception("Max digit 1 not compatible with this training script. Try train.py"))

    num_classes = max_digit + 1
    device = torch.device("cuda:%d" % args.device_id)
    net = Network(num_classes=num_classes).to(device)
    if pretrain:
        net.forward(torch.zeros((8, 2, 34, 34, 300)).to(device))
        net.load_state_dict(torch.load("spikemax_counts-10class_count_scale4.00/network.pt", map_location = device), strict=True)

    training_set = TWONMNISTDataset(train=True)
    testing_set  = TWONMNISTDataset(train=False)

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=testing_set , batch_size=batch_size, shuffle=True)
    trained_folder = "spikemoid-%dclass%dk" % (num_classes, k)
    
    if fixed_alpha_theta:
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.025]).to(device=device), requires_grad=False)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMoid(alpha = alpha[0], theta=theta[0], reduction="sum")
    elif global_alpha_theta:
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([.001]).to(device=device), requires_grad=True)
        error = slayer.loss.SpikeMoid(alpha = alpha[0], theta=theta[0], reduction="sum")
        trained_folder += "_global_alpha_theta"    
    else:
        alpha = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.1).to(device=device), requires_grad=True)
        theta = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.001).to(device=device), requires_grad=True)
        error = slayer.loss.SpikeMoid(alpha = alpha, theta=theta, reduction="sum")
    if pretrain:
        with open("spikemax_counts-10class_count_scale4.00" + "/params.json", 'r') as fp:
            params = json.load(fp)  # load in old alpha
            alpha.data[:] = params["alpha"][0]
            
    params = list(net.parameters())
    params.append(alpha)
    params.append(theta)

    optimizer = torch.optim.Adam(params, lr=1e-3)
    print(trained_folder)
    os.makedirs(trained_folder, exist_ok=True)

    def classifier(x1):
        mean_spikes_pred = torch.mean(x1, axis = -1)
        return (torch.sigmoid((mean_spikes_pred - theta.squeeze(-1))/alpha.squeeze(-1)) > .5).long()


    if resume:
        stats_df = pd.read_csv(trained_folder + '/stats.csv')
        del(stats_df["Unnamed: 0"])  # removes useless column
        metrics_dict_list = stats_df.to_dict('records')
        best_acc = max(stats_df["testing_accuracy"])
        start_idx = np.argmax(stats_df["testing_accuracy"]) + 1
        metrics_dict_list = metrics_dict_list[:start_idx]
        with open(trained_folder + "/params.json", 'r') as fp:
            params = json.load(fp)
            theta.data = torch.from_numpy(np.array(params["theta"])).reshape(theta.shape)
            alpha.data = torch.from_numpy(np.array(params["alpha"]).reshape(alpha.shape))
        net.forward(torch.zeros(batch_size, 2, 34, 34, 300).to(device))
        net.load_state_dict(torch.load(trained_folder + '/network.pt'), strict = False)
    else:
        start_idx = 0
        metrics_dict_list = []
        best_acc = -sys.maxsize

    for epoch in range(epochs):
        metrics_dict = {}
        print("Alpha: ", alpha, flush = True)
        print("theta: ", theta, flush = True)

        if epoch >= 25:
            alpha.requires_grad = False
            theta.requires_grad = False

        total_true_positives = 0
        total_false_positives = 0
        total_true_negatives = 0
        total_false_negatives = 0
        total_predictions = 0
        total_loss = 0
        net.train()
        for i, (input, label) in enumerate(train_loader): # training loop
            if (i+1) % 32 == 0:
                print(i, flush = True)
            start = time.time()
            output = net(input.to(device))

            optimizer.zero_grad()
            loss = error(output, label.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()
            pred_class_mask = classifier(output).detach().cpu().numpy()
            pred_detections = [np.argwhere(i) for i in pred_class_mask]
            pred_negatives = [np.argwhere(i == 0) for i in pred_class_mask]

            truth_detections = [np.argwhere(l.cpu().numpy()) for l in label]
            truth_negatives = [np.argwhere(l.cpu().numpy() == 0) for l in label]

            true_positives, false_positives, true_negatives, false_negatives = calculate_statistics(pred_detections, truth_detections, pred_negatives, truth_negatives)
            total_true_positives += true_positives
            total_false_negatives += false_positives
            total_true_negatives += true_negatives
            total_false_negatives += false_negatives

            total_predictions += sum([sum(i) for i in truth_detections])

        metrics_dict["alpha"] =  alpha.detach().cpu().numpy().tolist()
        metrics_dict["theta"] = theta.detach().cpu().numpy().tolist()
        metrics_dict["train_loss"] = total_loss/(i + 1)
        metrics_dict["training_true_positives"] = total_true_positives
        metrics_dict["training_false_positives"] = total_false_positives
        metrics_dict["training_true_negatives"] = total_true_negatives
        metrics_dict["training_false_negatives"] = total_false_negatives
        print(metrics_dict)


        total_true_positives = 0
        total_false_positives = 0
        total_true_negatives = 0
        total_false_negatives = 0
        total_predictions = 0

        total_loss = 0
        net.eval()
        for i, (input, label) in enumerate(test_loader): # training loop
            if (i+1) % 32 == 0:
                print(i, flush = True)
            with torch.no_grad():
                output = net(input.to(device))
                val_loss = error(output, label.to(device)).detach().cpu().numpy()
                total_loss += val_loss
                pred_class_mask = classifier(output).detach().cpu().numpy()
                pred_detections = [np.argwhere(i) for i in pred_class_mask]
                pred_negatives = [np.argwhere(i == 0) for i in pred_class_mask]

                truth_detections = [np.argwhere(l.cpu().numpy()) for l in label]
                truth_negatives = [np.argwhere(l.cpu().numpy() == 0) for l in label]
                true_positives, false_positives, true_negatives, false_negatives = calculate_statistics(pred_detections, truth_detections, pred_negatives, truth_negatives)
                total_true_positives += true_positives
                total_false_negatives += false_positives
                total_true_negatives += true_negatives
                total_false_negatives += false_negatives
                total_predictions += sum([sum(i) for i in truth_detections])

        metrics_dict["testing_true_positives"] = total_true_positives
        metrics_dict["testing_false_positives"] = total_false_positives
        metrics_dict["testing_true_negatives"] = total_true_negatives
        metrics_dict["testing_false_negatives"] = total_false_negatives
        metrics_dict["test_loss"] = total_loss/(i + 1)
        combine_stats(metrics_dict)

        if metrics_dict["testing_accuracy"] > best_acc:
            best_acc = metrics_dict["testing_accuracy"]
            print("Best accuracy", best_acc, flush = True)
            torch.save(net.state_dict(), trained_folder + '/network.pt')
            with open(trained_folder + "/params.json", 'w') as fp:
                json.dump({"alpha": metrics_dict["alpha"], "theta":metrics_dict["theta"] }, fp)

        print(f'[Epoch {epoch:2d}/{epochs}] {metrics_dict}', flush = True)
        metrics_dict_list.append(metrics_dict)
        pd.DataFrame(metrics_dict_list).to_csv(trained_folder + "/stats.csv")
