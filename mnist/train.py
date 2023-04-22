from nmnist import augment, NMNISTDataset
from torch.utils.data import DataLoader
import lava.lib.dl.slayer as slayer
import pandas as pd
import numpy as np
import argparse
import os, sys
import torch
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
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.05),}
        num_logits = num_classes
        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Conv(neuron_params_drop,2, 16, 5, padding = 1,weight_norm = True),
                slayer.block.cuba.Pool(neuron_params_drop, 2),
                slayer.block.cuba.Conv(neuron_params_drop,16, 32, 3, padding = 1,weight_norm = True),
                slayer.block.cuba.Pool(neuron_params_drop, 2),
                slayer.block.cuba.Conv(neuron_params_drop,32, 64, 3, padding = 1, weight_norm = True),
                slayer.block.cuba.Flatten(),
                slayer.block.cuba.Dense(neuron_params, 4096, 512, weight_norm=True),
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

    parser.add_argument("--loss_fn", "--lfn", default = "spikemax_counts", type = str, help = "Options are spikemax_counts, spikemax_rates, spikemax_alpha, spikemax_alpha_theta")
    parser.add_argument("--global_alpha_theta", action= "store_true", default = False)
    parser.add_argument("--max_digit", "--md", default = 9, type = int)
    parser.add_argument("--device_id", "-d_id", default = 0, type = int)
    parser.add_argument("--resume",action = "store_true", default = False)
    parser.add_argument("--count_scale", type = float, default = 1)
    parser.add_argument("--epochs", "-ep", type = int, default = 25)
    args = parser.parse_args()
    count_scale = args.count_scale
    epochs = args.epochs
    print(args, flush = True)

    max_digit = args.max_digit
    loss_fn = args.loss_fn
    global_alpha_theta = args.global_alpha_theta
    resume = args.resume

    num_classes = max_digit + 1
    device = torch.device("cuda:%d" %args.device_id)
    #torch.cuda.set_per_process_memory_fraction(1.0, device=device)
    net = Network(num_classes=num_classes).to(device)
    batch_size = 128

    training_set = NMNISTDataset(train=True, transform=augment, max_digit=max_digit)
    testing_set  = NMNISTDataset(train=False, max_digit=max_digit)

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=testing_set , batch_size=batch_size, shuffle=True)


    if loss_fn == "spikemax_counts":
        trained_folder = "spikemax_counts-%dclass_count_scale%.2f" % (num_classes, count_scale)
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([1/300 * count_scale]).to(device=device), requires_grad=False)  # this essentially undoes the mean operation in the default spikemax function. (Each sample is 300 timesteps)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
        if not global_alpha_theta:
            raise(Exception("Using spikemax counts requires global alpha theta. Alpha and theta will be shared for all classes"))
    elif loss_fn == "spikemax_rates":
        if not global_alpha_theta:
            raise(Exception("Using spikemax rates requires global alpha theta. Alpha and theta will be shared for all classes"))
        trained_folder = "spikemax_rates-%dclass" % num_classes
        alpha = torch.nn.parameter.Parameter(data=torch.Tensor([1]).to(device=device), requires_grad=False)
        theta = torch.nn.parameter.Parameter(data=torch.Tensor([0]).to(device=device), requires_grad=False)
        error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
    elif loss_fn == "spikemax_alpha":
        
        if not global_alpha_theta:
            trained_folder = "spikemax_alpha-%dclass" % num_classes
            alpha = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.1).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),0.0).to(device=device), requires_grad=False)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha, theta = theta).to(device)
        else:
            trained_folder = "spikemax_alpha-%dclass_global_alpha" % num_classes
            alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.Tensor([0.0]).to(device=device), requires_grad=False)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
    elif loss_fn == "spikemax_alpha_theta":
        theta_penalty = 1e-3
        if not global_alpha_theta:
            trained_folder = "spikemax_alpha_theta-%dclass" % num_classes
            alpha = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.1).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.full((1, num_classes, 1),.001).to(device=device), requires_grad=True)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha, theta = theta).to(device)
        else:
            trained_folder = "spikemax_alpha_theta-%dclass_global_alpha_theta" % num_classes
            alpha = torch.nn.parameter.Parameter(data=torch.Tensor([.1]).to(device=device), requires_grad=True)
            theta = torch.nn.parameter.Parameter(data=torch.Tensor([.001]).to(device=device), requires_grad=True)
            error = slayer.loss.SpikeMax(mode="softmax", reduction = "mean", alpha = alpha[0]).to(device)
            raise(Exception("This test is proabbly useless"))

    params = list(net.parameters())
    params.append(alpha)
    params.append(theta)

    optimizer = torch.optim.Adam(params, lr=0.001)
    os.makedirs(trained_folder, exist_ok=True)

    if global_alpha_theta:
        classifier = slayer.classifier.Rate.predict  # just choose the class with most votes
    else:
        def classifier(spikes):
            return torch.argmax(torch.softmax((spikes.mean(-1) - theta.squeeze(-1))/alpha.squeeze(-1), -1), dim = -1).long() # every class has its own special theta and alpha meaning we need to manually compute probs

    if resume:
        stats_df = pd.read_csv(trained_folder + '/stats.csv')
        del(stats_df["Unnamed: 0"])  # removes useless column
        metrics_dict_list = stats_df.to_dict('records')
        start_idx = len(metrics_dict_list)
        best_acc = max(stats_df["test_acc"])
        with open(trained_folder + "/params.json", 'r') as fp:
            params = json.load(fp)
            theta.data[:] = torch.from_numpy(np.array(params["theta"])).reshape(theta.shape)
            alpha.data[:] = torch.from_numpy(np.array(params["alpha"]).reshape(alpha.shape))
        net(torch.zeros(batch_size, 2, 34, 34, 300).to(device))
        net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    else:
        start_idx = 0
        metrics_dict_list = []
        best_acc = -sys.maxsize
    torch.cuda.empty_cache()
    for epoch in range(start_idx, epochs):
        metrics_dict = {}

        if epoch >= 25:  # at this point stop updating alpha and theta
            alpha.requires_grad = False
            theta.requires_grad = False

        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        net.train()
        for i, (input, label) in enumerate(train_loader): # training loop
            if i % 8 == 0:
                print(i, flush = True)
            output = net(input.to(device))
            loss = error(output, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()

            correct_predictions += torch.sum(classifier(output).flatten() == label.to(device)).detach().cpu().numpy()
            total_predictions += output.shape[0]

        metrics_dict["train_loss"] = total_loss/(i + 1)
        metrics_dict["train_acc"] = correct_predictions/total_predictions
        metrics_dict["alpha"] = alpha.detach().cpu().numpy().tolist()
        metrics_dict["theta"] = theta.detach().cpu().numpy().tolist()
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        net.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(test_loader): # training loop
                if i % 32 == 0:
                    print(i, flush = True)
                output = net(input.to(device))
                val_loss = error(output, label.to(device)).detach().cpu().numpy()
                total_loss += val_loss
                correct_predictions += torch.sum(classifier(output).flatten() == label.to(device)).detach().cpu().numpy()
                total_predictions += output.shape[0]

        metrics_dict["test_loss"] = total_loss/(i + 1)
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