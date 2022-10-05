import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from model import GAT

print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(NC):
    model_index = []
    optimizer_index = []
    adj_index = []
    features_index = []
    labels_index = []
    idx_train_index = []
    idx_val_index = []
    idx_test_index = []
    adj = 0
    for calculator_index in range(NC):
        adj_now, features_now, labels_now, idx_train_now, idx_val_now, idx_test_now, adj = load_data(calculator_index, NC)
        model_now = GAT(in_features=features_now.shape[1],
                        nhid=args.hidden,
                        nclass=labels_now.max().item() + 1,
                        dropout=args.dropout)
        model_now = model_now.to(device)
        optimizer_now = optim.Adam(model_now.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        adj_index.append(adj_now)
        features_index.append(features_now)
        labels_index.append(labels_now)
        idx_train_index.append(idx_train_now)
        idx_val_index.append(idx_val_now)
        idx_test_index.append(idx_test_now)
        model_index.append(model_now)
        optimizer_index.append(optimizer_now)

    t_total = time.time()
    time_all = 0
    for epoch in range(args.epochs):
        weight_1 = 0
        weight_2 = 0
        bias_1 = 0
        bias_2 = 0
        for calculator_index in range(NC):
            t = time.time()
            model_now = model_index[calculator_index]
            optimizer_now = optimizer_index[calculator_index]
            features_now = features_index[calculator_index]
            adj_now = adj_index[calculator_index]
            idx_train_now = idx_train_index[calculator_index]
            labels_now = labels_index[calculator_index]
            idx_val_now = idx_val_index[calculator_index]
            model_now.train()
            optimizer_now.zero_grad()
            output = model_now(features_now, adj_now)
            loss_train = F.nll_loss(output[idx_train_now].to(device), labels_now[idx_train_now].to(device))
            acc_train = accuracy(output[idx_train_now], labels_now[idx_train_now])
            loss_train.backward()
            optimizer_now.step()

            if not args.fastmode:
                model_now.eval()
                output = model_now(features_now, adj_now)
            loss_val = F.nll_loss(output[idx_val_now].to(device), labels_now[idx_val_now].to(device))
            acc_val = accuracy(output[idx_val_now].to(device), labels_now[idx_val_now].to(device))
            print('DEVICE{:04d}'.format(calculator_index + 1),
                  'Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

            if calculator_index == 0:
                weight_1 = model_now.gcn1.weight.data
                weight_2 = model_now.gcn2.weight.data
                bias_1 = model_now.gcn1.bias.data
                bias_2 = model_now.gcn2.bias.data
                time_all = time_all + time.time() - t
            else:
                weight_1 = weight_1 + model_now.gcn1.weight.data
                weight_2 = weight_2 + model_now.gcn2.weight.data
                bias_1 = bias_1 + model_now.gcn1.bias.data
                bias_2 = bias_2 + model_now.gcn2.bias.data

        weight_1 = weight_1 / NC
        weight_2 = weight_2 / NC
        bias_1 = bias_1 / NC
        bias_2 = bias_2 / NC

        t = time.time()
        for calculator_index in range(NC):
            model_now = model_index[calculator_index]
            model_now.gcn1.weight.data = weight_1
            model_now.gcn2.weight.data = weight_2
            model_now.gcn1.bias.data = bias_1
            model_now.gcn2.bias.data = bias_2
            model_index[calculator_index] = model_now

        time_all = time_all + time.time() - t
        model_now = model_index[0]
        features_now = features_index[0]
        labels_now = labels_index[0]
        idx_test_now = idx_test_index[0]
        test(model_now, adj, features_now, labels_now, idx_test_now)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Estimated time required: {:.4f}s".format(time_all))
    print("device with complete adj:")
    model_now = model_index[0]
    features_now = features_index[0]
    labels_now = labels_index[0]
    idx_test_now = idx_test_index[0]
    test(model_now, adj, features_now, labels_now, idx_test_now)


def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test].to(device), labels[idx_test].to(device))
    acc_test = accuracy(output[idx_test].to(device), labels[idx_test].to(device))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


if __name__ == '__main__':
    NC = 6
    load_data(0, 0)
    train(NC)
