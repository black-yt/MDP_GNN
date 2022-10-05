import numpy as np
import scipy.sparse as sp
import torch
import h5py
from load_pubmed import load_pubmed

laplace = 0
laplace_a = 0
laplace_b = 0
features = 0
labels = 0


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(miss_1_or_2, NC, path="./datasets/citeseer/", dataset="citeseer"):
    global laplace, laplace_a, laplace_b, features, labels

    if NC == 0:
        if dataset != "pubmed":
            print('Loading {} dataset...'.format(dataset))

            idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                                dtype=np.dtype(str))
            features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
            labels = encode_onehot(idx_features_labels[:, -1])
            idx = np.array(idx_features_labels[:, 0], dtype=np.str)
            nodes_count = idx.shape[0]
            idx_map = {j: i for i, j in enumerate(idx)}
            edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.str)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                             dtype=np.str).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)

            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])

            laplace = adj.toarray()
            eigen_value, eigen_vector = np.linalg.eig(laplace)

            creat_miss = np.arange(0, nodes_count, 1)
            miss_1 = np.mod(creat_miss, 2)

            eigen_value_1 = sp.diags(eigen_value * miss_1).A
            laplace_a = np.dot(eigen_vector, eigen_value_1)
            laplace_a = np.dot(laplace_a, np.linalg.inv(eigen_vector))
            laplace_a = laplace_a.real
            laplace_b = laplace - laplace_a

            laplace = sp.coo_matrix(laplace)
            laplace_a = sp.coo_matrix(np.int64(laplace_a > 0.5))
            laplace_b = sp.coo_matrix(np.int64(laplace_b > 0.5))

        else:
            adj, features, labels = load_pubmed()
            adj = adj + sp.eye(adj.shape[0])
            labels = encode_onehot(labels)

            nodes_count = adj.shape[0]

            mat = h5py.File(f'datasets/pubmed/m1.mat')
            aa = np.array(mat['m1'], dtype='int')
            laplace_a = sp.coo_matrix((np.ones(aa.shape[0]), (aa[:, 0], aa[:, 1])),
                                      shape=(labels.shape[0], labels.shape[0]),
                                      dtype=np.float32)
            laplace_a = laplace_a + sp.eye(laplace_a.shape[0])
            laplace_a = sp.coo_matrix(laplace_a > 0, dtype='int')
            mat = h5py.File(f'datasets/pubmed/m2.mat')
            bb = np.array(mat['m2'], dtype='int')
            laplace_b = sp.coo_matrix((np.ones(bb.shape[0]), (bb[:, 0], bb[:, 1])),
                                      shape=(labels.shape[0], labels.shape[0]),
                                      dtype=np.float32)
            laplace_b = laplace_b + sp.eye(laplace_b.shape[0])
            laplace_b = sp.coo_matrix(laplace_b > 0, dtype='int')

        features = normalize(features)
        the_h = features.max(0)
        epsilon = 30
        the_lambda = the_h / epsilon
        noise = np.random.laplace(0, the_lambda, features.shape)
        features = features + noise

        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(np.where(labels)[1])
        laplace = sparse_mx_to_torch_sparse_tensor(adj)
        laplace_a = sparse_mx_to_torch_sparse_tensor(laplace_a)
        laplace_b = sparse_mx_to_torch_sparse_tensor(laplace_b)


    if dataset == "cora":
        if NC == 2:
            idx_train_up = np.array(range(0, 200))
            idx_train_down = np.array(range(1500, 1700))
            idx_train = np.append(idx_train_up, idx_train_down)
            idx_val = range(200, 500)
            idx_test = range(500, 1500)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train = torch.LongTensor(idx_train)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_b, features, labels, idx_train, idx_val, idx_test, laplace

        if NC == 4:
            idx_train_up = range(0, 200)
            idx_train_down = range(1500, 1700)
            idx_val = range(200, 500)
            idx_test = range(500, 1500)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train_up = torch.LongTensor(idx_train_up)
            idx_train_down = torch.LongTensor(idx_train_down)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_a, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 2:
                return laplace_b, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 3:
                return laplace_b, features, labels, idx_train_down, idx_val, idx_test, laplace

        if NC == 6:
            idx_train_up = range(0, 200)
            idx_train_down = range(1500, 1700)
            idx_train_mid = range(2500, 2700)
            idx_val = range(200, 500)
            idx_test = range(500, 1500)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train_up = torch.LongTensor(idx_train_up)
            idx_train_down = torch.LongTensor(idx_train_down)
            idx_train_mid = torch.LongTensor(idx_train_mid)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_a, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 2:
                return laplace_a, features, labels, idx_train_mid, idx_val, idx_test, laplace

            if miss_1_or_2 == 3:
                return laplace_b, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 4:
                return laplace_b, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 5:
                return laplace_b, features, labels, idx_train_mid, idx_val, idx_test, laplace

    if dataset == "citeseer":
        if NC == 2:
            idx_train_up = np.array(range(0, 3000, 3))
            idx_train_down = np.array(range(1, 3001, 3))
            idx_train = np.append(idx_train_up, idx_train_down)
            idx_val = range(3100, 3300)
            idx_test = range(2, 3002, 3)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train = torch.LongTensor(idx_train)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_b, features, labels, idx_train, idx_val, idx_test, laplace

        if NC == 4:
            idx_train_up = range(0, 3000, 3)
            idx_train_down = range(1, 3001, 3)
            idx_val = range(3100, 3300)
            idx_test = range(2, 3002, 3)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train_up = torch.LongTensor(idx_train_up)
            idx_train_down = torch.LongTensor(idx_train_down)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_a, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 2:
                return laplace_b, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 3:
                return laplace_b, features, labels, idx_train_down, idx_val, idx_test, laplace

        if NC == 6:
            idx_train_up = range(0, 3000, 4)
            idx_train_down = range(1, 3001, 4)
            idx_train_mid = range(2, 3002, 4)
            idx_val = range(3100, 3300)
            idx_test = range(3, 3003, 4)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train_up = torch.LongTensor(idx_train_up)
            idx_train_down = torch.LongTensor(idx_train_down)
            idx_train_mid = torch.LongTensor(idx_train_mid)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_a, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 2:
                return laplace_a, features, labels, idx_train_mid, idx_val, idx_test, laplace

            if miss_1_or_2 == 3:
                return laplace_b, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 4:
                return laplace_b, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 5:
                return laplace_b, features, labels, idx_train_mid, idx_val, idx_test, laplace

    if dataset == "pubmed":
        if NC == 2:
            idx_train_up = np.array(range(0, 15000, 3))
            idx_train_down = np.array(range(1, 15001, 3))
            idx_train = np.append(idx_train_up, idx_train_down)
            idx_val = range(16000, 18000)
            idx_test = range(2, 15002, 3)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train = torch.LongTensor(idx_train)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_b, features, labels, idx_train, idx_val, idx_test, laplace

        if NC == 4:
            idx_train_up = range(0, 15000, 3)
            idx_train_down = range(1, 15001, 3)
            idx_val = range(16000, 18000)
            idx_test = range(2, 15002, 3)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train_up = torch.LongTensor(idx_train_up)
            idx_train_down = torch.LongTensor(idx_train_down)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_a, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 2:
                return laplace_b, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 3:
                return laplace_b, features, labels, idx_train_down, idx_val, idx_test, laplace

        if NC == 6:
            idx_train_up = range(0, 15000, 4)
            idx_train_down = range(1, 15001, 4)
            idx_train_mid = range(2, 15002, 4)
            idx_val = range(16000, 18000)
            idx_test = range(3, 15003, 4)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
            idx_train_up = torch.LongTensor(idx_train_up)
            idx_train_down = torch.LongTensor(idx_train_down)
            idx_train_mid = torch.LongTensor(idx_train_mid)

            if miss_1_or_2 == 0:
                return laplace_a, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 1:
                return laplace_a, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 2:
                return laplace_a, features, labels, idx_train_mid, idx_val, idx_test, laplace

            if miss_1_or_2 == 3:
                return laplace_b, features, labels, idx_train_up, idx_val, idx_test, laplace

            if miss_1_or_2 == 4:
                return laplace_b, features, labels, idx_train_down, idx_val, idx_test, laplace

            if miss_1_or_2 == 5:
                return laplace_b, features, labels, idx_train_mid, idx_val, idx_test, laplace


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def degree(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, 1).flatten()
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv


def degree_0_5(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)