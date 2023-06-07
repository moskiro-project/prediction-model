import torch
from tqdm import tqdm
import torch.nn
import torch_geometric.utils as utils
import numpy as np
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv


import util

class GNN(torch.nn.Module):
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def __init__(self):
        # build GNN here
        super(GNN, self).__init__()
        self.input = GCNConv(128, 256, bias=False)
        self.hidden = GCNConv(256, 256, bias=False)
        self.output = GCNConv(256, 256, bias=False)

    def forward(self, x, edge_index, mask=None):
        h = self.input(x, edge_index)
        X = relu(h)
        h = self.hidden(X, edge_index)
        X = relu(h)
        h = self.output(X, edge_index)
        return h


class NN(torch.nn.Module):
    """
    3-Layer MLP with 256 input and hidden neurons and 1 output neuron
    """

    def __init__(self):
        # build MLP here
        super(NN, self).__init__()
        self.input = torch.nn.Linear(256, 256, bias=False)
        self.hidden = torch.nn.Linear(256, 256, bias=False)
        self.output = torch.nn.Linear(256, 8, bias=False)

    def forward(self, src):
        x = src
        h = self.input(x)
        X = relu(h)
        h = self.hidden(X)
        X = relu(h)
        h = self.output(X)
        return h

def train(batchsize, x, edge_index, y,optimizer, gnn, nn):
    criterion = torch.nn.CrossEntropyLoss()

    # generating random permutation
    permutation = torch.randperm(x.shape[0]-8)
    total_loss = []
    num_sample = 0

    # todo another tqdm maybe ?
    for i in range(0, x.shape[0]-8, batchsize):
        optimizer.zero_grad()
        # Set up the batch
        idx = permutation[i:i + batchsize]
        src,tar = x[idx+8],y[idx]

        # add links to all "centers"
        train_tar = torch.arange(0,8)
        tmp = utils.to_dense_adj(edge_index.type(torch.int64)).squeeze()
        tmp[idx,train_tar] = 1
        tmp[train_tar, idx] = 1
        tmp = utils.dense_to_sparse(tmp)[0]
        graph_rep = gnn(x, tmp)

        # positive sampling
        out = torch.sigmoid(nn(graph_rep[idx]))
        loss = torch.mean(criterion((out + 1e-15),tar))

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize

    return sum(total_loss) / num_sample

@torch.no_grad()
#Todo do 1  and top 3
def test(batchsize, data_set, x, adj, evaluator, gnn, nn, accuracy=False):

    # add links to all "centers"
    train_tar = torch.arange(0, 8)
    edge_index = utils.to_dense_adj(edge_index)
    edge_index[src, train_tar] = 1
    edge_index[train_tar, src] = 1
    tmp = utils.to_edge_index(edge_index)
    graph_rep = gnn(x, tmp)

    pos_preds = []
    for i in range(0, src.shape[0], batchsize):
        #Todo this
        preds += [torch.sigmoid(nn(graph_rep[src_tmp]).squeeze().cpu())]
        preds = [torch.softmax(x) for x in preds]

        preds = [torch.topk(x,3)[1] for x in preds] if top3 else [torch.topk(x,1) for x in preds]

    #TODO this needs to be done
    return pred


def main(batchsize=1, epochs=1, save=False,train_model=True, load=False, plot=False):
    # ----------------------- Set up globals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.default_rng()

    data,y = util.create_dataset()
    x,edge_index = data[0],data[1]
    # Todo move this into function
    y = torch.from_numpy(y)

    # initilaization models
    gnn, nn = GNN(), NN()
    if load:
        gnn.load_state_dict(torch.load("models/gnn_2100_50_0015"))
        nn.load_state_dict(torch.load("models/nn_2100_50_0015"))
    gnn.to(device), nn.to(device)

    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(nn.parameters()), lr=0.0005)

    with tqdm(range(epochs),desc="Training epochs") as pbar:
        for i in pbar:
            if train_model:
                loss[i] = train(batchsize, x, edge_index,y, optimizer, gnn, nn).detach()
            #test_mrr[i] = 0  # test(batchsize, test_set, data.x, data.adj_t, evaluator, gnn, nn)

            if train_mrr[i] > best and save:
                best = train_mrr[i]
                tmp_gnn = copy.deepcopy(gnn.state_dict())
                tmp_nn = copy.deepcopy(nn.state_dict())

            if i == epochs - 1:
                if save:
                    torch.save(tmp_gnn, "models/gnn_None_50_001_new")
                    torch.save(tmp_nn, "models/nn_None_50_001_new")
                if plot:
                    plots.plot_curves(epochs, [valid_mrr, test_mrr, loss],
                                      ["Valid MRR", "Test MRR", "Trainings Error"], 'Model Error',
                                      file_name="GNN" + "performance")

            pbar.set_description(f"Epoch {i}")
if __name__ == '__main__':
    main()
#Todo which activation function ?