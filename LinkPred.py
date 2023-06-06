import torch
import tqdm

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

    def forward(self, src, tar):
        x = src + tar
        h = self.input(x)
        X = relu(h)
        h = self.hidden(X)
        X = relu(h)
        h = self.output(X)
        print(h.shape) #Todo check if this ia an 1 times 8
        return h

def train(batchsize, train_set, x, adj, optimizer, gnn, nn):
    criterion = nn.CrossEntropyLoss()

    # generating random permutation
    #Todo change this permutation = torch.randperm(train_set["source_node"].shape[0])
    total_loss = []
    num_sample = 0

    # todo another tqdm maybe ?
    for i in range(0, train_set["source_node"].shape[0], batchsize):
        optimizer.zero_grad()
        # Set up the batch
        idx = permutation[i:i + batchsize]
        #Todo change this train_src, train_tar = train_set["source_node"][idx], train_set["target_node"][idx]

        # add links to all "centers"
        train_tar = torch.arange(0,8)
        tmp = adj.to_dense()
        tmp[train_src, train_tar] = 1
        tmp = torch_sparse.SparseTensor.from_dense(tmp)
        graph_rep = gnn(x, tmp)

        # positive sampling
        out = torch.sigmoid(nn(graph_rep[train_src]))
        loss = torch.mean(criterion((out + 1e-15),y))

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize

    return sum(total_loss) / num_sample

@torch.no_grad()
#Todo do 1  and top 3
def test(batchsize, data_set, x, adj, evaluator, gnn, nn, accuracy=False):

    tmp = data_set["source_node"].shape[0]
    permutation = torch.randperm(tmp)
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]

    train_tar = torch.arange(0, 8)
    tmp = adj.to_dense()
    tmp[train_src, train_tar] = 1
    tmp = torch_sparse.SparseTensor.from_dense(tmp)
    graph_rep = gnn(x, tmp)

    pos_preds = []
    for i in range(0, src.shape[0], batchsize):
        #Todo this
        idx = permutation[i:i + batchsize]
        src_tmp = src[idx]
        tar_tmp = tar[idx]
        tar_neg_tmp = tar_neg[idx]

        preds += [torch.sigmoid(nn(graph_rep[src_tmp]).squeeze().cpu())]
        preds = [torch.softmax(x) for x in preds]

        preds = [torch.topk(x,3)[1] for x in preds] if top3 else [torch.topk(x,1) for x in preds]

    #TODO this needs to be done
    return pred


def main(batchsize=1, epochs=1, save=False,train_model=False, load=False, plot=False):
    # ----------------------- Set up globals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.default_rng()

    #Todo need to do dataset
    dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)
    data = dataset.load()
    split = dataset.get_edge_split()
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    tmp = data.adj_t.set_diag()
    deg = tmp.sum(dim=0).pow(-0.5)
    deg[deg == float('inf')] = 0
    tmp = deg.view(-1, 1) * tmp * deg.view(1, -1)
    data.adj_t = tmp

    # initilaization models
    gnn, nn = GNN(), NN()
    if load:
        gnn.load_state_dict(torch.load("models/gnn_2100_50_0015"))
        nn.load_state_dict(torch.load("models/nn_2100_50_0015"))
    gnn.to(device), nn.to(device)

    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(nn.parameters()), lr=0.0005)

    with trange(epochs,desc="Training epochs") as pbar:
        for i in pbar:
            if train_model:
                loss[i] = train(batchsize, train_set, data.x, data.adj_t, optimizer, gnn, nn).detach()
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

#Todo which activation function ?