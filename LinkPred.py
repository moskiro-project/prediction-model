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


class model():

    def __init__(self, load_model=False, load_test=True, save_model=True, lr=0.0015,epochs=50, batchsize=100):
        # model parameters
        self.gnn = GNN()
        self.nn = NN()
        if load_model:
            self.gnn.load_state_dict(torch.load("model/gnn_11064_50_0.0005"))
            self.nn.load_state_dict(torch.load("model/nn_11064_50_0.0005"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x, self.edge_index, self.y = util.create_dataset(test=load_test)
        self.optimizer = torch.optim.Adam(list(self.gnn.parameters()) + list(self.nn.parameters()), lr=lr)
        self.epochs= epochs
        self.lr=lr
        if batchsize is None:
            self.batchsize = self.edge_index.shape[1]
        else:
            self.batchsize = batchsize

        self.save_model = save_model
        self.gnn.to(self.device), self.nn.to(self.device)

    def train(self,plot_train=True):
        criterion = torch.nn.CrossEntropyLoss()
        np.random.default_rng()

        # generating random permutation
        permutation = torch.randperm(self.x.shape[0] - 8)
        total_loss = []
        error=[0]*self.epochs
        num_sample = 0

        # TODO change tqdm to floor len(dat)/bvatchsize *epochs
        with tqdm(range(self.epochs), desc="Training epochs") as pbar:
            for j in pbar:
                for i in range(0, self.x.shape[0] - 8, self.batchsize):
                    self.optimizer.zero_grad()
                    # Set up the batch
                    idx = permutation[i:i + self.batchsize]
                    src, tar = idx + 8, self.y[idx]

                    # add links to all "centers"
                    links= torch.vstack((torch.asarray(src.tolist()*8),
                                  torch.asarray(torch.arange(0, 8).tolist()*src.shape[0])))
                    tmp = utils.to_dense_adj(self.edge_index.type(torch.int64)).squeeze()
                    tmp = utils.dense_to_sparse(tmp)[0]
                    tmp = torch.cat((tmp, links),dim=1)
                    graph_rep = self.gnn(self.x, tmp)

                    # positive sampling
                    out = torch.sigmoid(self.nn(graph_rep[src]))
                    loss = torch.mean(criterion((out + 1e-15), tar))

                    # backward pass
                    loss.backward()
                    self.optimizer.step()

                    total_loss.append(loss)
                    num_sample += self.batchsize
                error[j] = (sum(total_loss) / num_sample).detach()
                if j == self.epochs-1:
                    if self.save_model:
                        torch.save(self.gnn.state_dict(), "model/gnn_" + str(self.batchsize) + "_" + str(self.epochs) + "_" + str(self.lr))
                        torch.save(self.nn.state_dict(), "model/nn_" + str(self.batchsize) + "_" + str(self.epochs) + "_" + str(self.lr))
                    if plot_train:
                        util.plot_curves(self.epochs, [error],
                                         ["Trainings Error"], 'Trainings Error',
                                         file_name=(str(self.batchsize) + "_" + str(self.epochs) + "_" + str(self.lr)))

    # Todo do 1  and top 3
    @torch.no_grad()
    def test(self,topk=1):
        src, tar = torch.arange(self.x.shape[0]-self.y.shape[0],self.x.shape[0]), self.y
        # add links to all "centers"
        links = torch.vstack((torch.asarray(src.tolist() * 8),
                              torch.asarray(torch.arange(0, 8).tolist()*self.y.shape[0])))
        tmp = utils.to_dense_adj(self.edge_index.type(torch.int64)).squeeze()
        tmp = utils.dense_to_sparse(tmp)[0]
        tmp = torch.cat((tmp, links), dim=1)
        graph_rep = self.gnn(self.x, tmp)

        preds=[]
        preds += [torch.sigmoid(self.nn(graph_rep[src]).cpu())]
        preds = [torch.softmax(x,dim=1) for x in preds]
        preds = [torch.topk(x, topk)[1] for x in preds]

        return preds


def main(batchsize=1, epochs=1, save=False, train_model=True, load=False, plot=False):
    # ----------------------- Set up globals
    """
    # for training
        test = model(load_model=False,save_model=True,load_test=False)
        test.train()
    # for testing
        test = model(load_model=True,save_model=False,load_test=True)
        test.test(3) for top 3 or test.test(n) for top n default is 1
    """
    test = model(load_model=False,save_model=True,load_test=False)

    test.train()
    #print(test.test(3))
if __name__ == '__main__':
    main()
# Todo which activation function ?
