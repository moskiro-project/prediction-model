# Purely for testing purposes, not used in any further scripts.

from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit
from emgraph.datasets import BaseDataset, DatasetType
from emgraph.models import TransE


def train_transe(data):
    
    model = TransE(batches_count=64, seed=0, epochs=20, k=100, eta=20,
                   optimizer='adam', optimizer_params={'lr': 0.0001},
                   loss='pairwise', verbose=True, large_graphs=False)
    model.fit(data['train'])
    scores = model.predict(data['test'])
    return scores
    

if __name__ == '__main__':
    
    wn11_dataset = BaseDataset.load_dataset(DatasetType.WN11)
    
    scores = train_transe(data=wn11_dataset)
    print("Scores: ", scores)
    print("Brier score loss:", brier_score_loss(wn11_dataset['test_labels'], expit(scores)))