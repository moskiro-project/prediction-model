import os

import pandas as pd
import numpy as np
import torch

from emgraph.datasets import BaseDataset, DatasetType
from emgraph.models import ComplEx, ConvKB, DistMult, HolE, TransE
from emgraph.evaluation.protocol import filter_unseen_entities

class model():

    def __init__(self, lr=0.0005,epochs=50, batchsize=64):
        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize

        self.model = TransE(batches_count=self.batchsize,seed=0,epochs=self.epochs,k=100,
                            eta=10,optimizer="adam",optimizer_params={"lr": self.lr},loss="pairwise",
                            verbose=False,large_graphs=False,)

        self.train_data = BaseDataset.load_from_csv(os.getcwd(), "data/train_data_graph.csv", ',')
        self.test_data = BaseDataset.load_from_csv(os.getcwd(), "data/test_data_graph.csv", ',')

    def train(self):
        model.fit(self.train_train)

    def test(self,topk=1):

        scores = model.predict(self.test_data)
        preds = scores.reshape(288,8)
        preds = [torch.topk(x, topk)[1] for x in preds]

        return preds


def main(batchsize=1, epochs=1, save=False, train_model=True, load=False, plot=False):
    # ----------------------- Set up globals
    """
    # for training
        test = model()
        test.train()
    # for testing
        model is currently ot savable
        test = model()
        test.train()
        test.test(3) for top 3 or test.test(n) for top n default is 1
    """

    test = model()
    test.train()
    test.test(3)

"""
#testdf = pd.DataFrame(test)

#print(testdf.head())

test_skills = test[test[:,1] == '1']

#test_skills = np.where(test[:,1] == 1, test)
print(test_skills)

train = np.concatenate((train, np.array(test_skills)), axis = 0)

print(train.shape) 



test_jobs = test[test[:,1] == '0']

ids = list(map(str, test_jobs[:,0]))

test_jobs_cloned = np.zeros((test_jobs.shape[0] * 8, test_jobs.shape[1]))

for i in range(8):
    test_jobs_cloned[i*test_jobs.shape[0]:(i+1)*test_jobs.shape[0],0] = ids
    test_jobs_cloned[:,1] = '0'
    test_jobs_cloned[i*test_jobs.shape[0]:(i+1)*test_jobs.shape[0],2] = str(i)

test_jobs_cloned = test_jobs_cloned.astype(int)
test_jobs_cloned = test_jobs_cloned.astype(str)

print(test_jobs_cloned)
#py_test_jobs = test_jobs_cloned.tolist()
#py_test_jobs = list(map(str, py_test_jobs[:,:]))
#print(py_test_jobs)
#print(test_jobs_cloned)

#test_jobs_cloned = filter_unseen_entities(test_jobs_cloned, model)

#print(test_jobs_cloned.shape)

scores = model.predict(test_jobs_cloned)

print(test_jobs[3])
for i in range(8):
    print(scores[i*test_jobs.shape[0] + 3])
    

average = np.mean(scores)

# Calculate the standard deviation
std_deviation = np.std(scores)

print("Average:", average)
print("Standard Deviation:", std_deviation)

"""