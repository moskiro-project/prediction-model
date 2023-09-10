import os

import tensorflow as tf

import pandas as pd
import numpy as np
import torch

from Emgraph2.emgraph.datasets import BaseDataset, DatasetType
from Emgraph2.emgraph.models import ComplEx, ConvKB, DistMult, HolE, TransE
from Emgraph2.emgraph.evaluation.protocol import filter_unseen_entities
from util import KG_data

# This needs to be extended to have a proper interface for training and testing! 
# Since this graph doesn't have save+load mechanisms, we would need to keep the object around in the testing/evaluation script!
# Check carefully whether all invalid cases are caught here, e.g. what to do if NER didn't give any results and if everything is formatted in a standardized way

class model():

    def __init__(self, lr=0.0005,epochs=50, batchsize=64, train_file = './data/Complete_Data_Clustered_Cleaned.csv', test_file = './data/Complete_Data_Clustered_Cleaned_test.csv',
                 totalClusters = 20, ground_truth_file = 'data/test_data_graph_org_new.csv'):
        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize

        self.clusters = totalClusters

        self.model = TransE(batches_count=self.batchsize,seed=0,epochs=self.epochs,k=100,
                            eta=10,optimizer="adam",optimizer_params={"lr": self.lr},loss="pairwise",
                            verbose=False,large_graphs=False,)


        tr, te = KG_data(train_file, test_file, train_save = "./data/train_data_graph_new.csv", test_save = "./data/test_data_graph_new.csv", ground_truth_save=ground_truth_file)
        self.train_data = BaseDataset.load_from_csv(os.getcwd(), "./data/train_data_graph_new.csv", ',')
        self.test_data = BaseDataset.load_from_csv(os.getcwd(), "./data/test_data_graph_new.csv", ',')
        self.test_data = self.test_data[1:,]

    def train(self):
        self.model.fit(self.train_data)

    def test(self,topk=1):
        testData = filter_unseen_entities(self.test_data, self.model)
        testDataColumnCount = testData.shape[1] # should be 3?!
        # add one column for results
        output = np.zeros((testData.shape[0], testDataColumnCount + 1))
        output[:,0:testDataColumnCount] = testData
        output[:,testDataColumnCount] = self.model.predict(testData)
        # here we reshape to combine all outputs of one person (should check whether this division is valid!)
        output = output.reshape((int)(testData.shape[0] / self.clusters), self.clusters, testDataColumnCount+1)

        outputTorch = torch.from_numpy(output)
        preds = [(int(x[0][0]), (torch.topk(x[:,3], topk))[1].numpy().tolist()) for x in outputTorch]
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

    print(test.test_data.shape)
    print(test.train_data.shape)
    
    test.train()
    output = test.test(3)
    df = pd.DataFrame(output, columns=["Person", "Top3 Predictions"])
    df.to_csv("GraphResults.csv")

if __name__ == "__main__":
    main()


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