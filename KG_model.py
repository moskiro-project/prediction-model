import os

import tensorflow as tf

import pandas as pd
import numpy as np
import torch

from Emgraph2.emgraph.datasets import BaseDataset, DatasetType
from Emgraph2.emgraph.models import ComplEx, ConvKB, DistMult, HolE, TransE
from Emgraph2.emgraph.evaluation.protocol import filter_unseen_entities
from util import KG_data


class model:
    """
    Knowledge Graph model class
    """
    def __init__(self, lr=0.0005, epochs=50, batchsize=64, train_file='./data/Complete_Data_Clustered_Cleaned.csv',
                 test_file='./data/Complete_Data_Clustered_Cleaned_test.csv',
                 totalClusters=20, ground_truth_file='data/KG_test_data_graph_org_new.csv', write_ground_truth=True):
        """

        :param lr: learning rate for the model
        :param epochs: number of trainings epochs
        :param batchsize: batchsize for trainings procedure
        :param train_file: trainings file to load
        :param test_file: testfile to load
        :param totalClusters: number of cluster used
        :param ground_truth_file: ground truth file to load
        :param write_ground_truth: boolean, wether to save the ground truth to disk
        """
        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize

        self.clusters = totalClusters

        self.model = TransE(batches_count=self.batchsize, seed=0, epochs=self.epochs, k=100,
                            eta=10, optimizer="adam", optimizer_params={"lr": self.lr}, loss="pairwise",
                            verbose=False, large_graphs=False, )
        # creates the files needed for the emgraph model
        KG_data(train_file, test_file, train_save="./data/KG_train_data_graph_new.csv",
                test_save="./data/KG_test_data_graph_new.csv",
                ground_truth_save=ground_truth_file,
                write_ground_truth=write_ground_truth)

        self.train_data = BaseDataset.load_from_csv(os.getcwd(), "data/KG_train_data_graph_new.csv", ',')
        self.test_data = BaseDataset.load_from_csv(os.getcwd(), "data/KG_test_data_graph_new.csv", ',')
        self.test_data = self.test_data[1:, ]

    def train(self):
        self.model.fit(self.train_data)

    def test(self, topk=1):
        testData = filter_unseen_entities(self.test_data, self.model)
        testDataColumnCount = testData.shape[1]  # should be 3?!
        # add one column for results
        output = np.zeros((testData.shape[0], testDataColumnCount + 1))
        output[:, 0:testDataColumnCount] = testData
        output[:, testDataColumnCount] = self.model.predict(testData)
        # here we reshape to combine all outputs of one person (should check whether this division is valid!)
        output = output.reshape(int(testData.shape[0] / self.clusters), self.clusters, testDataColumnCount + 1)

        outputTorch = torch.from_numpy(output)
        preds = [(int(x[0][0]), (torch.topk(x[:, 3], topk))[1].numpy().tolist()) for x in outputTorch]
        return preds


def main():

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
    output = test.test(3)
    df = pd.DataFrame(output, columns=["Person", "Top3 Predictions"])
    df.to_csv("./data/GraphResults.csv")


if __name__ == "__main__":
    main()
