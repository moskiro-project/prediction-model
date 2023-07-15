import os

import pandas as pd

import numpy as np

from emgraph.datasets import BaseDataset, DatasetType

from emgraph.models import ComplEx, ConvKB, DistMult, HolE, TransE

from emgraph.evaluation.protocol import filter_unseen_entities

#classes = (ComplEx, HolE, TransE, ConvKB, DistMult)

model = TransE(
    batches_count=64,
    seed=0,
    epochs=3,
    k=100,
    eta=10,
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    loss="pairwise",
    verbose=True,
    large_graphs=False,
)

#path1 = os.path.join(os.getcwd(), "train_data_graph.csv")

train = BaseDataset.load_from_csv(os.getcwd(), "train_data_graph.csv", ',')
test = BaseDataset.load_from_csv(os.getcwd(), "test_data_graph.csv", ',')
print(test)

print("Training data shape: " + str(train.shape))

#testdf = pd.DataFrame(test)

#print(testdf.head())

test_skills = test[test[:,1] == '1']

#test_skills = np.where(test[:,1] == 1, test)
print(test_skills)

train = np.concatenate((train, np.array(test_skills)), axis = 0)

print(train.shape) 

model.fit(train)

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