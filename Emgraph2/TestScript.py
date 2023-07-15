import os

from emgraph.datasets import BaseDataset, DatasetType

from emgraph.models import ComplEx, ConvKB, DistMult, HolE, TransE

from emgraph.evaluation.protocol import filter_unseen_entities

#classes = (ComplEx, HolE, TransE, ConvKB, DistMult)

model = TransE(
    batches_count=64,
    seed=0,
    epochs=0,
    k=100,
    eta=20,
    optimizer="adam",
    optimizer_params={"lr": 0.0001},
    loss="pairwise",
    verbose=True,
    large_graphs=False,
)

path1 = os.path.join(os.getcwd(), "train_data_graph.csv")

train = BaseDataset.load_from_csv(os.getcwd(), "train_data_graph.csv", ',')
test = BaseDataset.load_from_csv(os.getcwd(), "test_data_graph.csv", ',')

model.fit(train)

test = filter_unseen_entities(test, model)

scores = model.predict(test)
print("scores: ", scores)