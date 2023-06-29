from ogb.linkproppred import LinkPropPredDataset, Evaluator
import model
dataset = LinkPropPredDataset(name='ogbl-wikikg2')
print(dataset[0])

edge_index = dataset.graph['edge_index']
feat = dataset.graph["edge_feat"]
rel = dataset.graph["edge_reltype"]

print(edge_index,rel.shape,feat)

kge_model = model.KGEModel(
    model_name="TransE",
    nentity=2500604,
    nrelation= dataset.graph['num_nodes'],
    hidden_dim=500,
    gamma=12.0,
    evaluator=Evaluator("ogbl-wikikg2")
)
print(kge_model)

for name, param in kge_model.named_parameters():
    print(name,param, param.shape)