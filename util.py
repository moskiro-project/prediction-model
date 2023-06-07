import numpy
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import torch

def helper_clean(column):
    column = column.split(",")
    for i in range(len(column)):
        column[i] = column[i].lower().strip().replace(" ", "_")
    return column

df_data = pd.read_csv("data/naukri_data_science_jobs_india.csv")
df_data = df_data.drop(columns=['Company', 'Location', 'Job Experience'])
df_data["Skills/Description"] =df_data["Skills/Description"].apply(lambda x : helper_clean(x))
df_data.to_csv("naukri_data_science_jobs_india_cleaned.csv",index=False)

def create_edgeindex(emb,df,undirected=False):
    src,tar = [],[]
    avg = torch.zeros((8,128))
    for i in range(0,8):
        idx = df[df["lda_topic"] == i].index
        avg[i]= torch.from_numpy(np.mean(emb[idx]))
        src += idx.to_list()
        tar += [i]*len(idx)

    if undirected:
        edge_index = torch.vstack((torch.asarray(src+tar),torch.asarray(tar+src)))
    else: edge_index = torch.vstack((torch.asarray(src),torch.asarray(tar)))

    emb = torch.cat((avg,torch.from_numpy(np.stack(emb.values))))
    return emb,edge_index


def create_dataset(doc2vec=True,train=True):
    if doc2vec :
        model = Doc2Vec.load("model/doc2vec")
    else: model = Word2Vec.load("model/word2vec")

    if train:
        df= pd.read_csv("data/naukri_data_science_jobs_india_cleaned_clusterd.csv",converters={"Skills/Description": pd.eval})
    else : df= pd.read_csv("data/naukri_data_science_jobs_india_cleaned_clusterd_test.csv",converters={"Skills/Description": pd.eval})

    # Todo for word2vec
    #Todo incorperate test
    embbedding = df["Skills/Description"].apply(lambda x: model.infer_vector(x))
    print(type(embbedding),type(embbedding[0]))
    return create_edgeindex(embbedding,df),df["lda_topic"].to_numpy()

if __name__ == '__main__':
    create_dataset(True)