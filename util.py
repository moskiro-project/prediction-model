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


def create_dataset(doc2vec=True,test=True):
    if doc2vec :
        model = Doc2Vec.load("model/doc2vec")
    else: model = Word2Vec.load("model/word2vec")


    df= pd.read_csv("data/naukri_data_science_jobs_india_cleaned_clusterd.csv",converters={"Skills/Description": pd.eval})

    # Todo for word2vec
    #Todo incorperate test
    embbedding = df["Skills/Description"].apply(lambda x: model.infer_vector(x))
    x, edge_index = create_edgeindex(embbedding, df)
    y = torch.from_numpy(df["lda_topic"].to_numpy())

    if test:
        df2 = pd.read_csv("data/naukri_data_science_jobs_india_cleaned_clusterd_test.csv",
                         converters={"Skills/Description": pd.eval})
        embbedding = df2["Skills/Description"].apply(lambda x: model.infer_vector(x))
        x = torch.cat((x, torch.from_numpy(np.stack(embbedding.values))))
        y = torch.from_numpy(df2["lda_topic"].to_numpy())
        print(y,y.shape)

    return x,edge_index,y

def plot_curves(epochs, curves, labels, title, file_name="errors.pdf", combined=True):
    # we assume all curves have the same length
    # if we use combined we also assume that loss is always the last
    if combined:
        fig, (axs, ax2) = plt.subplots(1, 2, sharex="all")
        ax2.grid(True)
    else:
        fig, axs = plt.subplots()

    x = np.arange(0, epochs)

    colors = ["mediumslateblue", "plum", "mediumslateblue"]
    for i in range(len(curves)):
        if i == len(curves) - 1 and combined:  # last elem
            ax2.plot(x, curves[i], label=labels[i], color=colors[i])

        else:
            axs.plot(x, curves[i], label=labels[i], color=colors[i])
            axs.legend()

    fig.suptitle(title)
    axs.grid(True)
    plt.xlim([0, epochs + 1])
    plt.subplots_adjust(wspace=0.4)
    plt.legend()
    plt.savefig("plots/" + file_name + ".svg")
    plt.show()

if __name__ == '__main__':
    create_dataset(True)