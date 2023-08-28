import numpy
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import torch
import matplotlib.pyplot as plt


def helper_clean(column):
    column = column.split(",")
    for i in range(len(column)):
        column[i] = column[i].lower().strip().replace(" ", "_")
    return column


df_data = pd.read_csv("data/naukri_data_science_jobs_india.csv")
df_data = df_data.drop(columns=['Company', 'Location', 'Job Experience'])
df_data["Skills/Description"] = df_data["Skills/Description"].apply(lambda x: helper_clean(x))
df_data.to_csv("naukri_data_science_jobs_india_cleaned.csv", index=False)



def create_edgeindex(emb,df,undirected=False):
    src,tar = [],[]
    avg = torch.zeros((20,100))
    for i in range(0,20):
        idx = df[df["newCluster"] == i].index
        avg[i]= torch.from_numpy(np.mean(emb[idx]))

        src += idx.to_list()
        tar += [i] * len(idx)

    if undirected:
        edge_index = torch.vstack((torch.asarray(src + tar), torch.asarray(tar + src)))
    else:
        edge_index = torch.vstack((torch.asarray(src), torch.asarray(tar)))

    emb = torch.cat((avg, torch.from_numpy(np.stack(emb.values))))
    return emb, edge_index

def KGE_edgeindex(df,emb):
    skills = df["Skills/Description"].explode().unique().tolist()
    nodes = np.arange(len(df))


def create_dataset(doc2vec=True,test=True):
    if doc2vec :
        model = Doc2Vec.load("model/doc2vec_newData")
    else: model = Word2Vec.load("model/word2vec_newData")


    rel_type += [0]*len(src) #translates to has_skil


    df= pd.read_csv("data/Complete_Data_Clustered_Cleaned.csv",converters={"Skills/Description": pd.eval})



    # Todo for word2vec
    # Todo incorperate test
    embbedding = df["Skills/Description"].apply(lambda x: model.infer_vector(x))
    x, edge_index = create_edgeindex(embbedding, df)
    y = torch.from_numpy(df["newCluster"].to_numpy())

    if test:

        df2 = pd.read_csv("data/Complete_Data_Clustered_Cleaned_test.csv",
                         converters={"Skills/Description": pd.eval})
        embbedding = df2["Skills/Description"].apply(lambda x: model.infer_vector(x))
        x = torch.cat((x, torch.from_numpy(np.stack(embbedding.values))))
        y = torch.from_numpy(df2["newCluster"].to_numpy())
        print(y,y.shape)
    print(edge_index)
    return x,edge_index,y


def plot_curves(epochs, curves, labels, title, file_name="errors.pdf", combined=False):
    # we assume all curves have the same length
    # if we use combined we also assume that loss is always the last
    if combined:
        fig, (axs, ax2) = plt.subplots(1, 2, sharex="all")
        ax2.grid(True)
    else:
        fig, axs = plt.subplots()

    x = np.arange(1, len(curves[0])+1)
    #TODO somehow we still have the legend thingy in there
    colors = ["mediumslateblue", "plum", "mediumslateblue"]
    for i in range(len(curves)):
        if i == len(curves) - 1 and combined:  # last elem
            ax2.plot(x, curves[i], color=colors[i])

        else:
            axs.plot(x, curves[i],  color=colors[i])
            axs.legend()

    fig.suptitle(title)
    axs.grid(True)
    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    plt.xlim([1, epochs])
    plt.subplots_adjust(wspace=0.4)
    #plt.legend()
    plt.savefig("plots/" + file_name + ".jpg")
    plt.show()

def KG_data():
    # Load the Excel file
    data = pd.read_csv('naukri_data_science_jobs_india_cleaned_clusterd.csv',
                       converters={"Skills/Description": pd.eval})
    test = pd.read_csv('naukri_data_science_jobs_india_cleaned_clusterd_test.csv',
                       converters={"Skills/Description": pd.eval})

    # Train data
    trainDataFinal = []
    for index, row in data.iterrows():
        trainDataFinal.append([index, 0, row["lda_topic"]])
        for element in row["Skills/Description"]:
            trainDataFinal.append([index, 1, element])

    # Test data --> we only want to predicti te role so skills are added to train
    testDataFinal = []
    for index, row in test.iterrows():
        testDataFinal.append([index+len(data), 0, row["lda_topic"]])
        for element in row["Skills/Description"]:
            trainDataFinal.append([index+len(data), 1, element])

    # Save the train and test datasets to separate Excel files
    trainDf = pd.DataFrame(trainDataFinal, columns=["head", "relation", "tail"])
    testDf_org = pd.DataFrame(testDataFinal, columns=["head", "relation", "tail"])
    #testDf = pd.DataFrame(test_data_preprocessed, columns=["head", "relation", "tail"])


    # Add other categories in for ranking
    ids = testDf_org.index
    test_data_preprocessed = np.zeros((testDf_org.shape[0] * 8, testDf_org.shape[1]))

    for i in range(8):
        test_data_preprocessed[i * testDf_org.shape[0]:(i + 1) * testDf_org.shape[0], 0] = ids
        test_data_preprocessed[:, 1] = '0'
        test_data_preprocessed[i * testDf_org.shape[0]:(i + 1) * testDf_org.shape[0], 2] = str(i)

    testDf = pd.DataFrame(test_data_preprocessed,columns=["head","relation","tail"])


    trainDf.to_csv('data/train_data_graph.csv', index=False)
    testDf.to_csv('data/test_data_graph.csv', index=False)
    testDf_org.to_csv('data/test_data_graph_org.csv', index=False)
    #train_data.to_csv('train_data.csv', index=False)
    #test_data.to_csv('test_data.csv', index=False)

if __name__ == '__main__':
    create_dataset(True)
    KDE_data()

