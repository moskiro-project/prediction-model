import numpy
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import torch
import matplotlib.pyplot as plt
import random


def create_edgeindex(emb, df, undirected=False):
    """
    Creates the edge inde for the link prediction model
    :param emb: feature embbeding
    :param df: dataframe of the data set
    :param undirected: boolean, decides if the graph should be directed or undirected
    :return: selected embbedings, edgeindex
    """
    src, tar = [], []

    avg = torch.zeros((8, 100))
    for i in range(0, 8):
        idx = df[df["newCluster"] == i].index
        avg[i] = torch.from_numpy(np.mean(emb[idx]))

        src += idx.to_list()
        tar += [i] * len(idx)

    if undirected:
        edge_index = torch.vstack((torch.asarray(src + tar), torch.asarray(tar + src)))
    else:
        edge_index = torch.vstack((torch.asarray(src), torch.asarray(tar)))

    emb = torch.cat((avg, torch.from_numpy(np.stack(emb.values))))
    return emb, edge_index


def create_dataset(doc2vec=True, test=True):
    """
    Function to read the input file for the link prediction model
    :param doc2vec: boolean, decides which model to load
    :param test: boolean, true if test set should be loaded additionally
    :return: features, edgeindex, ground truth
    """
    if doc2vec:
        model = Doc2Vec.load("model/doc2vec_newData")
    else:
        model = Word2Vec.load("model/word2vec_newData")
    df = pd.read_csv("data/Complete_Data_Clustered_Cleaned.csv", converters={"Skills/Description": pd.eval})

    embbedding = df["Skills/Description"].apply(lambda w: model.infer_vector(w))
    x, edge_index = create_edgeindex(embbedding, df)
    y = torch.from_numpy(df["newCluster"].to_numpy())

    if test:
        df2 = pd.read_csv("data/Complete_Data_Clustered_Cleaned_test.csv",
                          converters={"Skills/Description": pd.eval})
        embbedding = df2["Skills/Description"].apply(lambda w: model.infer_vector(w))
        x = torch.cat((x, torch.from_numpy(np.stack(embbedding.values))))
        y = torch.from_numpy(df2["newCluster"].to_numpy())
        # print(y,y.shape)
    # print(edge_index)
    return x, edge_index, y


def plot_curves(epochs, curves, title, file_name="errors.pdf", combined=False):
    """
    Plot function for the model error
    :param epochs: number of trainings epochs
    :param curves: the amount of diffrent curves
    :param title: title to plot on graph
    :param file_name: where to save the plot
    :param combined: Boolean, parameter to plot all graph in one plot or one seperate
    :return: None, saves plot to disk
    """
    # we assume all curves have the same length
    # if we use combined we also assume that loss is always the last
    if combined:
        fig, (axs, ax2) = plt.subplots(1, 2, sharex="all")
        ax2.grid(True)
    else:
        fig, axs = plt.subplots()

    x = np.arange(1, len(curves[0]) + 1)
    colors = ["mediumslateblue", "plum", "mediumslateblue"]
    for i in range(len(curves)):
        if i == len(curves) - 1 and combined:  # last elem
            ax2.plot(x, curves[i], color=colors[i])

        else:
            axs.plot(x, curves[i], color=colors[i])
            axs.legend()

    fig.suptitle(title)
    axs.grid(True)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlim([1, epochs])
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(file_name + ".jpg")
    plt.show()


def KG_data(train_data_file='./data/Complete_Data_Clustered_Cleaned.csv',
            test_data_file='./data/Complete_Data_Clustered_Cleaned_test.csv',
            train_skill_column="NewSkills_lowercase", test_skill_column="NewSkills_lowercase",
            train_save='data/KG_train_data_graph_new.csv', test_save='data/KG_test_data_graph_new.csv',
            ground_truth_save='data/KG_test_data_graph_org_new.csv', write_ground_truth=True,
            totalClusters=20):
    """
    Function to create the graph necessary for the KG model
    :param train_data_file: trainings file to process
    :param test_data_file: test file to process
    :param train_skill_column: column name to process
    :param test_skill_column: column name to process
    :param train_save: file name for the new train graph
    :param test_save: file name for the new test graph
    :param ground_truth_save: file name for the ground truth
    :param write_ground_truth: Boolean, wether to save the ground truth or not
    :param totalClusters: number of cluster
    :return: test and trainings dataframe
    """
    # Load the Excel file

    data = pd.read_csv(train_data_file,
                       converters={train_skill_column: pd.eval})
    test = pd.read_csv(test_data_file,
                       converters={test_skill_column: pd.eval})

    # Train data
    trainDataFinal = []
    for index, row in data.iterrows():

        trainDataFinal.append([index + totalClusters, 0, row["newCluster"]])
        for element in row[train_skill_column]:
            trainDataFinal.append([index + totalClusters, 1, element])

    # Test data --> we only want to predicti te role so skills are added to train
    testDataFinal = []
    for index, row in test.iterrows():
        if write_ground_truth:
            testDataFinal.append([index + len(data) + totalClusters, 0, row["newCluster"]])
        else:
            testDataFinal.append([index + len(data) + totalClusters, 0, 0])
        for element in row[test_skill_column]:
            trainDataFinal.append([index + len(data) + totalClusters, 1, element])

    # Save the train and test datasets to separate Excel files
    trainDf = pd.DataFrame(trainDataFinal, columns=["head", "relation", "tail"])
    testDf_org = pd.DataFrame(testDataFinal, columns=["head", "relation", "tail"])
    # testDf = pd.DataFrame(test_data_preprocessed, columns=["head", "relation", "tail"])

    # Add other categories in for ranking

    ids = testDf_org["head"]
    test_data_preprocessed = np.zeros((testDf_org.shape[0] * totalClusters, testDf_org.shape[1]))

    for i in range(20):
        test_data_preprocessed[i * testDf_org.shape[0]:(i + 1) * testDf_org.shape[0], 0] = ids
        test_data_preprocessed[:, 1] = '0'
        test_data_preprocessed[i * testDf_org.shape[0]:(i + 1) * testDf_org.shape[0], 2] = str(i)

    testDf = pd.DataFrame(test_data_preprocessed, columns=["head", "relation", "tail"])

    testDf = testDf.sort_values(by=['head', 'tail'], ascending=True)
    testDf["head"] = testDf["head"].astype(int).astype(str)
    testDf["relation"] = testDf["relation"].astype(int).astype(str)
    testDf["tail"] = testDf["tail"].astype(int).astype(str)

    trainDf.to_csv(train_save, index=False)
    testDf.to_csv(test_save, index=False)
    if write_ground_truth:
        testDf_org.to_csv(ground_truth_save, index=False)

    return trainDf, testDf


def split_data(file="data/Complete Data Clustered Cleaned.xlsx", verbose=False):
    """
    Function to create the train and test split for the prediction models
    :param file: input file to be split into a test and train set
    :param verbose: parameter to change feedback level
    :return: None, saves to files (train set & test set ) to disk
    """

    # read input and selction of relevant cluster
    df = pd.read_excel(file)
    df_new = df[df.groupby('Cluster')['Cluster'].transform('size') >= 75].reset_index(drop=True)
    if verbose: print(df_new["Description"].value_counts())

    # find minum number of samples for test set
    values = df_new["Cluster"].value_counts().values
    index = df_new["Cluster"].value_counts().index
    min_vals = min(values)

    # here the sampling happens
    l = []
    for i in range(len(index)):
        l.append(np.where(df_new["Cluster"] == index[i])[0].tolist())
    sample = []
    for i in range(len(index)):
        sample.append(random.sample(l[i], min_vals))
    test_size = int((min_vals / 100) * 20)
    if verbose: print(sample)
    test = []
    for i in range(len(index)):
        test.append(random.sample(sample[i], test_size))

    # renumber the cluster for the prediction models
    idx = df_new["Cluster"].value_counts().index.tolist()
    df_new["newCluster"] = ""
    n = 0
    for i in idx:
        df_new.loc[df_new["Cluster"] == i, "newCluster"] = n
        n += 1

    test = [item for sublist in test for item in sublist]
    if verbose: print(df_new["newCluster"].value_counts(), "\n", list(set(df.index) - set(test)))
    df_test = df_new.iloc[test]
    df = df_new.iloc[list(set(df_new.index) - set(test))]

    df.to_csv("data/Complete_Data_Clustered_Cleaned.csv", index=False)
    df_test.to_csv("data/Complete_Data_Clustered_Cleaned_test.csv", index=False)

# applies the NER row-wise
def extract_entities(row, descript_column="Description"):
    nlp_ner = spacy.load('ner_model_new')
    print(row)
    doc = nlp_ner(row[descript_column])
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities


# call to extract skills using NER on a dataframe
def apply_NER(in_filename='./data/Complete_Data_Clustered_Cleaned.csv', column_name="Skills/Description",
              descript_column="Description", out_filename='./data/Complete_Data_Clustered_Cleaned.csv'):
    df = pd.read_csv(in_filename)
    print("start")
    df[column_name] = df.apply(extract_entities, axis=1)
    print("end")
    df.to_csv(out_filename, index=False)

if __name__ == '__main__':
     create_dataset(True)
    #KG_data()