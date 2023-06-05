import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_score_distribution(df, topics_list= None):
    if topics_list is None:
        topics_list = df["lda_topic"].unique()
    num_topics = df["lda_topic"].nunique()
    values = df["lda_topic"].value_counts()
    labels = values.index.tolist()

    x = np.arange(0,10,1)
    fig, ax = plt.subplots()
    bar = ax.bar(x,(values.to_numpy()/len(df))* 100,align= "center")
    ax.set_xticks(x,labels)
    ax.set_ylabel("Precentage")
    ax.bar_label(bar, labels= ["%.1f" % value for value in values.to_list()], fmt="%.1f")
    fig.show()
    print(num_topics, topics_list, (values.to_numpy()/len(df))* 100, values)

df = pd.read_csv("naukri_data_science_jobs_india_cleaned_clusterd.csv", converters={"Skills/Description":pd.eval},index_col=0)
#get_score_distribution(df)

tmp = df[df["lda_topic"]==7]
#tmp = tmp["Job_Role"].drop_duplicates()
print(tmp.value_counts(), tmp.nunique())


#0 Cloud Engineer, 1 Analyst,2 Scientist / Analyst, 3 engineer, 4 Analyst, 5 Engineer,6 Analyst,7  Big Data Engineer, 8 lead Engineer,  9 Lead Scientist
#