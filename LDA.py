import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from operator import itemgetter
from gensim.models import CoherenceModel
import random


# Todo make nicer plot function
def plot_lda(data, points):
    data = [x[0] for x in data]
    fig = plt.figure()  # Create matplotlib figure

    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax.bar(len(data), data, width=0.4)
    ax.set_xlabel(points)
    ax.set_ylabel('coherence score')

    plt.savefig("coherence.pdf")
    plt.show()


def find_lda(id2word, corpus):
    max_cohe = (0, -1)
    var = np.arange(3, 13).tolist()
    var = [1] + var + [15, 20, 30, 50]
    res = [] * len(var)

    for i in var:
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=i, random_state=42, passes=10,
                             alpha="auto",
                             per_word_topics=True)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence="c_v")
        coherence_lda = coherence_model_lda.get_coherence()

        if coherence_lda > max_cohe[0]:
            max_cohe = (coherence_lda, i)
            lda_cohe = lda_model
        res.append((lda_model.log_perplexity(corpus), coherence_lda))

    return (max_cohe, lda_cohe), var, res


def save_lda(id2word, corpus, num_topics, df):
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10,
                         alpha="auto",
                         per_word_topics=True)
    #TODO save LDA to disk
    #   lda.save(temp_file)
    df["lda_topic"] = [lda_model.get_document_topics(item) for item in corpus]
    df["lda_score"] = df["lda_topic"].apply(lambda x: max(x, key=itemgetter(1))[1])
    df["lda_topic"] = df["lda_topic"].apply(lambda x: max(x, key=itemgetter(1))[0])
    df.to_csv("naukri_data_science_jobs_india_cleaned_clusterd.csv", index=False)


def split_data(df, ratio=15):
    values = df["lda_topic"].value_counts().values
    index = df["lda_topic"].value_counts().index
    min_vals = min(values)
    print(values, index)
    l = []
    for i in range(len(index)):
        l.append(np.where(df["lda_topic"] == i)[0].tolist())
    sample = []
    for i in range(len(index)):
        sample.append(random.sample(l[i], min_vals))
    test_size = (min_vals // ratio)
    test = []
    for i in range(len(index)):
        test.append(random.sample(sample[i], test_size))

    sample = [item for sublist in sample for item in sublist]
    test = [item for sublist in test for item in sublist]
    sample = list(set(sample) - set(test))

    df_balanced = df.iloc[sample]
    df_test = df.iloc[test]
    df = df.iloc[list(set(df.index) - set(test))]

    df_balanced.to_csv("naukri_data_science_jobs_india_cleaned_clusterd_balanced.csv", index=False)
    df_test.to_csv("naukri_data_science_jobs_india_cleaned_clusterd_test.csv", index=False)
    df.to_csv("naukri_data_science_jobs_india_cleaned_clusterd.csv", index=False)


if __name__ == '__main__':
    # Read data from file

    df = pd.read_csv("data/naukri_data_science_jobs_india_cleaned.csv",
                     converters={"Skills/Description": pd.eval})
    texts = df["Skills/Description"].tolist()

    # LDA setup
    num_topics = 9
    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]

    coherence, num_topics, results = find_lda(id2word, corpus)
    save_lda(coherence[1], df, "cohe")

    plot_lda(results, num_topics)

    # save_lda(id2word,corpus,8,df)

    df = pd.read_csv("data/naukri_data_science_jobs_india_cleaned_clusterd.csv",
                     converters={"Skills/Description": pd.eval})
    split_data(df)
