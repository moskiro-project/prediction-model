import pandas as pd
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from operator import itemgetter

# Read data from file
df = pd.read_csv("naukri_data_science_jobs_india_cleaned.csv", converters={"Skills/Description":pd.eval})
texts = df["Skills/Description"].tolist()

# LDA setup
num_topics= 10
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10, alpha="auto",
                     per_word_topics=True)

#TODO coherence score for checking how good this is :https://bennett-holiday.medium.com/a-step-by-step-guide-to-writing-an-lda-program-in-python-690aa99119ea

# Save assinged Topics with data
topics = lda_model.get_topics()
df["lda_topic"] = [lda_model.get_document_topics(item) for item in corpus]
df["lda_score"]  = df["lda_topic"].apply(lambda x:max(x,key=itemgetter(1))[1])
df["lda_topic"]  = df["lda_topic"].apply(lambda x:max(x,key=itemgetter(1))[0])
df.to_csv("naukri_data_science_jobs_india_cleaned_clusterd.csv")

# TODO check how many topics
# TODO check for names and assing them