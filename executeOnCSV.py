import heapq
from torch import cdist
from applyNER import apply_NER
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import sys
#import KG_model as graph
import LinkPred

sys.path.append('./Emgraph2/emgraph')
sys.path.append('./Emgraph2')




def main(filename, desc_column_name, selectedModel, topPreds):
    apply_NER(filename, column_name="Skills", descript_column=desc_column_name, out_filename="intermediate.csv")
    df_data = pd.read_csv("intermediate.csv")
    if selectedModel == 0:
        model = Doc2Vec.load("./model/doc2vec_newData")
        predictions_option1 = []
        for skills_example in df_data["Skills"]:
            infer_vector_option1 = model.infer_vector(skills_example)
            similar_jobs_option1 = model.dv.most_similar([infer_vector_option1], topn=topPreds)
            predicted_jobs_option1 = [job for job, similarity in similar_jobs_option1]
            predictions_option1.append(predicted_jobs_option1)

    elif selectedModel == 1:
        model = Word2Vec.load("./model/word2vec_newData")
        predictions_option2 = []
        # cluster_centres = ..
        # keys = range(0, len(cluster_centres))
        # for skills_example in df_data["Skills"]:
        # avg_option2 = averageOfSkills(model, skills_example)
        # distVec_option2 = cdist([avg_option2], cluster_centres)
        # topJobs_option2 = [k for dist, k in heapq.nsmallest(topPreds, zip(distVec_option2.transpose(), keys))]
        # predictions_option2.append(topJobs_option2)
    elif selectedModel == 2:
        model = Word2Vec.load("./model/doc2vec_newData")
        predictions_option2 = []
        # cluster_centres = ..
        # keys = range(0, len(cluster_centres))
        # for skills_example in df_data["Skills"]:
        # avg_option2 = averageOfSkills(model, skills_example)
        # distVec_option2 = cdist([avg_option2], cluster_centres)
        # topJobs_option2 = [k for dist, k in heapq.nsmallest(topPreds, zip(distVec_option2.transpose(), keys))]
        # predictions_option2.append(topJobs_option2)
    elif selectedModel == 3:
        link_pred = model(load_model=True, save_model=False, load_test=True)
        link_pred.test()

        model_out = link_pred(topPreds)
    elif selectedModel == 4:
        graph_model = graph.model(epochs=1, test_file="intermediate.csv", totalClusters=20, ground_truth_file="foo.csv",
                                  write_ground_truth=False)
        graph_model.train()
        model_out = graph_model.test(topPreds)

    return model_out
if __name__ == "__main__":
    main()
