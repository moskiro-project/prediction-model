import torch
import numpy as np
import sys
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import spacy

import LinkPred

example_text = """Bachelor or Master Degree in IT, natural sciences, engineering or comparable
    High competencies in data exploration/analysis and machine learning
    Good programming skills in Python, R or Matlab necessary
    Data related expertise in SQL or MS Azure preferred
    Good experience with GitLab, DevOps, CI/CD pipelines or Docker desirable
    Good knowledge of PowerBi or Tableau is a plus
    Wind industry experience preferred"""

def extract_entities(text):
    nlp_ner = spacy.load('ner_model')
    doc = nlp_ner(text)
    
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities

def main(text = "", doc2vec = True):
    print("Example Text: " + example_text)
    input("")
    result = extract_entities(example_text)
    print("NER Result: " + str(result))
    input("")
    #example result for testing
    #result = ["statistics", "feature_engineering", "scala", "aws"]
    pred_model = LinkPred.model(load_model=True,save_model=False,load_test=False)
    if doc2vec :
        model = Doc2Vec.load("model/doc2vec")
    else: model = Word2Vec.load("model/word2vec")
    embedding = model.infer_vector(result)
    print("Embedding result: " + str(embedding))
    input("")
    #print(embedding.shape)
    embedding = embedding.reshape((1, 128))
    #print(embedding.shape)
    pred_model.x = torch.cat((pred_model.x, torch.from_numpy(embedding)))
    pred_model.y = np.asarray([0])
    final_prediction = pred_model.test(topk=3)
    print("Final Prediction: " + str(final_prediction))

if __name__ == '__main__':
    main()
    #if(len(sys.argv) > 0):
    #    main(sys.argv[0][0])
