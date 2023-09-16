# Moskiro Project - prototype
Prototype to predict job roles based on a provided list of skills

## File Overview
- **Emgraph2** code written by [emgraph team](https://github.com/bi-graph/Emgraph), for the knowledge graph prediction
- **data** folder with the data set files, intermediate and final files
- **model** folder for all model data to reload from disk or save to disk
- **ner_model** spacy ner model folder, due to size not in the model folder

Model files :
- LinkPred.py & KG_model.py
  
Utility files:
- chatGPT.py & util.py
  
Demo and Evaluation:
- demo_prediction.py & Evaluation.ipynb

## Examples of use
Either Evaluation.ipynb or demo_prediction.py provides a quick overview. 
The following code provides an example if the prototype should be used in a pipeline or run from start to finish without any pre-trained data:
```python
    apply_NER(filename, column_name="Skills", descript_column=desc_column_name, out_filename="intermediate.csv")
    df_data = pd.read_csv("intermediate.csv")
    if selectedModel == 'doc2vec':
        model = Doc2Vec.load("./model/doc2vec_newData")
        model_out = []
        for skills_example in df_data["Skills"]:
            similar_jobs = model.dv.most_similar([model.infer_vector(skills_example)], topn=topPreds)
            predicted_jobs = [job for job, similarity in similar_jobs]
            model_out.append(predicted_jobs)
      
    elif selectedModel == 'LinkPrediction':
      link_pred = model(load_model=True, save_model=False, load_test=True)
      link_pred.test()
      model_out = link_pred(topPreds)
      
    elif selectedModel == 'KnowledgeGraph'
      graph_model = graph.model(epochs=1, test_file="intermediate.csv", totalClusters=20, ground_truth_file="foo.csv",
                                  write_ground_truth=False)
      graph_model.train()
      model_out = graph_model.test(topPreds)
```   

### Data
The dataset needs to be in the correct layout for the prototype to work, which is mainly a list of skills and a corresponding job title. Additional functionality found in util.py, chatGPT.py, and the second project [repro](https://github.com/moskiro-project/GPTAPI) can help with the creation of a new dataset.
A lot of column names are currently hardcoded and would need to be replicated for a different dataset.

## Installation & Requirments
All relevant files are present in the git, so only the [required packages](https://github.com/moskiro-project/prediction-model/blob/main/requirements.txt) need to be installed
