import spacy
import pandas as pd
import sys

nlp_ner = spacy.load('ner_model_new')

# applies the NER row-wise
def extract_entities(row, descript_column = "Description"):
    doc = nlp_ner(row[descript_column])
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities
    
# call to extract skills using NER on a dataframe
def apply_NER(in_filename = './data/Complete_Data_Clustered_Cleaned.csv', column_name = "Skills/Description", descript_column = "Description", out_filename = './data/Complete_Data_Clustered_Cleaned.csv'):
    df = pd.read_csv(in_filename)
    print("start")
    df[column_name] = df.apply(extract_entities, descript_column, axis = 1)
    print("end")
    df.to_csv(out_filename, index=False)
    #return df

# native behavior from commandline
def main(filename = './data/NER Input.xlsx', column_name = "Skills/Description"):
    df = pd.read_excel(filename)
    print("Start")
    df[column_name] = df.apply(extract_entities, axis = 1)
    print("End")
    output_file = './data/FullDataset.xlsx'
    df.to_excel(output_file, index=False)

if __name__ == '__main__':
    if(len(sys.argv) < 3):
        apply_NER()
    else:
        main(sys.argv[1], sys.argv[2])
    #if(len(sys.argv) > 0):
    #    main(sys.argv[0][0])