import spacy
import pandas as pd
import sys

nlp_ner = spacy.load('ner_model')

# applies the NER row-wise
def extract_entities(row, descript_column = "Description"):
    doc = nlp_ner(row[descript_column])
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities
    
# call to extract skills using NER on a dataframe
def apply_NER(df, descript_column, output_column_name):
    df[output_column_name] = df.apply(extract_entities, descript_column, axis = 1)
    return df

# native behavior from commandline
def main(filename = './data/NER Input.xlsx', column_name = "Skills/Description"):
    df = pd.read_excel(filename)
    print("Start")
    df[column_name] = df.apply(extract_entities, axis = 1)
    print("End")
    output_file = './data/FullDataset.xlsx'
    df.to_excel(output_file, index=False)

if __name__ == '__main__':
    if(len(sys.argv < 3)):
        main()
    else:
        main(sys.argv[1], sys.argv[2])
    #if(len(sys.argv) > 0):
    #    main(sys.argv[0][0])