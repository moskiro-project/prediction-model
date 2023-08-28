import spacy
import pandas as pd


nlp_ner = spacy.load('ner_model')

def extract_entities(row):
    doc = nlp_ner(row["Description"])
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities
    
    


def main():
    df = pd.read_excel('./data/NER Input.xlsx')
    print("Start")
    df["Skills/Description"] = df.apply(extract_entities, axis = 1)
    print("End")
    output_file = './data/FullDataset.xlsx'
    df.to_excel(output_file, index=False)

if __name__ == '__main__':
    main()
    #if(len(sys.argv) > 0):
    #    main(sys.argv[0][0])