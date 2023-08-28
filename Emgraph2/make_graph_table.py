import openai
import pandas as pd
import time
import json
import ast

top3_file = "GraphResults.csv"
correct_file = "test_data_graph_org_new.csv"

def main():
    df = pd.read_csv(top3_file, converters={"Top3 Predictions": pd.eval})
    df2 = pd.read_csv(correct_file)
    output = pd.concat([df, df2], axis=0)
    #output = []
    #j = 0
    #for head, relation, tail in correct_file:
    #    output.append([head, tail, df2])
    output_file = 'Graph_Results_Finished.csv'
    output.to_csv(output_file, index=False)
    
if __name__ == '__main__':
    main()