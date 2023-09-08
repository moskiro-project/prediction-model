import openai
import pandas as pd
import time
import json
import ast

top3_file = "GraphResults.csv"
correct_file = "test_data_graph_org_new.csv"

# This would need to be moved into the Evaluation notebook!

def process_row(row):
    print(row)
    if(row["tail"] in row["Top3 Predictions"]):
        #correctGuesses = correctGuesses + 1
        return row
    return []

def main():
    correctGuesses = 0
    df = pd.read_csv(top3_file, converters={"Top3 Predictions": pd.eval})["Top3 Predictions"].values
    df2 = pd.read_csv(correct_file)["tail"].values
    for i in range(len(df)):
        if df2[i] in df[i]:
            print("Correct guess: " + str(df2[i]) + " in: " + str(df[i]))
            correctGuesses = correctGuesses + 1
    #output = pd.concat([df, df2], axis=1)
    #outputFiltered = output.apply(process_row, axis = 1)
    #print(outputFiltered.shape)
    print(correctGuesses)
    #output.to_csv("CorrectResults.csv")
    
if __name__ == '__main__':
    main()