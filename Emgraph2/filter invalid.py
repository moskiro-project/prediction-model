import openai
import pandas as pd
import time
import json
import ast

cluster_file = "Graph_Results_Sorted.csv"


def main():
    df = pd.read_csv(cluster_file)
    prev = int(df.iloc[0,0]) - 1
    for head, _, _ in table:
        if(int(head) != prev+1):
            print(triple)


    
if __name__ == '__main__':
    main()