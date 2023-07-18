import pandas as pd
import numpy as np
import random


df = pd.read_excel("data/Complete Data Clustered Cleaned.xlsx")
df_new = df[df.groupby('Cluster')['Cluster'].transform('size') >= 75].reset_index(drop=True)
print(df_new["Cluster"].value_counts())

values = df_new["Cluster"].value_counts().values
index = df_new["Cluster"].value_counts().index
min_vals = min(values)
l = []
for i in range(len(index)):
    l.append(np.where(df_new["Cluster"] == index[i])[0].tolist())
sample = []
for i in range(len(index)):
    sample.append(random.sample(l[i], min_vals))
test_size = int((min_vals / 100) * 20 )
print(sample)
test = []
for i in range(len(index)):
    test.append(random.sample(sample[i], test_size))

sample = [item for sublist in sample for item in sublist]
test = [item for sublist in test for item in sublist]
sample = list(set(sample) - set(test))

print(list(set(df.index) - set(test)))
df_balanced = df_new.iloc[sample]
df_test = df_new.iloc[test]
df = df_new.iloc[list(set(df_new.index) - set(test))]

df.to_csv("data/Complete_Data_Clustered_Cleaned.csv", index=False)
df_test.to_csv("data/Complete_Data_Clustered_Cleaned_test.csv", index=False)

df = pd.read_csv("data/Complete_Data_Clustered_Cleaned_test.csv")
print(df["Cluster"].value_counts())
df = pd.read_csv("data/Complete_Data_Clustered_Cleaned.csv")
print(df["Cluster"].value_counts())