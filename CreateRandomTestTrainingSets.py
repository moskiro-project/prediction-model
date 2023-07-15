import pandas as pd
import ast
from sklearn.model_selection import train_test_split

trainDataFinal = []
testDataFinal = []

# Load the Excel file
data = pd.read_csv('naukri_data_science_jobs_india_cleaned_clusterd.csv')

# Split the data into train and test datasets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

for index, row in train_data.iterrows():
    trainDataFinal.append([index, 0, row["lda_topic"]])
    for element in ast.literal_eval(row["Skills/Description"]):
        trainDataFinal.append([index, 1, element])

for index, row in test_data.iterrows():
    testDataFinal.append([index, 0, row["lda_topic"]])
    for element in ast.literal_eval(row["Skills/Description"]):
        testDataFinal.append([index, 1, element])

# Save the train and test datasets to separate Excel files
trainDf = pd.DataFrame(trainDataFinal)
testDf = pd.DataFrame(testDataFinal)
trainDf.to_csv('train_data_graph.csv', index=False)
testDf.to_csv('test_data_graph.csv', index=False)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)