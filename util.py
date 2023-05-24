import pandas as pd

def helper_clean(column):
    column = column.split(",")
    for i in range(len(column)):
        column[i] = column[i].lower().strip().replace(" ", "_")
    return column

df_data = pd.read_csv("naukri_data_science_jobs_india.csv")
df_data = df_data.drop(columns=['Company', 'Location', 'Job Experience'])
df_data["Skills/Description"] =df_data["Skills/Description"].apply(lambda x : helper_clean(x))
df_data.to_csv("naukri_data_science_jobs_india_cleaned.csv",index=False)