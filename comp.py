import pickle
import pandas as pd
def helper(column):
    return str(column).split(",")

def helper_clean(column):
    print(column)
    for i in range(len(column)):
        column[i]= column[i].lower().strip().replace(" ","_")
    return column
df = pd.read_csv("musthave_dice_naruki_100d.model",sep=" ",header=None,skiprows=[0])
df_em = df.loc[:,0]

df_data = pd.read_csv("data/naukri_data_science_jobs_india.csv")
df_data = df_data.drop(columns=['Company', 'Location', 'Job Experience'])
df_data["Skills/Description"] =df_data["Skills/Description"].apply(lambda x : helper(x))

df_data["Skills/Description"]  = df_data["Skills/Description"].apply(lambda x : helper_clean(x))
df_data.to_csv("naukri_data_science_jobs_india_cleaned.csv",index=False)


print(df_data.head())
print(df_data.values.tolist())

df_data =df_data.explode("Skills/Description")
df_data = set(df_data.drop_duplicates())
df_em = set(df_em)
tmp = df_data-df_em
print(tmp)
