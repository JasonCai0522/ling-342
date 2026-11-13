from datasets import load_dataset
import pandas as pd

def load_data(dataset_url: str, name = ""):
    """
    Purpose: Load a dataset from huggingface and turn into csv
    """
    return load_dataset(dataset_url)

# Dataset
# https://huggingface.co/datasets/knoveleng/emotion-datasets
ds = load_data("knoveleng/emotion-datasets")
print(ds)


for split, dataset in ds.items():
    dataset.to_csv(f"./dataset/{split}.csv", index=False)

df = pd.read_csv("./dataset/train.csv")


# Add code for filtering here
# df_row = df[df['emotion'] == 'angry']


df.to_csv('df1.csv', columns = ['emotion', 'text'], index=False)
df_row_col = pd.read_csv('df1.csv')

# Sampling 200 rows from the dataframe
#df_sample = df_row_col.sample(n=200)

# Sample 

# print(ds)
# row = ds["train"][0]
# print(row["emotion"], "—", row["topic"])
# for i, story in enumerate(row["stories"], 1):   # 3 stories per row
#     print(f"[story {i}] {story[:80]}...")
