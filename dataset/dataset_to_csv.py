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
    dataset.to_csv(f"./dataset/dataset/full_dataset.csv", index=False)

df = pd.read_csv("./dataset/dataset/full_dataset.csv")

df.to_csv("./dataset/dataset/full_dataset.csv", columns = ['emotion', 'text'], index=False)
df_row_col = pd.read_csv("./dataset/dataset/full_dataset.csv")

# Sample 170 rows for training and 75 rows for testing
df_training = df_row_col.sample(n=2000, random_state=42)
df_testing = df_row_col.drop(df_training.index).sample(n=500, random_state=42)

df_training.to_csv('./dataset/dataset/trainingLong.csv', index=False)
df_testing.to_csv('./dataset/dataset/testingLong.csv', index=False)