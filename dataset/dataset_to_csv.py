from datasets import load_dataset


def load_data(dataset_url: str):
    return load_dataset(dataset_url)

# https://huggingface.co/datasets/knoveleng/emotion-datasets
ds = load_data("knoveleng/emotion-datasets")


for split, dataset in ds.items():
    dataset.to_csv(f"./dataset/{split}.csv", index=False)



# print(ds)
# row = ds["train"][0]
# print(row["emotion"], "—", row["topic"])
# for i, story in enumerate(row["stories"], 1):   # 3 stories per row
#     print(f"[story {i}] {story[:80]}...")
