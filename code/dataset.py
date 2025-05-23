from datasets import load_dataset

#dataset = load_dataset("imdb")
ds = load_dataset("wangrongsheng/ag_news")

"""
with open("data/train.txt", "w", encoding="utf-8") as f:
    for row in dataset["train"]:
        f.write(f"{row['label']}\t{row['text']}\n")
"""

with open("data/train_news.txt","w",  encoding="utf-8") as f:
    for row in ds["train"]:
        f.write(f"{row['label']}\t{row['text']}\n")



print("Path to dataset files:", path)

"""
ds is a DatasetDict object from the Hugging Face datasets library.
Specifically, load_dataset("wangrongsheng/ag_news") returns a DatasetDict containing multiple splits (e.g., train, test, etc.).
You can access each split by key, like ds["train"], which is a Dataset object.
(the term split refers to the partitioning of a dataset into different subsets)

row is a Python dictionary (dict) representing one record (or “row”) in the train split.
For this dataset, each row likely has keys "label" and "text", so something like:

"""