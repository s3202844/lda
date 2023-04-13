import pandas as pd


# read dataset0 and dataset1
dataset0 = pd.read_csv("data/dataset0.csv")
dataset1 = pd.read_csv("data/dataset1.csv")
# check if the header of dataset0 and dataset1 are the same
is_same = dataset0.columns.values.tolist() == dataset1.columns.values.tolist()
# if the header is the same, then we can merge dataset0 and dataset1
if is_same:
    dataset = pd.concat([dataset0, dataset1])
    dataset.to_csv("data/dataset.csv", index=False)
# check dataset.csv's header, shape, and first 5 rows
dataset = pd.read_csv("data/dataset.csv")
print(dataset.columns.values.tolist())
print(dataset.shape)
print(dataset.head())
