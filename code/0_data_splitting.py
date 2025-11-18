import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Read dataset
data_raw = pd.read_csv("data/data_raw.csv")


# Split data (84%-16%)
random.seed(311)
stud = data_clean["id"].unique().tolist()
random.shuffle(stud)
train_id, test_id = stud[:231], stud[231:]

data_train = data_clean[data_clean["id"].isin(train_id)]
data_test = data_clean[data_clean["id"].isin(test_id)]


# Write splitted dataset to .csv file
data_train.to_csv("data/data_train.csv", index=False)
data_test.to_csv("data/data_test.csv", index=False)
