import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Read dataset
data_raw = pd.read_csv("data/data_raw.csv")


# Basic cleaning
data_clean = data_raw.fillna("")
data_clean.replace("#NAME?", "", inplace=True)
data_clean.replace("[THIS MODEL]", " ", inplace=True)
data_clean.replace("[ANOTHER MODEL]", " ", inplace=True)
data_clean.columns = ["id", "t1", "n1", "c1", "n2", "c2", "t2", "n3", "n4", "t3", "label"]


# Split data (84%-16%)
random.seed(311)
stud = data_clean["id"].unique().tolist()
random.shuffle(stud)
train_id, test_id = stud[:231], stud[231:]

data_train = data_clean[data_clean["id"].isin(train_id)]
data_test = data_clean[data_clean["id"].isin(test_id)]
data = [data_train, data_test]


# Vectorize data
def vec(df: pd.DataFrame, vl: list[CountVectorizer]) -> pd.DataFrame:
    data_vec = df[["id"]].copy()

    # Vectorize `numeric` columns
    for n in range(1, 5):
        val = pd.to_numeric(df[f"n{n}"].str[0], errors="coerce")
        med = str(int(val.median()))
        data_vec[f"n{n}"] = df[f"n{n}"].str[0].fillna(med).astype(int)

    # Vectorize `choice` columns
    option = ["computation", "code", "analysis", "concept", "format", "essay", "text", "idea"]
    for c in range(1, 3):
        for opt in option:
            data_vec[f"c{c}_{opt}"] = df[f"c{c}"].str.contains(opt).astype(int)

    # Vectorize `text` columns
    for t in range(1, 4):
        t_mat = vl[t - 1].transform(df[f"t{t}"])
        col = []
        for word in vl[t - 1].get_feature_names_out():
            col.append(f"t{t}_{word}")
        data_t = pd.DataFrame(t_mat.toarray(), columns=col)
        data_t.index = df.index
        data_vec = pd.concat([data_vec, data_t], axis=1)

    # Categorize `label` column
    data_vec["label"] = pd.factorize(df["label"])[0]

    return data_vec


vs = []
for i in range(1, 4):
    v = CountVectorizer(max_features=3000, binary=True)
    v.fit(data_train[f"t{i}"])
    vs.append(v)

data_vec_train = vec(data_train, vs)
data_vec_test = vec(data_test, vs)


# Write cleaned dataset to .csv file
data_vec_train.to_csv("data/data_train.csv", index=False)
data_vec_test.to_csv("data/data_test.csv", index=False)
