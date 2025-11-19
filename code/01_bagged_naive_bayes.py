import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import CategoricalNB


# ---------------------------------------------------------------------
# 1. Data loading and preprocessing (copied from naive_bayes.ipynb)
# ---------------------------------------------------------------------

# +
def load_clean(path_raw: str):
    """
    Load the raw dataset, return the cleaned dataset.
    """

    # Read dataset
    data_raw = pd.read_csv(path_raw)

    # Basic cleaning
    data_clean = data_raw.fillna("")
    data_clean.replace("#NAME?", "", inplace=True)
    data_clean.replace("[THIS MODEL]", " ", inplace=True)
    data_clean.replace("[ANOTHER MODEL]", " ", inplace=True)
    data_clean.columns = [
        "id", "t1", "n1", "c1", "n2",
        "c2", "t2", "n3", "n4", "t3", "label"
    ]
    
    return data_clean


def vec(df: pd.DataFrame, vl: list[CountVectorizer]):
    """
    Vectorize all columns in the dataset.
    """
    data_vec = df[["id"]].copy()

    # Vectorize `numeric` columns
    for n in range(1, 5):
        val = pd.to_numeric(df[f"n{n}"].str[0], errors="coerce")
        med = str(int(val.median()))
        data_vec[f"n{n}"] = df[f"n{n}"].str[0].fillna(med).astype(int)

    # Vectorize `choice` columns
    option = ["computation", "code", "analysis", "concept",
              "format", "essay", "text", "idea"]
    
    for c in range(1, 3):
        for opt in option:
            data_vec[f"c{c}_{opt}"] = df[f"c{c}"].str.contains(opt).astype(int)

    # Vectorize `text` columns using a bag-of-words representation
    for t in range(1, 4):
        t_mat = vl[t - 1].transform(df[f"t{t}"])
        col = []
        for word in vl[t - 1].get_feature_names_out():
            col.append(f"t{t}_{word}")
        data_t = pd.DataFrame(t_mat.toarray(), columns=col)
        data_t.index = df.index
        data_vec = pd.concat([data_vec, data_t], axis=1)

    # Convert labels to {0, 1, 2}
    data_vec["label"] = pd.factorize(df["label"])[0]

    return data_vec


# -

# ---------------------------------------------------------------------
# 2. Train and validate the bagging Naive Bayes model
# ---------------------------------------------------------------------

def train_valid_bagged_nb(X: np.ndarray, t: np.ndarray, groups: np.ndarray, n: int):
    """
    Grid search for Bagged Naive Bayes.
    """
    # Create Bagged Naive Bayes model
    categ = [np.unique(X[:, i]).shape[0] for i in range(X.shape[1])]
    bagged_nb = BaggingClassifier(
        estimator=CategoricalNB(min_categories=categ),
        random_state=311
    )
    
    # Create the parameter grid to search on
    param_grid = {
        "n_estimators": [10, 30, 50, 100],
        "estimator__alpha": [0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    # Train and validate the model
    grid = GridSearchCV(
        estimator=bagged_nb,
        param_grid=param_grid,
        cv=GroupKFold(n_splits=n),
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X, t, groups=groups)
    
    # Print the result
    print("=== Bagged Naive Bayes ===")
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


# ---------------------------------------------------------------------
# 3. Putting everything together: evaluation script
# ---------------------------------------------------------------------

def main():
    # 1. Load and vectorize data
    train = load_clean("data/data_train.csv")
    test = load_clean("data/data_test.csv")
    
    vs = []
    for i in range(1, 4):
        v = CountVectorizer(max_features=3000, binary=True)
        v.fit(train[f"t{i}"])
        vs.append(v)
    
    train_vec = vec(train, vs)
    test_vec = vec(test, vs)
    
    ids = train_vec['id'].values
    X_train = train_vec.drop(["id", "label"], axis=1).values
    t_train = train_vec["label"].values
    X_test = test_vec.drop(["id", "label"], axis=1).values
    t_test = test_vec["label"].values


    # 2. Bagged Naive Bayes model
    best_model = train_valid_bagged_nb(X_train, t_train, ids, 7)
    y = best_model.predict(X_test)
    accuracy = accuracy_score(t_test, y)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(t_test, y))
    print("\nClassification Report:")
    print(classification_report(t_test, y))

if __name__ == "__main__":
    main()
