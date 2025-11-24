import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_name = "../data/training_data_clean.csv"


def process_multiselect(series, target_tasks):
    """
    Convert multiselect strings to lists, keeping only tasks from target_tasks.
    """
    processed = []
    for response in series:
        if pd.isna(response) or response == "":
            processed.append([])
        else:
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from '3 - Sometimes' style strings.
    If no leading number, return 0.
    """
    m = re.match(r"^(\d+)", str(response))
    return int(m.group(1)) if m else 0


def build_features(df_train, df_test):
    """
    - multi-select → MultiLabelBinarizer
    - rating → MinMaxScaler + PolynomialFeatures(degree=3)
    """
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    target_tasks = []
    for multiselect in df_train[
        "Which types of tasks do you feel this model handles best? (Select all that apply.)"
    ]:
        multiselect = str(multiselect)
        tasks = multiselect.split(",")
        for task in tasks:
            task = task.strip()
            if task and task not in target_tasks:
                target_tasks.append(task)

    # train side
    best_tasks_train = process_multiselect(
        df_train[
            "Which types of tasks do you feel this model handles best? (Select all that apply.)"
        ],
        target_tasks,
    )
    subopt_tasks_train = process_multiselect(
        df_train[
            "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
        ],
        target_tasks,
    )

    mlb_best = MultiLabelBinarizer()
    mlb_sub = MultiLabelBinarizer()
    best_tasks_encoded_train = mlb_best.fit_transform(best_tasks_train)
    subopt_tasks_encoded_train = mlb_sub.fit_transform(subopt_tasks_train)

    # test side
    best_tasks_test = process_multiselect(
        df_test[
            "Which types of tasks do you feel this model handles best? (Select all that apply.)"
        ],
        target_tasks,
    )
    subopt_tasks_test = process_multiselect(
        df_test[
            "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
        ],
        target_tasks,
    )

    best_tasks_encoded_test = mlb_best.transform(best_tasks_test)
    subopt_tasks_encoded_test = mlb_sub.transform(subopt_tasks_test)

    scaler = MinMaxScaler()

    def scale(col_tr, col_te):
        tr = np.array(col_tr.apply(extract_rating)).reshape(-1, 1)
        te = np.array(col_te.apply(extract_rating)).reshape(-1, 1)
        tr_s = scaler.fit_transform(tr).reshape(-1)
        te_s = scaler.transform(te).reshape(-1)
        return tr_s, te_s

    acad_tr, acad_te = scale(
        df_train["How likely are you to use this model for academic tasks?"],
        df_test["How likely are you to use this model for academic tasks?"],
    )
    subfreq_tr, subfreq_te = scale(
        df_train[
            "Based on your experience, how often has this model given you a response that felt suboptimal?"
        ],
        df_test[
            "Based on your experience, how often has this model given you a response that felt suboptimal?"
        ],
    )
    refer_tr, refer_te = scale(
        df_train[df_train.columns[-4]],
        df_test[df_test.columns[-4]],
    )
    verify_tr, verify_te = scale(
        df_train[df_train.columns[-3]],
        df_test[df_test.columns[-3]],
    )

    num_train = np.vstack((acad_tr, subfreq_tr, refer_tr, verify_tr)).T
    num_test = np.vstack((acad_te, subfreq_te, refer_te, verify_te)).T

    poly = PolynomialFeatures(degree=3)
    num_train_poly = poly.fit_transform(num_train)
    num_test_poly = poly.transform(num_test)

    tr1_tr = df_train[df_train.columns[1]].fillna("").astype(str).values
    tr2_tr = df_train[df_train.columns[-2]].fillna("").astype(str).values
    tr3_tr = df_train[df_train.columns[-5]].fillna("").astype(str).values

    tr1_te = df_test[df_test.columns[1]].fillna("").astype(str).values
    tr2_te = df_test[df_test.columns[-2]].fillna("").astype(str).values
    tr3_te = df_test[df_test.columns[-5]].fillna("").astype(str).values

    v1 = TfidfVectorizer(
        max_features=1000, stop_words="english", strip_accents="unicode"
    )
    v2 = TfidfVectorizer(
        max_features=1000, stop_words="english", strip_accents="unicode"
    )
    v3 = TfidfVectorizer(
        max_features=1000, stop_words="english", strip_accents="unicode"
    )

    tr1_tr_vec = v1.fit_transform(tr1_tr).toarray()
    tr2_tr_vec = v2.fit_transform(tr2_tr).toarray()
    tr3_tr_vec = v3.fit_transform(tr3_tr).toarray()

    tr1_te_vec = v1.transform(tr1_te).toarray()
    tr2_te_vec = v2.transform(tr2_te).toarray()
    tr3_te_vec = v3.transform(tr3_te).toarray()

    X_train = np.hstack(
        (
            tr1_tr_vec,
            tr2_tr_vec,
            tr3_tr_vec,
            best_tasks_encoded_train,
            subopt_tasks_encoded_train,
            num_train_poly,
        )
    ).astype(np.float32)

    X_test = np.hstack(
        (
            tr1_te_vec,
            tr2_te_vec,
            tr3_te_vec,
            best_tasks_encoded_test,
            subopt_tasks_encoded_test,
            num_test_poly,
        )
    ).astype(np.float32)

    return X_train, X_test, y_train, y_test


def main():
    np.random.seed(0)

    df = pd.read_csv(file_name)
    df.dropna(inplace=True)
    df.replace("#NAME?", "", inplace=True)
    df.replace("[THIS MODEL]", " ", inplace=True)
    df.replace("[ANOTHER MODEL]", " ", inplace=True)

    ids = np.unique(df["student_id"].values)
    n_train_ids = int(np.floor(len(ids) * 0.8))
    train_ids = np.random.choice(ids, n_train_ids, replace=False)
    test_ids = ids[~np.isin(ids, train_ids)]

    df_train = df[df["student_id"].isin(train_ids)].copy()
    df_test = df[df["student_id"].isin(test_ids)].copy()

    X_train, X_test, y_train, y_test = build_features(df_train, df_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=0, n_jobs=-1)

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )

    print("Running RandomForest GridSearchCV...")
    grid.fit(X_tr, y_tr)

    print("\nBest RF params:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)

    best_rf = grid.best_estimator_
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\nFinal Random Forest test accuracy:", test_acc)
    print("\nConfusion Matrix (test):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report (test):")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
