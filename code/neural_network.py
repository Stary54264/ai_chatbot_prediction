import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    MinMaxScaler,
    LabelEncoder,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
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
            present = [task for task in target_tasks if task in str(response)]
            processed.append(present)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from strings like '3 - Sometimes'.
    If it doesn't match, return 0.
    """
    m = re.match(r"^(\d+)", str(response))
    return int(m.group(1)) if m else 0


def build_features(df_train, df_test):
    """
    Preprocess data into features:
    - combine the 3 open-ended text responses into one text field
    - TF-IDF on combined text (with unigrams + bigrams)
    - one-hot for multi-select questions
    - scaled numeric ratings
    """
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    col_text1 = df_train.columns[1]
    col_text2 = df_train.columns[-2]
    col_text3 = df_train.columns[-5]

    train_text = (
        df_train[col_text1].fillna("").astype(str)
        + " [SEP] "
        + df_train[col_text2].fillna("").astype(str)
        + " [SEP] "
        + df_train[col_text3].fillna("").astype(str)
    )

    test_text = (
        df_test[col_text1].fillna("").astype(str)
        + " [SEP] "
        + df_test[col_text2].fillna("").astype(str)
        + " [SEP] "
        + df_test[col_text3].fillna("").astype(str)
    )

 
    tfidf = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
    )

    X_text_train = tfidf.fit_transform(train_text).toarray()
    X_text_test = tfidf.transform(test_text).toarray()

    target_tasks = []
    for s in df_train[
        "Which types of tasks do you feel this model handles best? (Select all that apply.)"
    ]:
        for t in str(s).split(","):
            t = t.strip()
            if t and t not in target_tasks:
                target_tasks.append(t)

    best_train = process_multiselect(
        df_train[
            "Which types of tasks do you feel this model handles best? (Select all that apply.)"
        ],
        target_tasks,
    )
    subopt_train = process_multiselect(
        df_train[
            "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
        ],
        target_tasks,
    )

    mlb_best = MultiLabelBinarizer()
    mlb_sub = MultiLabelBinarizer()

    best_train_enc = mlb_best.fit_transform(best_train)
    sub_train_enc = mlb_sub.fit_transform(subopt_train)

    best_test_enc = mlb_best.transform(
        process_multiselect(
            df_test[
                "Which types of tasks do you feel this model handles best? (Select all that apply.)"
            ],
            target_tasks,
        )
    )
    sub_test_enc = mlb_sub.transform(
        process_multiselect(
            df_test[
                "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
            ],
            target_tasks,
        )
    )

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

    X_num_train = np.vstack((acad_tr, subfreq_tr, refer_tr, verify_tr)).T
    X_num_test = np.vstack((acad_te, subfreq_te, refer_te, verify_te)).T

    X_train = np.hstack(
        (
            X_text_train,
            best_train_enc,
            sub_train_enc,
            X_num_train,
        )
    ).astype(np.float32)

    X_test = np.hstack(
        (
            X_text_test,
            best_test_enc,
            sub_test_enc,
            X_num_test,
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
    train_ids = np.random.choice(ids, int(len(ids) * 0.8), replace=False)
    test_ids = ids[~np.isin(ids, train_ids)]

    df_train = df[df["student_id"].isin(train_ids)].copy()
    df_test = df[df["student_id"].isin(test_ids)].copy()

    X_train, X_test, y_train, y_test = build_features(df_train, df_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train_enc,
        test_size=0.2,
        random_state=0,
        stratify=y_train_enc,
    )

    hidden_sizes = [(256,), (256, 128), (512, 256)]
    alphas = [1e-4, 5e-4, 1e-3]
    lrs = [1e-3, 5e-4]

    best_cfg = None
    best_val_acc = 0.0

    print("Validation results (Neural Network):")
    for h in hidden_sizes:
        for a in alphas:
            for lr in lrs:
                clf = MLPClassifier(
                    hidden_layer_sizes=h,
                    activation="relu",
                    solver="adam",
                    learning_rate_init=lr,
                    learning_rate="adaptive",
                    alpha=a,
                    max_iter=400,
                    random_state=0,
                    tol=1e-4,
                )
                clf.fit(X_tr, y_tr)
                y_val_pred = clf.predict(X_val)
                acc = accuracy_score(y_val, y_val_pred)
                print(
                    f"hidden={h}, alpha={a:.0e}, lr={lr:.0e} | val acc={acc:.3f}"
                )
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_cfg = (h, a, lr)

    print("\nBest config on validation:", best_cfg, "val acc =", best_val_acc)

    best_hidden, best_alpha, best_lr = best_cfg
    nn_final = MLPClassifier(
        hidden_layer_sizes=best_hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=best_lr,
        learning_rate="adaptive",
        alpha=best_alpha,
        max_iter=600,
        random_state=0,
        tol=1e-4,
    )
    nn_final.fit(X_train, y_train_enc)

    y_pred = nn_final.predict(X_test)
    test_acc = accuracy_score(y_test_enc, y_pred)

    print("\nFinal NN test accuracy:", test_acc)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, y_pred))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test_enc,
            y_pred,
            target_names=le.classes_,
        )
    )


if __name__ == "__main__":
    main()