import numpy as np
import pandas as pd
import re
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, PolynomialFeatures
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_name = "../data/training_data_clean.csv"


def process_multiselect(series, target_tasks):
    processed = []
    for response in series:
        if pd.isna(response) or response == "":
            processed.append([])
        else:
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    m = re.match(r"^(\d+)", str(response))
    return int(m.group(1)) if m else 3


def main():
    np.random.seed(273)

    # Load data
    df = pd.read_csv(file_name)
    
    print(f"Total samples before cleaning: {len(df)}")

    # Clean data 
    df.replace("#NAME?", "", inplace=True)
    df.replace("[THIS MODEL]", " ", inplace=True)
    df.replace("[ANOTHER MODEL]", " ", inplace=True)

    # Fill NA in text columns
    text_cols = [df.columns[1], df.columns[-2], df.columns[-5]]
    for col in text_cols:
        df[col] = df[col].fillna("")

    # Fill NA in multi-select columns
    multiselect_cols = [df.columns[3], df.columns[5]]
    for col in multiselect_cols:
        df[col] = df[col].fillna("")

    # Fill NA in rating columns with "3"
    rating_cols = [
        "How likely are you to use this model for academic tasks?",
        "Based on your experience, how often has this model given you a response that felt suboptimal?",
        df.columns[-4],
        df.columns[-3]
    ]
    for col in rating_cols:
        if col in df.columns:
            df[col] = df[col].fillna("3")

    print(f"Total samples after cleaning: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    ids = np.unique(df["student_id"].values)
    np.random.shuffle(ids)
    
    n_train = int(len(ids) * 0.68)
    n_val = int(len(ids) * 0.16)
    
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    df_train = df[df["student_id"].isin(train_ids)].copy()
    df_val = df[df["student_id"].isin(val_ids)].copy()
    df_test = df[df["student_id"].isin(test_ids)].copy()

    y_train = df_train["label"].values
    y_val = df_val["label"].values
    y_test = df_test["label"].values

    print(f"Train samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}\n")

    # ===== TARGET TASKS =====
    target_tasks = []
    for s in df_train[
        "Which types of tasks do you feel this model handles best? (Select all that apply.)"
    ]:
        for t in str(s).split(","):
            t = t.strip()
            if t and t not in target_tasks:
                target_tasks.append(t)

    # ===== MULTI-SELECT FEATURES =====
    # Train
    best_train_lists = process_multiselect(
        df_train[
            "Which types of tasks do you feel this model handles best? (Select all that apply.)"
        ],
        target_tasks,
    )
    sub_train_lists = process_multiselect(
        df_train[
            "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
        ],
        target_tasks,
    )

    mlb_best = MultiLabelBinarizer()
    mlb_sub = MultiLabelBinarizer()
    best_train = mlb_best.fit_transform(best_train_lists)
    sub_train = mlb_sub.fit_transform(sub_train_lists)

    # Val
    best_val_lists = process_multiselect(
        df_val[
            "Which types of tasks do you feel this model handles best? (Select all that apply.)"
        ],
        target_tasks,
    )
    sub_val_lists = process_multiselect(
        df_val[
            "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
        ],
        target_tasks,
    )
    
    best_val = mlb_best.transform(best_val_lists)
    sub_val = mlb_sub.transform(sub_val_lists)

    # Test
    best_test_lists = process_multiselect(
        df_test[
            "Which types of tasks do you feel this model handles best? (Select all that apply.)"
        ],
        target_tasks,
    )
    sub_test_lists = process_multiselect(
        df_test[
            "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
        ],
        target_tasks,
    )

    best_test = mlb_best.transform(best_test_lists)
    sub_test = mlb_sub.transform(sub_test_lists)

    # ===== RATING FEATURES =====
    # Train
    acad_tr_raw = df_train["How likely are you to use this model for academic tasks?"].apply(extract_rating).values.reshape(-1, 1)
    subfreq_tr_raw = df_train[
        "Based on your experience, how often has this model given you a response that felt suboptimal?"
    ].apply(extract_rating).values.reshape(-1, 1)
    refer_tr_raw = df_train[df_train.columns[-4]].apply(extract_rating).values.reshape(-1, 1)
    verify_tr_raw = df_train[df_train.columns[-3]].apply(extract_rating).values.reshape(-1, 1)

    # Val
    acad_val_raw = df_val["How likely are you to use this model for academic tasks?"].apply(extract_rating).values.reshape(-1, 1)
    subfreq_val_raw = df_val[
        "Based on your experience, how often has this model given you a response that felt suboptimal?"
    ].apply(extract_rating).values.reshape(-1, 1)
    refer_val_raw = df_val[df_val.columns[-4]].apply(extract_rating).values.reshape(-1, 1)
    verify_val_raw = df_val[df_val.columns[-3]].apply(extract_rating).values.reshape(-1, 1)

    # Test
    acad_te_raw = df_test["How likely are you to use this model for academic tasks?"].apply(extract_rating).values.reshape(-1, 1)
    subfreq_te_raw = df_test[
        "Based on your experience, how often has this model given you a response that felt suboptimal?"
    ].apply(extract_rating).values.reshape(-1, 1)
    refer_te_raw = df_test[df_test.columns[-4]].apply(extract_rating).values.reshape(-1, 1)
    verify_te_raw = df_test[df_test.columns[-3]].apply(extract_rating).values.reshape(-1, 1)

    # Individual scalers
    scaler_acad = MinMaxScaler()
    scaler_subfreq = MinMaxScaler()
    scaler_refer = MinMaxScaler()
    scaler_verify = MinMaxScaler()

    acad_tr = scaler_acad.fit_transform(acad_tr_raw).reshape(-1)
    acad_val = scaler_acad.transform(acad_val_raw).reshape(-1)
    acad_te = scaler_acad.transform(acad_te_raw).reshape(-1)

    subfreq_tr = scaler_subfreq.fit_transform(subfreq_tr_raw).reshape(-1)
    subfreq_val = scaler_subfreq.transform(subfreq_val_raw).reshape(-1)
    subfreq_te = scaler_subfreq.transform(subfreq_te_raw).reshape(-1)

    refer_tr = scaler_refer.fit_transform(refer_tr_raw).reshape(-1)
    refer_val = scaler_refer.transform(refer_val_raw).reshape(-1)
    refer_te = scaler_refer.transform(refer_te_raw).reshape(-1)

    verify_tr = scaler_verify.fit_transform(verify_tr_raw).reshape(-1)
    verify_val = scaler_verify.transform(verify_val_raw).reshape(-1)
    verify_te = scaler_verify.transform(verify_te_raw).reshape(-1)

    num_train = np.vstack((acad_tr, subfreq_tr, refer_tr, verify_tr)).T
    num_val = np.vstack((acad_val, subfreq_val, refer_val, verify_val)).T
    num_test = np.vstack((acad_te, subfreq_te, refer_te, verify_te)).T

    # Ensure no NaN
    num_train = np.nan_to_num(num_train, nan=0.0)
    num_val = np.nan_to_num(num_val, nan=0.0)
    num_test = np.nan_to_num(num_test, nan=0.0)

    # Polynomial features
    poly = PolynomialFeatures(degree=3)
    num_train_poly = poly.fit_transform(num_train)
    num_val_poly = poly.transform(num_val)
    num_test_poly = poly.transform(num_test)

    # ===== TEXT FEATURES =====
    # Train
    tr1_tr = df_train[df_train.columns[1]].fillna("").astype(str).values
    tr2_tr = df_train[df_train.columns[-2]].fillna("").astype(str).values
    tr3_tr = df_train[df_train.columns[-5]].fillna("").astype(str).values

    # Val
    tr1_val = df_val[df_val.columns[1]].fillna("").astype(str).values
    tr2_val = df_val[df_val.columns[-2]].fillna("").astype(str).values
    tr3_val = df_val[df_val.columns[-5]].fillna("").astype(str).values

    # Test
    tr1_te = df_test[df_test.columns[1]].fillna("").astype(str).values
    tr2_te = df_test[df_test.columns[-2]].fillna("").astype(str).values
    tr3_te = df_test[df_test.columns[-5]].fillna("").astype(str).values

    v1 = TfidfVectorizer(max_features=1000, stop_words="english", strip_accents="unicode")
    v2 = TfidfVectorizer(max_features=1000, stop_words="english", strip_accents="unicode")
    v3 = TfidfVectorizer(max_features=1000, stop_words="english", strip_accents="unicode")

    tr1_tr_vec = v1.fit_transform(tr1_tr).toarray()
    tr2_tr_vec = v2.fit_transform(tr2_tr).toarray()
    tr3_tr_vec = v3.fit_transform(tr3_tr).toarray()

    tr1_val_vec = v1.transform(tr1_val).toarray()
    tr2_val_vec = v2.transform(tr2_val).toarray()
    tr3_val_vec = v3.transform(tr3_val).toarray()

    tr1_te_vec = v1.transform(tr1_te).toarray()
    tr2_te_vec = v2.transform(tr2_te).toarray()
    tr3_te_vec = v3.transform(tr3_te).toarray()

    # ===== COMBINE ALL FEATURES =====
    X_train = np.hstack(
        (
            tr1_tr_vec,
            tr2_tr_vec,
            tr3_tr_vec,
            best_train,
            sub_train,
            num_train_poly,
        )
    ).astype(np.float32)

    X_val = np.hstack(
        (
            tr1_val_vec,
            tr2_val_vec,
            tr3_val_vec,
            best_val,
            sub_val,
            num_val_poly,
        )
    ).astype(np.float32)

    X_test = np.hstack(
        (
            tr1_te_vec,
            tr2_te_vec,
            tr3_te_vec,
            best_test,
            sub_test,
            num_test_poly,
        )
    ).astype(np.float32)

    print(f"Feature matrix shape: {X_train.shape}\n")

    # ===== TRAIN NAIVE BAYES WITH ALPHA TUNING =====
    print("="*70)
    print("TUNING ALPHA PARAMETER")
    print("="*70)

    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_alpha = None
    best_val_acc = 0

    for alpha in alphas:
        nb_temp = MultinomialNB(alpha=alpha)
        nb_temp.fit(X_train, y_train)
        val_acc = nb_temp.score(X_val, y_val)
        
        print(f"Alpha={alpha:.2f}: Val accuracy={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Use best alpha for final training
    nb = MultinomialNB(alpha=best_alpha)
    nb.fit(X_train, y_train)

    val_acc = nb.score(X_val, y_val)
    print(f"\nValidation accuracy with best alpha: {val_acc:.4f}")

    # Retrain on train + val
    print("\nRetraining on train + val combined...")
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    nb.fit(X_train_full, y_train_full)

    # Final test evaluation
    y_pred = nb.predict(X_test)
    train_acc = nb.score(X_train_full, y_train_full)
    test_acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Training (train+val) accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Train-Test gap: {train_acc - test_acc:.4f}")
    
    if train_acc - test_acc > 0.15:
        print("Warning: Possible overfitting (gap > 0.15)")
    else:
        print("Good generalization")
    
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # ===== SAVE EVERYTHING =====
    params = {
        "class_log_prior": nb.class_log_prior_.tolist(),
        "feature_log_prob": nb.feature_log_prob_.tolist(),
        "classes": nb.classes_.tolist(),
        
        # Text features
        "vocab1": list(v1.get_feature_names_out()),
        "vocab2": list(v2.get_feature_names_out()),
        "vocab3": list(v3.get_feature_names_out()),
        "idf1": v1.idf_.tolist(),
        "idf2": v2.idf_.tolist(),
        "idf3": v3.idf_.tolist(),
        
        # Multi-select
        "target_tasks": target_tasks,
        "mlb_best_classes": mlb_best.classes_.tolist(),
        "mlb_sub_classes": mlb_sub.classes_.tolist(),
        
        # Scalers
        "scaler_acad_min": scaler_acad.data_min_.tolist(),
        "scaler_acad_scale": scaler_acad.scale_.tolist(),
        "scaler_subfreq_min": scaler_subfreq.data_min_.tolist(),
        "scaler_subfreq_scale": scaler_subfreq.scale_.tolist(),
        "scaler_refer_min": scaler_refer.data_min_.tolist(),
        "scaler_refer_scale": scaler_refer.scale_.tolist(),
        "scaler_verify_min": scaler_verify.data_min_.tolist(),
        "scaler_verify_scale": scaler_verify.scale_.tolist(),
        
        # Polynomial
        "poly_powers": poly.powers_.tolist(),
    }

    with open("../data/nb_full_params.json", "w") as f:
        json.dump(params, f)

    print("\n" + "="*70)
    print("Saved complete parameters to ../data/nb_full_params.json")
    print(f"Final test accuracy: {test_acc:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()