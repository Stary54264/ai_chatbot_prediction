import random
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer  # <-- Added for numeric imputation
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix, \
    classification_report, log_loss
from sklearn.neural_network import MLPClassifier


# --- Helper Functions---

def process_multiselect(series, target_tasks):
    """Convert multiselect strings to lists, keeping only specified features"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            # Check which of the target tasks are present in the response
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def load_and_clean_data(file_name):
    """
    Loads the CSV file, performs initial cleaning, and imputes columns.
    """
    print(f"Loading and cleaning data from {file_name}...")
    df = pd.read_csv(file_name)

    df.replace('#NAME?', 'empty response', inplace=True)
    # df.replace(string.punctuation, '', regex=True, inplace=True)
    punctuation_pattern = f'[{re.escape(string.punctuation)}]'


    # Impute text and multiselect columns with an empty string
    text_cols = [
        df.columns[1],
        df.columns[-2],
        df.columns[-5]
    ]
    multiselect_cols = [df.columns[3], df.columns[5]]

    for col in text_cols + multiselect_cols:
        df[col] = df[col].fillna('empty response')
        df[col] = df[col].replace(punctuation_pattern, ' ', regex=True)

    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets based on unique student_id.
    """
    print("Splitting data into train and test sets")
    np.random.seed(random_state)

    ids = np.unique(df['student_id'].values)

    train_ids = np.random.choice(
        ids,
        np.floor(len(ids) * (1 - test_size)).astype(int),
        replace=False
    )
    test_ids = ids[~np.isin(ids, train_ids)]

    df_train = df[df['student_id'].isin(train_ids)]
    df_test = df[df['student_id'].isin(test_ids)]

    y_train = df_train['label'].values
    y_test = df_test['label'].values

    return df_train, df_test, y_train, y_test


def get_target_tasks(df_train):
    """
    Extracts all unique task names from the training set
    """
    print("Extracting target tasks from training data")
    target_tasks = []
    for multiselect in df_train['Which types of tasks do you feel this model handles best? (Select all that apply.)']:
        multiselect = str(multiselect)
        tasks = multiselect.split(',')
        for task in tasks:
            task = task.strip()
            if task not in target_tasks:
                target_tasks.append(task)
    return target_tasks


def preprocess_train_features(df_train, target_tasks):
    """
    Fits and transforms all features from the training data.
    Returns the processed X_train and a dictionary of fitted transformers.
    """
    transformers = {}

    # ---Multiselect Features---
    mlb_best = MultiLabelBinarizer(classes=target_tasks)
    mlb_subopt = MultiLabelBinarizer(classes=target_tasks)

    best_tasks_lists = process_multiselect(
        df_train['Which types of tasks do you feel this model handles best? (Select all that apply.)'],
        target_tasks
    )
    suboptimal_tasks_lists = process_multiselect(
        df_train[
            'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'],
        target_tasks
    )

    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)

    transformers['mlb_best'] = mlb_best
    transformers['mlb_subopt'] = mlb_subopt

    # ---Rating Features---
    # Extract
    academic_numeric = np.array(
        df_train['How likely are you to use this model for academic tasks?'].apply(extract_rating)).reshape(-1, 1)
    subopt_numeric = np.array(
        df_train['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(
            extract_rating)).reshape(-1, 1)
    refer_numeric = np.array(df_train[df_train.columns[-4]].apply(extract_rating)).reshape(-1, 1)
    verify_numeric = np.array(df_train[df_train.columns[-3]].apply(extract_rating)).reshape(-1, 1)

    # Impute
    imputer_acad = SimpleImputer(strategy='median')
    imputer_subopt = SimpleImputer(strategy='median')
    imputer_refer = SimpleImputer(strategy='median')
    imputer_verify = SimpleImputer(strategy='median')

    academic_numeric_imp = imputer_acad.fit_transform(academic_numeric)
    subopt_numeric_imp = imputer_subopt.fit_transform(subopt_numeric)
    refer_numeric_imp = imputer_refer.fit_transform(refer_numeric)
    verify_numeric_imp = imputer_verify.fit_transform(verify_numeric)

    transformers.update({
        'imputer_acad': imputer_acad, 'imputer_subopt': imputer_subopt,
        'imputer_refer': imputer_refer, 'imputer_verify': imputer_verify
    })

    # Scale
    scaler_acad = MinMaxScaler()
    scaler_subopt = MinMaxScaler()
    scaler_refer = MinMaxScaler()
    scaler_verify = MinMaxScaler()

    academic_numeric_scaled = scaler_acad.fit_transform(academic_numeric_imp).reshape(-1)
    subopt_numeric_scaled = scaler_subopt.fit_transform(subopt_numeric_imp).reshape(-1)
    refer_numeric_scaled = scaler_refer.fit_transform(refer_numeric_imp).reshape(-1)
    verify_numeric_scaled = scaler_verify.fit_transform(verify_numeric_imp).reshape(-1)

    transformers.update({
        'scaler_acad': scaler_acad, 'scaler_subopt': scaler_subopt,
        'scaler_refer': scaler_refer, 'scaler_verify': scaler_verify
    })

    # ---Text (Bag of Words) Features---
    tr1 = np.array(df_train[df_train.columns[1]])
    tr2 = np.array(df_train[df_train.columns[-2]])
    tr3 = np.array(df_train[df_train.columns[-5]])

    v1 = TfidfVectorizer(max_features=1000, stop_words='english', strip_accents='unicode', ngram_range=(1, 1))
    v2 = TfidfVectorizer(max_features=1000, stop_words='english', strip_accents='unicode', ngram_range=(1, 1))
    v3 = TfidfVectorizer(max_features=1000, stop_words='english', strip_accents='unicode', ngram_range=(1, 1))

    tr1_train = v1.fit_transform(tr1).toarray()
    tr2_train = v2.fit_transform(tr2).toarray()
    tr3_train = v3.fit_transform(tr3).toarray()

    transformers.update({'v1': v1, 'v2': v2, 'v3': v3})

    # ---Poly Features---
    X_numeric = np.hstack((
        academic_numeric_scaled.reshape(-1, 1),
        subopt_numeric_scaled.reshape(-1, 1),
        refer_numeric_scaled.reshape(-1, 1),
        verify_numeric_scaled.reshape(-1, 1)
    ))

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_numeric)
    transformers['poly'] = poly

    # ---Combine Features---
    X_train = np.hstack((
        tr1_train,
        tr2_train,
        tr3_train,
        best_tasks_encoded,
        suboptimal_tasks_encoded,
        X_poly
    ))

    return X_train, transformers


def preprocess_test_features(df_test, target_tasks, transformers):
    """
    Transforms the test data using the transformers fitted on the train data.
    """

    # ---Multi-select Features---
    best_tasks_lists = process_multiselect(
        df_test['Which types of tasks do you feel this model handles best? (Select all that apply.)'],
        target_tasks
    )
    suboptimal_tasks_lists = process_multiselect(
        df_test[
            'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'],
        target_tasks
    )

    best_tasks_encoded = transformers['mlb_best'].transform(best_tasks_lists)
    suboptimal_tasks_encoded = transformers['mlb_subopt'].transform(suboptimal_tasks_lists)

    # ---Rating Features---
    # Extract
    academic_numeric = np.array(
        df_test['How likely are you to use this model for academic tasks?'].apply(extract_rating)).reshape(-1, 1)
    subopt_numeric = np.array(
        df_test['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(
            extract_rating)).reshape(-1, 1)
    refer_numeric = np.array(df_test[df_test.columns[-4]].apply(extract_rating)).reshape(-1, 1)
    verify_numeric = np.array(df_test[df_test.columns[-3]].apply(extract_rating)).reshape(-1, 1)

    # Impute
    academic_numeric_imp = transformers['imputer_acad'].transform(academic_numeric)
    subopt_numeric_imp = transformers['imputer_subopt'].transform(subopt_numeric)
    refer_numeric_imp = transformers['imputer_refer'].transform(refer_numeric)
    verify_numeric_imp = transformers['imputer_verify'].transform(verify_numeric)

    # Scale
    academic_numeric_scaled = transformers['scaler_acad'].transform(academic_numeric_imp).reshape(-1)
    subopt_numeric_scaled = transformers['scaler_subopt'].transform(subopt_numeric_imp).reshape(-1)
    refer_numeric_scaled = transformers['scaler_refer'].transform(refer_numeric_imp).reshape(-1)
    verify_numeric_scaled = transformers['scaler_verify'].transform(verify_numeric_imp).reshape(-1)

    # --- 3. Text (Bag of Words) Features ---
    tr1 = np.array(df_test[df_test.columns[1]])
    tr2 = np.array(df_test[df_test.columns[-2]])
    tr3 = np.array(df_test[df_test.columns[-5]])

    tr1_test = transformers['v1'].transform(tr1).toarray()
    tr2_test = transformers['v2'].transform(tr2).toarray()
    tr3_test = transformers['v3'].transform(tr3).toarray()

    # ---Polynomial Features---
    X_numeric = np.hstack((
        academic_numeric_scaled.reshape(-1, 1),
        subopt_numeric_scaled.reshape(-1, 1),
        refer_numeric_scaled.reshape(-1, 1),
        verify_numeric_scaled.reshape(-1, 1)
    ))

    X_poly = transformers['poly'].transform(X_numeric)

    # ---Combine Features---
    X_test = np.hstack((
        tr1_test,
        tr2_test,
        tr3_test,
        best_tasks_encoded,
        suboptimal_tasks_encoded,
        X_poly
    ))

    return X_test


def train_model(X_train, y_train):
    """
    Performs GridSearchCV for base models and trains the final StackingClassifier.
    """
    print("Training models...")

    # ---Logistic Regression GridSearch---
    print("Running GridSearchCV for LogisticRegression...")
    param_grid_lr = [
        {'C': [0.2, 0.4, 0.6], 'penalty': ['elasticnet'], 'l1_ratio': [0.2, 1],
         'solver': ['saga'], 'fit_intercept': [True, False], 'max_iter': [2000],
         'tol': [0.001]}
    ]
    lr_search = GridSearchCV(LogisticRegression(), param_grid_lr, cv=3, scoring='accuracy', refit=True, n_jobs=-1, verbose=1)
    lr_search.fit(X_train, y_train)
    print(f"Best LR Params: {lr_search.best_params_}")

    # --- Build tuned LogisticRegression base estimator ---
    base_lr = LogisticRegression(
        fit_intercept=lr_search.best_params_['fit_intercept'],
        max_iter=lr_search.best_params_['max_iter'],
        solver=lr_search.best_params_['solver'],
        penalty=lr_search.best_params_['penalty'],
        C=lr_search.best_params_['C'],
        tol=lr_search.best_params_['tol'],
        l1_ratio=lr_search.best_params_['l1_ratio'],
    )

    # --- Tune n_estimators for BaggingClassifier ---
    n_values = [20, 50, 100]
    best_n = None
    best_cv_acc = -1.0

    print("Tuning n_estimators for BaggingClassifier...")
    for n in n_values:
        bagging_tmp = BaggingClassifier(
            estimator=base_lr,
            n_estimators=n,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=-1,
            random_state=42,
        )
        # 3-fold CV on the training data
        scores = cross_val_score(
            bagging_tmp,
            X_train,
            y_train,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
        )
        mean_acc = scores.mean()
        print(f"  n_estimators={n}: CV accuracy = {mean_acc:.4f}")

        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_n = n

    print(f"Best n_estimators based on CV: {best_n} (CV accuracy = {best_cv_acc:.4f})")

    # --- Train final BaggingClassifier with best n_estimators ---
    bagging_clf = BaggingClassifier(
        estimator=base_lr,
        n_estimators=best_n,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=42,
    )

    print("Fitting final BaggingClassifier with tuned LogisticRegression base estimator...")
    bagging_clf.fit(X_train, y_train)
    print("Bagging model training complete.")

    return bagging_clf



def evaluate_model(clf, X_test, y_test):
    """
    Evaluates the final model and prints the accuracy and classification report.
    """
    print("---Model Evaluation---")

    accuracy = clf.score(X_test, y_test)
    print(f"Final Test Accuracy: {accuracy:.4f}")

    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Test Set Confusion Matrix")
    plt.show()



def main():
    file_name = "data/training_data_clean.csv"

    # 1) Load and Clean
    df = load_and_clean_data(file_name)

    # 2) Get Task List
    target_tasks = get_target_tasks(df)

    # 3) Split
    df_train, df_test, y_train, y_test = split_data(df, test_size=0.2)

    # 4) Preprocess Features
    X_train, transformers = preprocess_train_features(df_train, target_tasks)
    X_test = preprocess_test_features(df_test, target_tasks, transformers)

    # 5) Train Model
    print("Starting training of Bagged Logistic Regression model...")
    model = train_model(X_train, y_train)

    # 6) Evaluate Model
    print("Training complete. Evaluating on test set...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
