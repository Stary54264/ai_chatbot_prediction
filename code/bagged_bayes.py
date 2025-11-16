import random
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------------------
# 1. Data loading and preprocessing (copied from naive_bayes.ipynb)
# ---------------------------------------------------------------------

def load_and_vectorize(path_raw: str = "data/data_raw.csv"):
    """
    Load the raw dataset, apply the same cleaning and vectorization
    steps used for the original Naive Bayes model, and return
    train / validation / test splits.
    """

    # Read dataset
    data_raw = pd.read_csv(path_raw)

    # Basic cleaning
    data_clean = data_raw.replace("#NAME?", "")
    data_clean = data_clean.fillna("")
    data_clean.columns = [
        "id", "t1", "n1", "c1", "n2",
        "c2", "t2", "n3", "n4", "t3", "label"
    ]

    # Start feature matrix with student id
    data_vec = data_clean[["id"]].copy()

    # Vectorize `choice` columns (same logic as in the notebook)
    option = ["computation", "code", "analysis", "concept",
              "format", "essay", "text", "idea"]

    for i in range(1, 3):
        for opt in option:
            # NOTE: this intentionally mirrors the notebook implementation
            # so that the feature space is identical.
            data_vec[f"c{i}_{opt}"] = data_clean[f"c{1}"].str.contains(opt).astype(int)

    # Vectorize `text` columns using a bag-of-words representation
    for i in range(1, 4):
        vec = CountVectorizer(max_features=3000, binary=True)
        t_mat = vec.fit_transform(data_clean[f"t{i}"])

        col = [f"t{i}_{word}" for word in vec.get_feature_names_out()]
        data_t = pd.DataFrame(t_mat.toarray(), columns=col)
        data_vec = pd.concat([data_vec, data_t], axis=1)

    # Convert labels to {0, 1, 2}
    data_vec["label"] = pd.factorize(data_clean["label"])[0]

    # Split data (68%-16%-16%) by id, as in the notebook
    random.seed(311)
    stud = data_vec["id"].unique().tolist()
    random.shuffle(stud)
    train_ids, valid_ids, test_ids = stud[:187], stud[187:231], stud[231:]

    data_train = data_vec[data_vec["id"].isin(train_ids)]
    data_valid = data_vec[data_vec["id"].isin(valid_ids)]
    data_test = data_vec[data_vec["id"].isin(test_ids)]

    X_train = data_train.drop(["id", "label"], axis=1).values
    t_train = data_train["label"].values

    X_valid = data_valid.drop(["id", "label"], axis=1).values
    t_valid = data_valid["label"].values

    X_test = data_test.drop(["id", "label"], axis=1).values
    t_test = data_test["label"].values

    return X_train, t_train, X_valid, t_valid, X_test, t_test


# ---------------------------------------------------------------------
# 2. Original Naive Bayes implementation (3-class, Bernoulli features)
# ---------------------------------------------------------------------

def nb_map3(X: np.ndarray, t: np.ndarray, a: float, b: float):
    """
    MAP estimate of the 3-class Bernoulli Naive Bayes model
    with Beta(a, b) priors on both the class prior pi and the
    feature probabilities theta.

    Parameters
    ----------
    X : array of shape (N, V)
        Binary bag-of-words features.
    t : array of shape (N,)
        Class labels in {0, 1, 2}.
    a, b : float
        Hyperparameters of the Beta prior.

    Returns
    -------
    pi : array of shape (3,)
        Class prior probabilities.
    theta : array of shape (V, 3)
        Feature probabilities for each class.
        theta[j, c] = P(x_j = 1 | y = c).
    """
    N = X.shape[0]
    pi = np.zeros(3)
    N_t = np.zeros(3)
    theta = np.zeros((X.shape[1], 3))

    for c in range(3):
        X_c = X[t == c]
        N_c = X_c.shape[0]
        N_t[c] = N_c

        # MAP estimate of pi_c with Beta(a, b) prior
        pi[c] = (a + N_c - 1) / (a + b + N - 2)

        # MAP estimate of theta_{jc} for each feature j
        # with Beta(a, b) prior and Bernoulli likelihood
        theta[:, c] = (a + np.sum(X_c, axis=0) - 1) / (a + b + N_c - 2)

    return pi, theta


def pred3(X: np.ndarray, pi: np.ndarray, theta: np.ndarray):
    """
    Predict class labels for a batch of examples using the
    3-class Bernoulli Naive Bayes model.

    Parameters
    ----------
    X : array of shape (N, V)
    pi : array of shape (3,)
    theta : array of shape (V, 3)

    Returns
    -------
    y : array of shape (N,)
        Predicted class labels in {0, 1, 2}.
    """
    N = X.shape[0]
    log_p = np.zeros((N, 3))

    # For each class c, compute log p(y=c) + sum_j log p(x_j | y=c)
    for c in range(3):
        # log p(x_j = 1 | y=c) and log p(x_j = 0 | y=c)
        log_theta_c = np.log(theta[:, c] + 1e-12)
        log_one_minus_theta_c = np.log(1.0 - theta[:, c] + 1e-12)

        # X @ log_theta_c is sum over j where x_j = 1
        # (1 - X) @ log_one_minus_theta_c is sum over j where x_j = 0
        log_p[:, c] = (
            np.log(pi[c] + 1e-12)
            + X @ log_theta_c
            + (1 - X) @ log_one_minus_theta_c
        )

    return np.argmax(log_p, axis=1)


# ---------------------------------------------------------------------
# 3. Bagging wrapper around the Naive Bayes model
# ---------------------------------------------------------------------

class BaggingNaiveBayes:
    """
    Bagging (Bootstrap Aggregating) ensemble using the custom
    3-class Naive Bayes model defined above.

    The idea is to:
      1. Draw B bootstrap samples from the training set.
      2. Fit one Naive Bayes model on each bootstrap sample.
      3. Aggregate predictions using majority vote.
    """

    def __init__(self, n_estimators: int = 50, a: float = 2.0, b: float = 2.0,
                 random_state: int | None = None):
        self.n_estimators = n_estimators
        self.a = a
        self.b = b
        self.random_state = random_state
        self.models_ = []  # list of (pi, theta) tuples

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        Fit n_estimators Naive Bayes models on bootstrap samples
        of the training data.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        N = X.shape[0]
        self.models_ = []

        for m in range(self.n_estimators):
            # Draw bootstrap sample indices with replacement
            indices = np.random.choice(N, size=N, replace=True)
            X_boot = X[indices]
            t_boot = t[indices]

            # Fit Naive Bayes on the bootstrap sample
            pi_m, theta_m = nb_map3(X_boot, t_boot, self.a, self.b)
            self.models_.append((pi_m, theta_m))

        return self

    def predict(self, X: np.ndarray):
        """
        Predict by aggregating the predictions of all base models
        using majority vote.
        """
        if not self.models_:
            raise RuntimeError("BaggingNaiveBayes has not been fitted yet.")

        N = X.shape[0]
        votes = np.zeros((self.n_estimators, N), dtype=int)

        # Collect predictions from each base model
        for m, (pi_m, theta_m) in enumerate(self.models_):
            votes[m] = pred3(X, pi_m, theta_m)

        # Majority vote over axis 0
        # labels are {0, 1, 2}, so we can use bincount per sample
        y_pred = np.zeros(N, dtype=int)
        for i in range(N):
            counts = np.bincount(votes[:, i], minlength=3)
            y_pred[i] = np.argmax(counts)

        return y_pred


# ---------------------------------------------------------------------
# 4. Putting everything together: evaluation script
# ---------------------------------------------------------------------

def main():
    # 1. Load and vectorize data
    X_train, t_train, X_valid, t_valid, X_test, t_test = load_and_vectorize()

    # 2. Choose hyperparameters (a, b)
    #    In the original notebook, these were tuned on a grid.
    #    Here, for simplicity, you can either:
    #      - plug in the best (a, b) found previously, or
    #      - keep a fixed reasonable choice (e.g., a = b = 2.0).
    #    If you already know the optimal a_opt and b_opt from the notebook,
    #    you can paste them here directly.
    a_opt = 1.4
    b_opt = 5.0
    def tune_hyperparameters(X_train, t_train, X_valid, t_valid):
        a_values = [1.0, 1.5, 2.0, 3.0]
        b_values = [1.0, 1.5, 2.0, 3.0]
        n_values = [20, 50, 80, 120]

        best_acc = -1
        best_params = None

        for a in a_values:
            for b in b_values:
                # Train a single NB just to compute baseline smoothing effects
                pi_single, theta_single = nb_map3(X_train, t_train, a, b)
                preds_val_single = pred3(X_valid, pi_single, theta_single)
                base_acc = accuracy_score(t_valid, preds_val_single)

                for n in n_values:
                    bag_nb = BaggingNaiveBayes(
                        n_estimators=n,
                        a=a,
                        b=b,
                        random_state=0
                    )
                    bag_nb.fit(X_train, t_train)
                    preds_val = bag_nb.predict(X_valid)
                    acc = accuracy_score(t_valid, preds_val)

                    print(f"[a={a}, b={b}, n={n}]  Validation Accuracy = {acc:.4f} (single={base_acc:.4f})")

                    if acc > best_acc:
                        best_acc = acc
                        best_params = (a, b, n)

        print("\n=== BEST HYPERPARAMETERS FOUND ===")
        print(f"a = {best_params[0]}, b = {best_params[1]}, n_estimators = {best_params[2]}")
        print(f"Validation Accuracy = {best_acc:.4f}")

        return best_params
    best_a, best_b, best_n = tune_hyperparameters(X_train, t_train, X_valid, t_valid)

    # 3. Baseline single Naive Bayes model (no bagging)
    pi_base, theta_base = nb_map3(X_train, t_train, a_opt, b_opt)

    y_train_base = pred3(X_train, pi_base, theta_base)
    y_valid_base = pred3(X_valid, pi_base, theta_base)
    y_test_base = pred3(X_test, pi_base, theta_base)

    print("=== Single Naive Bayes (baseline) ===")
    print(f"Train accuracy: {accuracy_score(t_train, y_train_base):.3f}")
    print(f"Valid accuracy: {accuracy_score(t_valid, y_valid_base):.3f}")
    print(f"Test  accuracy: {accuracy_score(t_test, y_test_base):.3f}")
    print()

    # 4. Bagged Naive Bayes model
    bag_nb = BaggingNaiveBayes(
        n_estimators=best_n,
        a=best_a,
        b=best_b,
        random_state=0
    )
    bag_nb.fit(X_train, t_train)

    y_train_bag = bag_nb.predict(X_train)
    y_valid_bag = bag_nb.predict(X_valid)
    y_test_bag = bag_nb.predict(X_test)

    print("=== Bagged Naive Bayes (Bagging) ===")
    print(f"Train accuracy: {accuracy_score(t_train, y_train_bag):.3f}")
    print(f"Valid accuracy: {accuracy_score(t_valid, y_valid_bag):.3f}")
    print(f"Test  accuracy: {accuracy_score(t_test, y_test_bag):.3f}")
    print()

    # Optional: print a confusion matrix and classification report on the test set
    print("Confusion matrix (test, Bagged Naive Bayes):")
    print(confusion_matrix(t_test, y_test_bag))
    print()
    print("Classification report (test, Bagged Naive Bayes):")
    print(classification_report(t_test, y_test_bag))


if __name__ == "__main__":
    main()
