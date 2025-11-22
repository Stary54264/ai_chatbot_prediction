import numpy as np
import pandas as pd
import json
import re
from collections import Counter

# Load all parameters
with open("nb_full_params.json", "r") as f:
    params = json.load(f)

class_log_prior = np.array(params["class_log_prior"])
feature_log_prob = np.array(params["feature_log_prob"])
classes = params["classes"]

vocab1 = params["vocab1"]
vocab2 = params["vocab2"]
vocab3 = params["vocab3"]
idf1 = np.array(params["idf1"])
idf2 = np.array(params["idf2"])
idf3 = np.array(params["idf3"])

target_tasks = params["target_tasks"]
mlb_best_classes = params["mlb_best_classes"]
mlb_sub_classes = params["mlb_sub_classes"]

scaler_acad_min = np.array(params["scaler_acad_min"])
scaler_acad_scale = np.array(params["scaler_acad_scale"])
scaler_subfreq_min = np.array(params["scaler_subfreq_min"])
scaler_subfreq_scale = np.array(params["scaler_subfreq_scale"])
scaler_refer_min = np.array(params["scaler_refer_min"])
scaler_refer_scale = np.array(params["scaler_refer_scale"])
scaler_verify_min = np.array(params["scaler_verify_min"])
scaler_verify_scale = np.array(params["scaler_verify_scale"])

poly_powers = np.array(params["poly_powers"])

vocab1_dict = {w: i for i, w in enumerate(vocab1)}
vocab2_dict = {w: i for i, w in enumerate(vocab2)}
vocab3_dict = {w: i for i, w in enumerate(vocab3)}


def preprocess_text(text):
    """
    Lowercase and tokenize.
    """
    text = str(text).lower()
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [w for w in text.split() if w]
    return tokens


def compute_tfidf(tokens, vocab_dict, idf_array):
    """Compute TF-IDF vector"""
    tf_counter = Counter(tokens)
    total_terms = len(tokens) if len(tokens) > 0 else 1
    
    vec = np.zeros(len(vocab_dict))
    
    for word, count in tf_counter.items():
        if word in vocab_dict:
            idx = vocab_dict[word]
            tf = count / total_terms
            vec[idx] = tf * idf_array[idx]
    
    return vec


def process_multiselect_single(response, target_tasks):
    """Process multi-select for one response"""
    if pd.isna(response) or response == "":
        return []
    present_tasks = [task for task in target_tasks if task in str(response)]
    return present_tasks


def encode_multiselect(task_list, mlb_classes):
    """Encode multi-select to binary vector"""
    vec = np.zeros(len(mlb_classes))
    for task in task_list:
        if task in mlb_classes:
            idx = list(mlb_classes).index(task)
            vec[idx] = 1
    return vec


def minmax_scale(value, data_min, scale):
    """Apply MinMaxScaler manually: (x - min) * scale"""
    return (value - data_min) * scale


def extract_rating(response):
    """Extract rating from text"""
    m = re.match(r"^(\d+)", str(response))
    return int(m.group(1)) if m else 3


def polynomial_features_manual(X, powers):
    """Manually compute polynomial features"""
    n_samples = X.shape[0]
    n_output_features = powers.shape[0]
    
    XP = np.empty((n_samples, n_output_features), dtype=X.dtype)
    
    for i, comb in enumerate(powers):
        XP[:, i] = np.prod(X ** comb, axis=1)
    
    return XP


def predict(row):
    """Predict label for a single row"""
    # ===== TEXT FEATURES =====
    text1 = row.iloc[1] if len(row) > 1 else ""
    text2 = row.iloc[-2] if len(row) >= 2 else ""
    text3 = row.iloc[-5] if len(row) >= 5 else ""
    
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    tokens3 = preprocess_text(text3)
    
    vec1 = compute_tfidf(tokens1, vocab1_dict, idf1)
    vec2 = compute_tfidf(tokens2, vocab2_dict, idf2)
    vec3 = compute_tfidf(tokens3, vocab3_dict, idf3)
    
    # ===== MULTI-SELECT FEATURES =====
    best_col = row.iloc[3] if len(row) > 3 else ""
    sub_col = row.iloc[5] if len(row) > 5 else ""
    
    best_tasks = process_multiselect_single(best_col, target_tasks)
    sub_tasks = process_multiselect_single(sub_col, target_tasks)
    
    best_vec = encode_multiselect(best_tasks, mlb_best_classes)
    sub_vec = encode_multiselect(sub_tasks, mlb_sub_classes)
    
    # ===== RATING FEATURES =====
    acad_col = row.iloc[2] if len(row) > 2 else ""
    subfreq_col = row.iloc[4] if len(row) > 4 else ""
    refer_col = row.iloc[-4] if len(row) >= 4 else ""
    verify_col = row.iloc[-3] if len(row) >= 3 else ""
    
    acad_raw = extract_rating(acad_col)
    subfreq_raw = extract_rating(subfreq_col)
    refer_raw = extract_rating(refer_col)
    verify_raw = extract_rating(verify_col)
    
    acad_scaled = minmax_scale(acad_raw, scaler_acad_min[0], scaler_acad_scale[0])
    subfreq_scaled = minmax_scale(subfreq_raw, scaler_subfreq_min[0], scaler_subfreq_scale[0])
    refer_scaled = minmax_scale(refer_raw, scaler_refer_min[0], scaler_refer_scale[0])
    verify_scaled = minmax_scale(verify_raw, scaler_verify_min[0], scaler_verify_scale[0])
    
    ratings = np.array([[acad_scaled, subfreq_scaled, refer_scaled, verify_scaled]])
    ratings_poly = polynomial_features_manual(ratings, poly_powers)[0]
    
    # ===== COMBINE ALL =====
    x = np.hstack((vec1, vec2, vec3, best_vec, sub_vec, ratings_poly))
    
    # ===== PREDICT =====
    scores = class_log_prior + x @ feature_log_prob.T
    pred_idx = int(np.argmax(scores))
    
    return classes[pred_idx]


def predict_all(filename):
    """Predict for all rows"""
    df = pd.read_csv(filename)
    predictions = []
    
    for _, row in df.iterrows():
        pred = predict(row)
        predictions.append(pred)
    
    return predictions