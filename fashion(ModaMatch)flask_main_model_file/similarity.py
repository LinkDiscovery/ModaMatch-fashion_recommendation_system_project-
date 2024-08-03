import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_predictions(csv_path):
    return pd.read_csv(csv_path, index_col=0)

def calculate_cosine_similarity(vec, df):
    vec = vec.reshape(1, -1)
    vectors = df.values
    similarities = cosine_similarity(vec, vectors).flatten()
    return similarities

def get_top_k_similar(similarities, df, k=10):
    top_k_indices = similarities.argsort()[-k:][::-1]
    top_k_files = df.index[top_k_indices]
    top_k_scores = similarities[top_k_indices]
    return top_k_files, top_k_scores