import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import vectorize
from vectorize import vectorize_image
from similarity import load_predictions, calculate_cosine_similarity, get_top_k_similar

def vectorize_image(img_path):
    return vectorize.predict_image(vectorize.model, img_path)

def load_predictions(csv_path):
    return pd.read_csv(csv_path, index_col=0)

def calculate_cosine_similarity(vec, df):
    vec = vec.reshape(1, -1)
    vectors = df.values
    similarities = cosine_similarity(vec, vectors).flatten()
    return similarities

def get_top_k_similar(similarities, df, k=5):
    top_k_indices = similarities.argsort()[-k:][::-1]
    top_k_files = df.index[top_k_indices]
    top_k_scores = similarities[top_k_indices]
    return top_k_files, top_k_scores

def display_images(image_files, img_dir):
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)
        img.show()

def main(img_path, category):
    # 1. 이미지 벡터화
    img_vector = vectorize_image(img_path)

    # 2. 카테고리별 CSV 파일 경로
    csv_path = f'C:/Users/user/Desktop/K-Fashion-Recommendation-Project-main/vector_img/vectorimg/{category}.csv'
    img_dir = f'C:/Users/user/Desktop/K-Fashion-Recommendation-Project-main/vector_img/img/{category}'
    
    # 3. 기존 벡터값 로드
    df = load_predictions(csv_path)

    # 4. 코사인 유사도 계산
    similarities = calculate_cosine_similarity(img_vector, df)

    # 5. 유사도가 높은 상위 10개 값 추림
    top_files, top_scores = get_top_k_similar(similarities, df, k=10)

    # 6. 유사도가 높은 상위 10개 파일명 출력
    for i, (file, score) in enumerate(zip(top_files, top_scores), 1):
        print(f"Rank {i}: File: {file}, Cosine Similarity: {score:.4f}")

    # 7. 추려진 값의 이미지 파일 불러오기
    display_images(top_files, img_dir)

if __name__ == "__main__":
    img_path = './static/top100/id41_AGG3BL002.jpg'  # 입력 이미지 경로
    category = '상의'  # 예시로 '상의' 카테고리 사용
    main(img_path, category)


