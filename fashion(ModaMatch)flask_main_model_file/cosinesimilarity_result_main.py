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
    # 입력 벡터를 2차원 형태로 변환 (필요한 경우)
    vec = vec.reshape(1, -1)
    
    # 데이터프레임의 값을 numpy 배열 형태로 추출
    vectors = df.values
    
    # 코사인 유사도를 계산하여 1차원 배열로 변환
    similarities = cosine_similarity(vec, vectors).flatten()
    
    # 유사도 배열 반환
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

def main(img_path, csv_path, img_dir, excel_path):
    # 1. 이미지 벡터화
    img_vector = vectorize_image(img_path)

    # 2. 기존 벡터값 로드
    df = load_predictions(csv_path)

    # 3. 코사인 유사도 계산
    similarities = calculate_cosine_similarity(img_vector, df)

    # 4. 유사도가 높은 상위 10개 값 추림
    top_files, top_scores = get_top_k_similar(similarities, df, k=10)

    # 5. 엑셀 파일에서 추가 정보 로드
    excel_df = pd.read_excel(excel_path, engine='openpyxl')

    # 6. 상위 10개 유사 이미지에 대한 추가 정보 출력
    similar_images_info = []
    for file, score in zip(top_files, top_scores):
        # 파일 이름에서 확장자 제거
        file_name_without_ext = os.path.splitext(file)[0]
        
        # 파일 이름과 일치하는 행 찾기
        matched_rows = excel_df[excel_df['상품 이름'] == file_name_without_ext]
        if not matched_rows.empty:
            row = matched_rows.iloc[0]
            similar_images_info.append({
                "file": file,
                "similarity": score,
                "product_name": row['상품 이름'],
                "product_link": row['상품 링크'],
                "image_link": row['이미지 주소'],
                "price": row['상품 가격'],
                "brand": row['브랜드']
            })
        else:
            print(f"File {file} not found in Excel data.")

    # 7. 추려진 값의 이미지 파일 불러오기
    display_images(top_files, img_dir)

    # 8. 유사도가 높은 상위 10개 파일명과 인덱스 및 추가 정보 출력
    for i, info in enumerate(similar_images_info, 1):
        print(f"Rank {i}: File: {info['file']}, Cosine Similarity: {info['similarity']:.4f}")
        print(f"    Product Name: {info['product_name']}")
        print(f"    Product Link: {info['product_link']}")
        print(f"    Image Link: {info['image_link']}")
        print(f"    Price: {info['price']}")
        print(f"    Brand: {info['brand']}")

if __name__ == "__main__":
    img_path = './static/top100/id61_AFE4PT102.jpg'  # 입력 이미지 경로
    csv_path = './vector_img/vectorimg/predictions.csv'  # 기존 벡터값 CSV 파일 경로
    img_dir = './vector_img/img'  # 이미지 디렉토리 경로
    excel_path = './vector_img/vectorimg/final_excel.xlsx'  # 엑셀 파일 경로
    main(img_path, csv_path, img_dir, excel_path)


# import os
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from PIL import Image
# import vectorize

# def vectorize_image(img_path):
#     return vectorize.predict_image(vectorize.model, img_path)

# def load_predictions(csv_path):
#     return pd.read_csv(csv_path, index_col=0)

# def calculate_cosine_similarity(vec, df):
#     # 입력 벡터를 2차원 형태로 변환 (필요한 경우)
#     vec = vec.reshape(1, -1)
    
#     # 데이터프레임의 값을 numpy 배열 형태로 추출
#     vectors = df.values
    
#     # 코사인 유사도를 계산하여 1차원 배열로 변환
#     similarities = cosine_similarity(vec, vectors).flatten()
    
#     # 유사도 배열 반환
#     return similarities

# def get_top_k_similar(similarities, df, k=5):
#     top_k_indices = similarities.argsort()[-k:][::-1]
#     top_k_files = df.index[top_k_indices]
#     top_k_scores = similarities[top_k_indices]
#     return top_k_files, top_k_scores

# def display_images(image_files, img_dir):
#     for img_file in image_files:
#         img_path = os.path.join(img_dir, img_file)
#         img = Image.open(img_path)
#         img.show()

# def main(img_path, csv_path, img_dir):
#     # 1. 이미지 벡터화
#     img_vector = vectorize_image(img_path)

#     # 2. 기존 벡터값 로드
#     df = load_predictions(csv_path)

#     # 3. 코사인 유사도 계산
#     similarities = calculate_cosine_similarity(img_vector, df)

#     # 4. 유사도가 높은 상위 5개 값 추림
#     top_files, top_scores = get_top_k_similar(similarities, df)

#     # 5. 추려진 값의 이미지 파일 불러오기
#     display_images(top_files, img_dir)

#     # 6. 유사도가 높은 상위 5개 파일명과 인덱스 출력
#     for i, (file, score) in enumerate(zip(top_files, top_scores), 1):
#         print(f"Rank {i}: File: {file}, Cosine Similarity: {score:.4f}")

# if __name__ == "__main__":
#     img_path = './static/top100/id24_AFE1TS301.jpg'  # 입력 이미지 경로
#     csv_path = './vector_img/vectorimg/predictions.csv'  # 기존 벡터값 CSV 파일 경로
#     img_dir = './vector_img/img'  # 이미지 디렉토리 경로
#     main(img_path, csv_path, img_dir)