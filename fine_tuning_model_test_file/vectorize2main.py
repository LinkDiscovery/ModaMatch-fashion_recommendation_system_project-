import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from backgroundremover.bg import remove
from PIL import Image
import io

# 모델 로드
model = load_model('./model/final_model.h5')

def remove_bg(src_img_path, out_img_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    with open(src_img_path, "rb") as f:
        data = f.read()
        img = remove(data, model_name=model_choices[0],
                     alpha_matting=True,
                     alpha_matting_foreground_threshold=240,
                     alpha_matting_background_threshold=10,
                     alpha_matting_erode_structure_size=10,
                     alpha_matting_base_size=1000)
    with open(out_img_path, "wb") as f:
        f.write(img)

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
    return img_array

def predict_image(model, img_path, target_size):
    img_array = preprocess_image(img_path, target_size)
    prediction = model.predict(img_array)
    return prediction

def get_top_style(prediction):
    top_index = prediction[0].argmax()
    labels = ['바캉스', '보헤미안', '섹시', '스포티', '오피스룩', '캐주얼', '트레디셔널', '페미닌', '힙합']
    top_style = labels[top_index]
    return top_style

def load_images_from_directory(directory, target_size):
    image_vectors = []
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img_array = preprocess_image(img_path, target_size)
            vector = model.predict(img_array)
            image_vectors.append(vector[0])
            image_paths.append(img_path)
    return np.array(image_vectors), image_paths

def vectorize_image(model, img_path, target_size):
    img_array = preprocess_image(img_path, target_size)
    vector = model.predict(img_array)
    return vector[0]

def calculate_cosine_similarity(input_vector, style_vectors):
    similarities = cosine_similarity([input_vector], style_vectors)
    return similarities

def get_top_similar_images(similarities, image_paths, top_n=10):
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    top_images = [image_paths[i] for i in top_indices]
    return top_images

# 메인 실행 코드
img_path = './top100/id8_AFI3KN001.jpg'  # 입력 이미지 파일 경로
target_size = (224, 224)  # 모델에 맞는 타겟 사이즈
nobg_img_path = img_path.replace('.jpg', '_nobg.png').replace('.png', '_nobg.png')

# 배경 제거
remove_bg(img_path, nobg_img_path)

# 스타일 예측
prediction = predict_image(model, nobg_img_path, target_size)
top_style = get_top_style(prediction)

# 스타일 디렉토리 설정
style_directory = os.path.join('./classification', top_style)

# 스타일 디렉토리에서 이미지 로드 및 벡터화
style_vectors, image_paths = load_images_from_directory(style_directory, target_size)

# 입력 이미지 벡터화
input_vector = vectorize_image(model, nobg_img_path, target_size)

# 코사인 유사도 계산
similarities = calculate_cosine_similarity(input_vector, style_vectors)

# 유사도가 높은 10개의 이미지 가져오기
top_similar_images = get_top_similar_images(similarities, image_paths)

# 결과 출력 및 시각화
print(f"Predicted Style: {top_style}")
for i, img_path in enumerate(top_similar_images):
    print(f"Rank {i+1}: {img_path}")

# 시각적으로 확인할 수 있도록 이미지 출력
plt.figure(figsize=(20, 10))
for i, img_path in enumerate(top_similar_images):
    img = image.load_img(img_path)
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"Rank {i+1}")
    plt.axis('off')
plt.savefig('top_similar_images.png')
plt.show()


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