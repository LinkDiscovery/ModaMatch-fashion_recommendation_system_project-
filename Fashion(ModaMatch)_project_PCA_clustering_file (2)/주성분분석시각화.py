import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

# 이미지 폴더 경로
image_folder = './data/kfashiondata'

# 이미지 데이터를 담을 리스트
image_data = []
image_paths = []  # 원본 이미지 경로를 저장할 리스트
image_labels = []  # 원본 이미지의 디렉토리 이름을 저장할 리스트

# 이미지 크기를 조정할 크기 (모든 이미지를 동일한 크기로 변환)
image_size = (64, 64)

# 각 디렉토리에서 최대 100개의 이미지만 불러오기
max_images_per_directory = 100

# 모든 하위 디렉토리의 이미지를 불러오기
for root, dirs, files in os.walk(image_folder):
    file_count = 0
    for file in tqdm(files, desc=f'Processing {root}'):
        if file_count >= max_images_per_directory:
            break
        if file.endswith(".jpg") or file.endswith(".png"):  # 확장자에 맞게 추가
            image_path = os.path.join(root, file)
            with Image.open(image_path) as img:
                img = img.resize(image_size)  # 이미지 크기 조정
                img = img.convert('L')  # 흑백 이미지로 변환
                img_data = np.asarray(img).flatten()  # 이미지를 1차원 배열로 변환
                image_data.append(img_data)
                image_paths.append(image_path)  # 이미지 경로 저장
                image_labels.append(os.path.basename(root))  # 디렉토리 이름 저장
                file_count += 1

# 리스트를 numpy 배열로 변환
image_data = np.array(image_data)

# 주성분 개수 설정
n_components = 10  # 설명력이 높은 주성분 개수로 설정

# PCA 실행
pca = PCA(n_components=n_components)
pca.fit(image_data)

# 주성분 벡터 시각화
for i in range(n_components):
    plt.figure(figsize=(4, 4))
    plt.imshow(pca.components_[i].reshape(image_size), cmap='gray')
    plt.title(f'Principal Component {i+1}')
    plt.colorbar()
    plt.show()

# 주성분을 사용하여 원본 데이터를 변환
transformed_data = pca.transform(image_data)
inverse_transformed_data = pca.inverse_transform(transformed_data)

# 변환된 이미지 시각화 (원본 데이터와 비교)
n_samples_to_show = 10  # 비교할 샘플 수 설정

plt.figure(figsize=(20, 4))
for i in range(n_samples_to_show):
    # 원본 이미지
    ax = plt.subplot(2, n_samples_to_show, i + 1)
    plt.imshow(image_data[i].reshape(image_size), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # 변환된 이미지
    ax = plt.subplot(2, n_samples_to_show, i + 1 + n_samples_to_show)
    plt.imshow(inverse_transformed_data[i].reshape(image_size), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.show()