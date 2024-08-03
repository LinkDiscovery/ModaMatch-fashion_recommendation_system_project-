import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# 이미지 폴더 경로
image_folder = './data/흑백분류/'

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
                img = img.convert('RGB')  # 흑백 이미지로 변환
                img_data = np.asarray(img).flatten()  # 이미지를 1차원 배열로 변환
                image_data.append(img_data)
                image_paths.append(image_path)  # 이미지 경로 저장
                image_labels.append(os.path.basename(root))  # 디렉토리 이름 저장
                file_count += 1

# 리스트를 numpy 배열로 변환
image_data = np.array(image_data)

# 주성분 개수 설정
n_components = 3  # 3차원 시각화를 위해 주성분 3개로 설정

# PCA 실행
pca = PCA(n_components=n_components)
transformed_data = pca.fit_transform(image_data)

# 3D 스캐터 플롯 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 각 디렉토리 이름(라벨)에 따라 색상을 다르게 설정
unique_labels = list(set(image_labels))
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
label_color_map = {label: color for label, color in zip(unique_labels, colors)}

for label in unique_labels:
    indices = [i for i, l in enumerate(image_labels) if l == label]
    ax.scatter(
        transformed_data[indices, 0], 
        transformed_data[indices, 1], 
        transformed_data[indices, 2], 
        label=label, 
        color=label_color_map[label]
    )

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Image Data')
ax.legend()
plt.show()