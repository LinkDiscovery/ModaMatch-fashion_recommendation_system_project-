import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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
                img = img.convert('RGB')  # 흑백 이미지로 변환
                img_data = np.asarray(img).flatten()  # 이미지를 1차원 배열로 변환
                image_data.append(img_data)
                image_paths.append(image_path)  # 이미지 경로 저장
                image_labels.append(os.path.basename(root))  # 디렉토리 이름 저장
                file_count += 1

# 리스트를 numpy 배열로 변환
image_data = np.array(image_data)

# PCA 실행
pca = PCA(n_components=10)  # 주성분 10개로 설정
principal_components = pca.fit_transform(image_data)

# 결과를 데이터프레임으로 변환
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(10)])
pca_df['Image_Path'] = image_paths  # 원본 이미지 경로를 데이터프레임에 추가
pca_df['Label'] = image_labels  # 디렉토리 이름을 데이터프레임에 추가

# 첫 번째 주성분에 따라 데이터 정렬
sorted_pca_df = pca_df.sort_values(by='PC2', ascending=False)

# 상위 100개의 이미지 경로를 가져오기
top_100_image_paths = sorted_pca_df.head(100)['Image_Path'].values
top_100_labels = sorted_pca_df.head(100)['Label'].values

# 상위 100개의 원본 이미지를 시각화
fig, axs = plt.subplots(2, 8, figsize=(80, 40))  # 10행 10열의 서브플롯 생성

for i, ax in enumerate(axs.flat):
    img_path = top_100_image_paths[i]
    label = top_100_labels[i]
    with Image.open(img_path) as img:
        img = img.resize(image_size)  # 이미지 크기 조정
        ax.imshow(img, cmap='gray')  # 원본 이미지를 회색조로 표시
        ax.set_title(label, fontsize=8)  # 라벨 추가
        ax.axis('off')  # 축 눈금과 축 라벨 숨김

plt.suptitle('Top 100 Images Corresponding to the First Principal Component')
plt.savefig('./data/pca/pca_white_2_1.png')  # 이미지 파일로 저장
plt.show()




