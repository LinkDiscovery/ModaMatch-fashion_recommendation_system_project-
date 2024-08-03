import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 이미지 폴더 경로 변경
image_folder = './data/9oz(top1000)/'

# 이미지 데이터를 담을 리스트
image_data = []

# 이미지 크기를 조정할 크기 (모든 이미지를 동일한 크기로 변환)
image_size = (64, 64)

# 모든 하위 디렉토리의 이미지를 불러오기
for root, dirs, files in os.walk(image_folder):
    for file in tqdm(files, desc=f'Processing {root}'):
        if file.endswith(".jpg") or file.endswith(".png"):  # 확장자에 맞게 추가
            image_path = os.path.join(root, file)
            with Image.open(image_path) as img:
                img = img.resize(image_size)  # 이미지 크기 조정
                img = img.convert('L')  # 흑백 이미지로 변환
                img_data = np.asarray(img).flatten()  # 이미지를 1차원 배열로 변환
                image_data.append(img_data)

# 리스트를 numpy 배열로 변환
image_data = np.array(image_data)

# PCA 실행
n_components = 150
pca = PCA(n_components=n_components)  # 주성분 10개로 설정
pca.fit(image_data)

# 주성분들의 설명 분산 비율과 누적 설명 분산 비율
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# 설명 분산 비율 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, n_components + 1), cumulative_explained_variance_ratio, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('./data/pca_9oz/explained_variance_ratio.png')
plt.show()





