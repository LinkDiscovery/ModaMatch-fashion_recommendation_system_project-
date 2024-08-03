import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import random

# 이미지 경로 설정
data_dir = './data/9oztrainingdata'

# 클래스와 색상 매핑
label_color_map = {'top': 'red', 'bottom': 'green', 'outer': 'blue', 'dress': 'purple'}

# 이미지 로드 함수 정의
def load_images_and_labels(folder):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        label = os.path.basename(subdir)
        if label in label_color_map:  # 유효한 클래스만 처리
            for file in tqdm(files, desc=f'Loading images from {label}'):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path).convert('RGB')  # 컬러 이미지로 변환
                img = img.resize((128, 128))  # 크기 조정
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

# 컬러 이미지 데이터 및 라벨 로드
images, labels = load_images_and_labels(data_dir)

# 1000개의 랜덤 이미지 선택
if len(images) > 1000:
    selected_indices = random.sample(range(len(images)), 1000)
    images = images[selected_indices]
    labels = labels[selected_indices]

# 채널별로 PCA 적용
def apply_pca_to_color_images(images, n_components=100):
    reshaped_images = images.reshape(images.shape[0], -1, 3)
    pca_models = [PCA(n_components=n_components) for _ in range(3)]
    pca_results = []
    for i in range(3):
        pca_results.append(pca_models[i].fit_transform(reshaped_images[:, :, i]))
    return pca_models, pca_results

# PCA 분석 수행
_, pca_results = apply_pca_to_color_images(images, n_components=100)

# DBSCAN 클러스터링 및 3D 시각화 함수
def apply_dbscan_and_plot(pca_results, labels, eps=0.5, min_samples=5):
    # 3개의 주성분을 사용하여 클러스터링
    pca_features = np.array([pca_results[i][:, :3] for i in range(3)])
    pca_features = np.concatenate(pca_features, axis=1)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(pca_features)
    
    # 라벨 색상 매핑
    colors = [label_color_map[label] for label in labels]

    # 3D 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c=colors, s=50)

    # 범례 추가
    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10, label=label)
               for label, color in label_color_map.items()]
    ax.legend(handles=handles, title="Classes")

    ax.set_title('DBSCAN Clustering on PCA Features')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.show()

# DBSCAN 클러스터링 및 시각화 수행
apply_dbscan_and_plot(pca_results, labels, eps=0.5, min_samples=5)
