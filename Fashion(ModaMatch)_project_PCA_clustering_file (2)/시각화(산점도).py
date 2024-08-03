import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import seaborn as sns

# 이미지 폴더 경로
image_folder = './data/kfashiondata/상하원아'

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

# 데이터가 올바르게 불러와졌는지 확인
print(f"Loaded {len(image_data)} images.")

# 리스트를 numpy 배열로 변환
image_data = np.array(image_data)

# 데이터가 2D 배열인지 확인
print(f"Image data shape: {image_data.shape}")

# PCA 실행
if len(image_data) > 0:
    pca = PCA(n_components=10)  # 주성분 10개로 설정
    principal_components = pca.fit_transform(image_data)

    # 결과를 데이터프레임으로 변환
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(10)])
    pca_df['Image_Path'] = image_paths  # 원본 이미지 경로를 데이터프레임에 추가
    pca_df['Label'] = image_labels  # 디렉토리 이름을 데이터프레임에 추가

    # 여러 쌍의 주성분에 대한 산점도 그리기 함수
    def plot_pca_scatter(pca_df, pc_x, pc_y, save_path):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=pc_x, y=pc_y, hue='Label', data=pca_df, palette='tab10', s=50, alpha=0.8)
        plt.title(f'PCA: {pc_x} vs {pc_y}')
        plt.xlabel(pc_x)
        plt.ylabel(pc_y)
        plt.legend(title='Directory')
        plt.savefig(save_path)
        plt.show()

    # 모든 조합에 대해 산점도 그리기
    pc_pairs = [
        ('PC1', 'PC2'),
        ('PC1', 'PC3'),
        ('PC2', 'PC3'),
    ]

    # 폴더가 존재하지 않을 경우 생성
    output_folder = './data/pca2/'
    os.makedirs(output_folder, exist_ok=True)

    # 산점도 생성 및 저장
    for pc_x, pc_y in pc_pairs:
        save_path = os.path.join(output_folder, f'scatterwhite_{pc_x}_vs_{pc_y}.png')
        plot_pca_scatter(pca_df, pc_x, pc_y, save_path)

# 주성분 1 vs 주성분 2 산점도 그리기
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Label', data=pca_df, palette='tab10', s=50, alpha=0.8)
plt.title('PCA: Principal Component 1 vs Principal Component 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Directory')
plt.savefig('./data/pca2/scatterwhite_상하원아.png')
plt.show()





