import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

# 이미지 경로 설정
data_dir = 'data/kfashiondata'

# 이미지 로드 함수 정의
def load_images_from_folder(folder):
    images = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = Image.open(img_path).convert('L')  # 흑백 이미지로 변환
            img = img.resize((128, 128))  # 크기 조정 (2배로 키움)
            img_array = np.array(img).flatten()  # 1차원 배열로 변환
            images.append(img_array)
    return np.array(images)

# 이미지 데이터 로드
images = load_images_from_folder(data_dir)

# PCA 분석 수행
pca = PCA(n_components=100)
pca_result = pca.fit_transform(images)

# 주성분 이미지 생성 및 복원 이미지 시각화 함수
def plot_pca_components_and_reconstruction(pca, images, pca_result, n_components_to_show=10, n_components_to_reconstruct=100):
    fig, axes = plt.subplots(3, (n_components_to_show // 2) + 1, figsize=(20, 18))
    
    original_img = images[0].reshape(128, 128)
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original')
    
    for i in range(n_components_to_show):
        row = (i + 1) // ((n_components_to_show // 2) + 1)
        col = (i + 1) % ((n_components_to_show // 2) + 1)
        component_img = pca.components_[i].reshape(128, 128)
        axes[row, col].imshow(component_img, cmap='gray')
        axes[row, col].set_title(f'PC {i + 1}')
    
    reconstructed_img = pca.inverse_transform(pca_result[0]).reshape(128, 128)
    axes[2, 0].imshow(reconstructed_img, cmap='gray')
    axes[2, 0].set_title(f'Reconstructed with {n_components_to_reconstruct} PCs')

    plt.tight_layout()
    plt.show()

# 원본 이미지 및 주성분 이미지, 복원된 이미지 시각화
plot_pca_components_and_reconstruction(pca, images, pca_result, n_components_to_show=10, n_components_to_reconstruct=100)