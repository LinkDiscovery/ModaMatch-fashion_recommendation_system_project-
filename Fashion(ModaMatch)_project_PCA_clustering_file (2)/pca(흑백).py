import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

def load_images_from_directory(directory, max_images=100, image_size=(64, 64)):
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if i >= max_images:
            break
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path).convert('L')  # Grayscale conversion
        img = img.resize(image_size)
        img_array = np.asarray(img).flatten()
        images.append(img_array)
    return np.array(images)

# 경로와 이미지 수 설정
base_path = 'data/kfashiondata'
directories = os.listdir(base_path)
image_data = []

for dir_name in directories:
    dir_path = os.path.join(base_path, dir_name)
    if os.path.isdir(dir_path):
        images = load_images_from_directory(dir_path)
        image_data.append(images)

# 각 디렉토리의 이미지를 PCA로 분석
pca_results = []
pca_components = []
for images in image_data:
    pca = PCA(n_components=24)
    pca_result = pca.fit_transform(images)
    pca_results.append(pca_result)
    pca_components.append(pca.components_)

# 주성분 시각화
fig, axs = plt.subplots(3, 8, figsize=(40, 20))

for i in range(3):
    for j in range(8):
        if j < len(pca_components):
            component = pca_components[j][i].reshape(64, 64)  # 이미지 크기로 변환
            axs[i, j].imshow(component, cmap='gray')
            axs[i, j].set_title(f'Dir {j + 1} PC{i + 1}')
            axs[i, j].axis('off')
        else:
            axs[i, j].axis('off')

plt.tight_layout()
plt.savefig('./data/pca/pca_components_visualization_1___.png')  # 이미지 파일로 저장
plt.show()