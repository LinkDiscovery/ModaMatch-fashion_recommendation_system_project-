import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

# 이미지 경로 설정
data_dir = './data/validation/원천데이터'
output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)

# 이미지 로드 함수 정의
def load_images_from_folder(folder):
    images = []
    for subdir, dirs, files in os.walk(folder):
        for file in tqdm(files, desc='Loading images'):
            img_path = os.path.join(subdir, file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((128, 128))
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)

# 컬러 이미지 데이터 로드
images = load_images_from_folder(data_dir)

# 채널별로 PCA 적용 (배치 처리 추가)
def apply_pca_to_color_images(images, n_components=45, batch_size=100):
    reshaped_images = images.reshape(images.shape[0], -1, 3)
    pca_models = [PCA(n_components=n_components) for _ in range(3)]
    pca_results = [[] for _ in range(3)]
    for i in tqdm(range(3), desc='Applying PCA'):
        for start in range(0, reshaped_images.shape[0], batch_size):
            end = min(start + batch_size, reshaped_images.shape[0])
            batch = reshaped_images[start:end, :, i]
            pca_results[i].append(pca_models[i].fit_transform(batch))
        pca_results[i] = np.vstack(pca_results[i])
    return pca_models, pca_results

# PCA 분석 수행
pca_models, pca_results = apply_pca_to_color_images(images, n_components=45)

# 주성분 이미지 생성 및 복원 이미지 시각화 함수
def plot_pca_components_and_reconstruction(pca_models, images, pca_results, n_components_to_show=10, n_components_to_reconstruct=100):
    fig, axes = plt.subplots(3, (n_components_to_show // 2) + 1, figsize=(20, 18))
    
    original_img = images[0]
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original')
    
    for i in range(n_components_to_show):
        row = (i + 1) // ((n_components_to_show // 2) + 1)
        col = (i + 1) % ((n_components_to_show // 2) + 1)
        component_imgs = [pca_models[j].components_[i].reshape(128, 128) for j in range(3)]
        component_img = np.stack(component_imgs, axis=-1)
        component_img -= component_img.min()
        component_img /= component_img.max()
        axes[row, col].imshow(component_img)
        axes[row, col].set_title(f'PC {i + 1}')
        
        # 이미지로 저장
        component_img_pil = Image.fromarray((component_img * 255).astype(np.uint8))
        component_img_pil.save(os.path.join(output_dir, f'PC_{i + 1}.png'))
    
    # 원본 이미지와 동일한 이미지를 복원
    reconstructed_imgs = [pca_models[j].inverse_transform(pca_results[j][0]).reshape(128, 128) for j in range(3)]
    reconstructed_img = np.stack(reconstructed_imgs, axis=-1).astype(np.uint8)
    axes[2, 0].imshow(reconstructed_img)
    axes[2, 0].set_title(f'Reconstructed with {n_components_to_reconstruct} PCs')

    # 복원된 이미지 저장
    reconstructed_img_pil = Image.fromarray(reconstructed_img)
    reconstructed_img_pil.save(os.path.join(output_dir, 'Reconstructed.png'))

    plt.tight_layout()
    plt.show()

# 원본 이미지 및 주성분 이미지, 복원된 이미지 시각화
plot_pca_components_and_reconstruction(pca_models, images, pca_results, n_components_to_show=10, n_components_to_reconstruct=100)

# 주성분 1~10이 설명하는 분산 비율 출력
for channel, pca_model in zip(['Red', 'Green', 'Blue'], pca_models):
    explained_variance_ratio = pca_model.explained_variance_ratio_[:10]
    print(f"\nExplained Variance Ratio for Principal Components 1-10 in {channel} channel:")
    for i, ratio in enumerate(explained_variance_ratio, start=1):
        print(f"PC {i}: {ratio:.4f}")

