import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from tqdm import tqdm

# 이미지 폴더 경로
image_folder = './data/kfashiondata/'

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

# t-SNE 실행
if len(image_data) > 0:
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(image_data)

    # t-SNE 결과를 데이터프레임으로 변환
    tsne_df = pd.DataFrame(data=tsne_embedding, columns=['TSNE1', 'TSNE2'])
    tsne_df['Image_Path'] = image_paths  # 원본 이미지 경로를 데이터프레임에 추가
    tsne_df['Label'] = image_labels  # 디렉토리 이름을 데이터프레임에 추가

    # t-SNE 결과 시각화 (이미지 포함)
    fig, ax = plt.subplots(figsize=(15, 12))
    for i in range(len(tsne_df)):
        img_path = tsne_df.iloc[i]['Image_Path']
        try:
            img = Image.open(img_path)
            img = img.resize((30, 30))  # 이미지 크기를 조정합니다
            imagebox = OffsetImage(img, zoom=1)
            ab = AnnotationBbox(imagebox, (tsne_df.iloc[i]['TSNE1'], tsne_df.iloc[i]['TSNE2']), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    plt.title('t-SNE: TSNE1 vs TSNE2')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.savefig('./data/tsne_embedding_images.png')
    plt.show()
else:
    print("No images found. Please check the image folder path and contents.")