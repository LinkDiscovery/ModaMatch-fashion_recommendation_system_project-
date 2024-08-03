import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os

# 엑셀 데이터 불러오기
file_path = './data/labeling.xlsx'
data = pd.read_excel(file_path)

# 이미지 파일 경로 설정
image_dir = './data/9oz(top1000)'

# 데이터프레임에서 이미지 파일 경로와 레이블 추출
data['파일명'] = data['파일명'].apply(lambda x: os.path.join(image_dir, x))
file_paths = data['파일명'].values
labels = data.drop('파일명', axis=1).values

# 학습 및 검증 데이터 생성
from sklearn.model_selection import train_test_split
train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# ImageDataGenerator를 사용하여 이미지 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 생성기
train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({'filename': train_paths, 'class': list(train_labels)}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    pd.DataFrame({'filename': val_paths, 'class': list(val_labels)}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

