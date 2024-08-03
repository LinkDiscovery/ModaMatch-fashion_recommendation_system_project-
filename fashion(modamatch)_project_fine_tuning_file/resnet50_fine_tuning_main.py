import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.keras import TqdmCallback
import pandas as pd
import numpy as np
import os

# 엑셀 데이터 불러오기
file_path = './data/labeling.xlsx'
data = pd.read_excel(file_path)

# 이미지 파일 경로 설정
image_dir = './data/9oz(1000)_remove_background'

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

# 데이터프레임 생성
train_df = pd.DataFrame({'filename': train_paths})
val_df = pd.DataFrame({'filename': val_paths})

# 레이블 데이터프레임 생성
label_df = pd.DataFrame(labels)
label_columns = label_df.columns.tolist()

# 레이블을 데이터프레임에 추가
train_labels_df = pd.DataFrame(train_labels, columns=label_columns)
val_labels_df = pd.DataFrame(val_labels, columns=label_columns)
train_df = pd.concat([train_df, train_labels_df], axis=1)
val_df = pd.concat([val_df, val_labels_df], axis=1)

# 데이터 생성기
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col=label_columns,
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col=label_columns,
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

# ResNet50 모델 불러오기 (사전 학습된 가중치 사용)
base_model = ResNet50(weights='imagenet', include_top=False)

# 새로운 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(labels.shape[1], activation='sigmoid')(x)  # 다중 클래스 분류

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 일부 레이어를 학습하지 않도록 고정
for layer in base_model.layers:
    layer.trainable = False

# 특징 추출 활용: 사전 학습된 모델은 대규모 데이터셋(예: ImageNet)에서 학습되어 다양한 이미지 특징을 이미 잘 학습하고 있습니다. 
# 이러한 특징을 재사용하여 새로운 데이터셋에 적용할 수 있습니다. 따라서, 이 특징 추출 레이어들을 고정하여 학습 시간을 줄이고, 데이터가 부족할 때도 좋은 성능을 얻을 수 있습니다.

# 과적합 방지: 모든 레이어를 학습시키면 모델이 새로운 데이터셋에 과적합(overfitting)될 가능성이 높아집니다. 특히, 데이터셋이 작을 경우에는 더욱 그렇습니다. 
# 사전 학습된 모델의 레이어를 고정하면 과적합을 줄이는 데 도움이 됩니다.

# 효율적인 학습: 사전 학습된 모델의 고정된 레이어를 사용하면, 모델의 마지막 레이어들만 학습하게 되어 학습 시간이 단축되고, 컴퓨팅 리소스를 절약할 수 있습니다.

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약 출력
model.summary()

# 콜백 설정
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# 모델 훈련
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, TqdmCallback(verbose=1)]
)

# 모델 평가
loss, accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# 모델 저장 (명시적으로 저장하는 코드 추가)
model.save('final_model.h5')
