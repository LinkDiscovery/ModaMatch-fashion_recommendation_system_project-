import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.keras import TqdmCallback
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 이미지 파일 경로 설정
image_dir = './data/kfashiondata'

# 디렉토리 내 모든 이미지 파일 경로와 클래스 레이블 추출
file_paths = []
labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_paths.append(os.path.join(root, file))
            labels.append(os.path.basename(root))  # 디렉토리 이름을 클래스 레이블로 사용

# 데이터를 데이터프레임으로 변환
data = pd.DataFrame({'filename': file_paths, 'class': labels})

# 학습 및 검증 데이터 생성
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class'])

# ImageDataGenerator를 사용하여 이미지 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 생성기
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ResNet50 모델 불러오기 (사전 학습된 가중치 사용)
base_model = ResNet50(weights='imagenet', include_top=False)

# 새로운 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)  # 다중 클래스 분류

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 일부 레이어를 학습하지 않도록 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

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

# 모델 저장
model.save('final_model.h5')
