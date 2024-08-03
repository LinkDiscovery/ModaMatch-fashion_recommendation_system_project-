import os
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
from tqdm import tqdm

# 사용자 정의 옵티마이저 정의
class CustomAdam(tf.keras.optimizers.Adam):
    def __init__(self, weight_decay=0.0, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, jit_compile=False, is_legacy_optimizer=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_decay = weight_decay
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.ema_overwrite_frequency = ema_overwrite_frequency
        self.jit_compile = jit_compile
        self.is_legacy_optimizer = is_legacy_optimizer

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_decay": self.weight_decay, 
            "use_ema": self.use_ema, 
            "ema_momentum": self.ema_momentum,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "jit_compile": self.jit_compile,
            "is_legacy_optimizer": self.is_legacy_optimizer
        })
        return config

# 사용자 정의 옵티마이저 로드
custom_objects = {'Custom>Adam': CustomAdam}

# TensorFlow 모델 로드 및 입력 크기 확인
def load_keras_model(model_path):
    model = load_model(model_path, custom_objects=custom_objects)
    input_shape = model.input_shape[1:3]  # (height, width)
    return model, input_shape

# 이미지 전처리 함수 (TensorFlow 모델용)
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
    return img_array

# 예측 함수 (TensorFlow 모델용)
def predict_image(model, img_path, target_size):
    img_array = preprocess_image(img_path, target_size)
    prediction = model.predict(img_array)
    return prediction

# 예측 결과 해석 함수
def get_prediction_result(prediction):
    caption = {
        '바캉스': '휴일이나 여행지에서 입을 것 같은 느슨하고 편안한 개방적인 스타일 입니다.', 
        '보헤미안': '자유분방한 생활을 즐기는 유랑인 보헤미안의 스타일을 뜻하며 헐렁한 레이어드와 패치, 자수가 특징인 스타일 입니다.', 
        '섹시': '신체의 노출이 많거나 몸에 꼭 맞는 의상이 특징인 여성스러운 스타일 입니다.', 
        '스포티': '자연스럽고 건강해 보이며 입기에 편한 기능성을 중요시하는 활동적인 스타일 입니다.', 
        '오피스룩': '직장인들이 선호하는 깔끔하고 편안하면서 세련된 스타일입니다.', 
        '캐주얼': '격식에 메이지 않고 가볍고 부담 없이 입을 수 있는 자연스러운 스타일 입니다.', 
        '트레디셔널': '세련되고 도회적이며 여유로운 편안함을 조화롭게 추구하는 신사복 스타일 입니다.', 
        '페미닌': '각 시대에 맞는 여성스러움을 표현한 우아하고 러블리한 분위기의 스타일 입니다.', 
        '힙합': '스트릿 문화에 기반한 자신의 개성을 살린 자유로운 분위기의 스타일 입니다.'
    }

    # 예측 결과 길이와 캡션 딕셔너리의 길이를 비교합니다.
    if len(prediction[0]) != len(caption):
        print(f"Error: Prediction length {len(prediction[0])} does not match the number of labels {len(caption)}")
        return []

    # 예측 결과에서 상위 3개의 인덱스를 내림차순으로 정렬하여 가져옵니다.
    top_indices = prediction[0].argsort()[-3:][::-1]

    # 캡션 딕셔너리의 키들을 리스트로 변환합니다.
    labels = list(caption.keys())

    # 상위 3개 인덱스에 해당하는 스타일 라벨을 저장합니다.
    top_labels = [labels[index] for index in top_indices]

    return top_labels

# 예측할 이미지 디렉토리 경로 및 타겟 사이즈 지정
img_dir = r'./vector_img/img'
output_dir = r'./classification'

# 이미지 파일 리스트
image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
random.shuffle(image_files)  # 파일 리스트를 랜덤으로 섞음
image_files = image_files[:10000]  # 상위 10000개의 파일 선택

# TensorFlow 모델 로드 및 입력 크기 확인
keras_model, target_size = load_keras_model(r'./model/best_resnet50_fashion.h5')
print(f'Model expects input size: {target_size}')

# 각 이미지에 대해 예측 수행 및 결과 출력
for image_file in tqdm(image_files, desc="Processing Images"):  # tqdm 적용
    img_path = os.path.join(img_dir, image_file)
    prediction = predict_image(keras_model, img_path, target_size)
    top_labels = get_prediction_result(prediction)
    
    if not top_labels:
        continue  # 예측 결과에 오류가 있을 경우 다음 이미지를 처리합니다.
    
    for label in top_labels:
        style_dir = os.path.join(output_dir, label)
        if not os.path.exists(style_dir):
            os.makedirs(style_dir)
        
        destination_path = os.path.join(style_dir, image_file)
        shutil.copy(img_path, destination_path)