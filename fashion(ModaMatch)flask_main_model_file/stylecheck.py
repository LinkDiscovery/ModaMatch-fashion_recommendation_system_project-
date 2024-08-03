import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 로드
model = load_model('./model/best_model_final.h5')

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
    return img_array

def predict_image(model, img_path, target_size):
    img_array = preprocess_image(img_path, target_size)
    prediction = model.predict(img_array)
    return prediction

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

    top_indices = prediction[0].argsort()[-3:][::-1]
    top_values = prediction[0][top_indices]
    labels = list(caption.keys())

    results = []
    for i, index in enumerate(top_indices):
        label = labels[index]
        prob = top_values[i]
        description = caption[label]
        results.append({
            "rank": i + 1,
            "style": label,
            "probability": f"{prob * 100:.2f}%"  # 확률 값을 퍼센트로 포맷팅
        })
    return results


