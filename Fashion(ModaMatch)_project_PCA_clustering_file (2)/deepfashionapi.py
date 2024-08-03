import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 환경 변수 설정 (OpenMP 오류 해결)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# YOLOv5 모델 로드 (사전 훈련된 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.to('cpu')  # 명시적으로 CPU로 설정

# 이미지 로드 및 전처리
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image):
    return np.array(image)

# 객체 탐지
def detect_objects(model, image):
    results = model(image)
    return results

# 결과 시각화
def visualize_predictions(image, results, threshold=0.5):
    np_image = np.array(image)
    detections = results.xyxy[0].numpy()

    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if score >= threshold:
            cv2.rectangle(np_image, 
                          (int(x1), int(y1)), 
                          (int(x2), int(y2)), 
                          (255, 0, 0), 2)
            cv2.putText(np_image, 
                        f'{model.names[int(class_id)]}: {score:.2f}', 
                        (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 0, 0), 
                        2)

    plt.figure(figsize=(12, 8))
    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

# 이미지 경로
image_path = '12200046_eh.jpg'

# 이미지 로드 및 전처리
image = load_image(image_path)
input_image = preprocess_image(image)

# 객체 탐지
results = detect_objects(model, input_image)

# 결과 시각화
visualize_predictions(image, results)

# import torch
# import torchvision.transforms as transforms
# from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import os

# # 환경 변수 설정 (OpenMP 오류 해결)
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # DeepFashion2 모델 로드 (최신 권장 방식 사용)
# weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
# model = fasterrcnn_resnet50_fpn(weights=weights)
# model.eval()

# # 이미지 로드 및 전처리
# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return image

# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     return transform(image).unsqueeze(0)

# # 객체 탐지
# def detect_objects(model, image):
#     with torch.no_grad():
#         predictions = model(image)
#     return predictions[0]

# # 결과 시각화
# def visualize_predictions(image, predictions, threshold=0.5):
#     np_image = np.array(image)
#     labels = predictions['labels'].numpy()
#     boxes = predictions['boxes'].detach().numpy()
#     scores = predictions['scores'].detach().numpy()

#     for label, box, score in zip(labels, boxes, scores):
#         if score >= threshold:
#             cv2.rectangle(np_image, 
#                           (int(box[0]), int(box[1])), 
#                           (int(box[2]), int(box[3])), 
#                           (255, 0, 0), 2)
#             cv2.putText(np_image, 
#                         f'{label}: {score:.2f}', 
#                         (int(box[0]), int(box[1] - 10)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 
#                         0.5, 
#                         (255, 0, 0), 
#                         2)

#     plt.figure(figsize=(12, 8))
#     plt.imshow(np_image)
#     plt.axis('off')
#     plt.show()

# # 이미지 경로
# image_path = '12200046_eh.jpg'

# # 이미지 로드 및 전처리
# image = load_image(image_path)
# input_image = preprocess_image(image)

# # 객체 탐지
# predictions = detect_objects(model, input_image)

# # 결과 시각화
# visualize_predictions(image, predictions)