import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PyTorch ResNet50 모델 로드 (사전 학습된 가중치 사용)
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        # 마지막 FC 레이어의 출력을 얻기 위해 레이어를 수정
        self.feature_extractor = nn.Sequential(
            *list(self.resnet50.children())[:-1],  # avg_pool까지
            nn.Flatten(),  # FC 레이어 입력을 위해 텐서를 평탄화
            self.resnet50.fc  # 마지막 FC 레이어
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features

model = ResNet50FeatureExtractor().to(device)  # 모델을 GPU로 이동
model.eval()

def preprocess_image(img_path, target_size=(224, 224)):
    """
    이미지를 전처리하여 모델 입력에 적합한 형태로 변환합니다.
    """
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가
    return img_tensor.to(device)  # 텐서를 GPU로 이동

def predict_image(model, img_path):
    """
    모델을 사용하여 이미지를 예측하고 마지막 FC 레이어의 출력을 반환합니다.
    """
    img_tensor = preprocess_image(img_path)
    features = model(img_tensor)
    features = features.squeeze().cpu().numpy()  # 배치 차원 제거 및 numpy 배열로 변환, CPU로 이동
    return features

def save_predictions_to_csv(predictions, filenames, output_path):
    """
    예측 결과와 파일명을 CSV 파일로 저장합니다.
    """
    # 데이터 프레임 생성
    df = pd.DataFrame(predictions, columns=[f'feature_{i}' for i in range(predictions.shape[1])], index=filenames)
    # CSV 파일로 저장
    df.to_csv(output_path)

if __name__ == "__main__":
    img_dir = './예시/8fbb2d45dc79b.jpg'  # 이미지가 저장된 디렉토리
    output_dir = './vectorimg'  # 벡터 값이 저장될 디렉토리

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictions = []
    filenames = []

    # 이미지 디렉토리의 모든 파일에 대해 예측 수행
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    for img_file in tqdm(img_files, desc="Processing images"):
        img_path = os.path.join(img_dir, img_file)
        
        # PyTorch 모델로 Feature Vector 추출
        prediction = predict_image(model, img_path)
        predictions.append(prediction)
        filenames.append(img_file)

    # 예측 결과를 CSV 파일로 저장
    output_csv_path = os.path.join(output_dir, './예시/8fbb2d45dc79b.jpg')
    save_predictions_to_csv(np.array(predictions), filenames, output_csv_path)

    print(f"Predictions saved to {output_csv_path}")


    