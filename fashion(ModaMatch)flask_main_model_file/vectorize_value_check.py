import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(self.resnet50.children())[:-1],  # avg_pool까지
            nn.Flatten(),  # FC 레이어 입력을 위해 텐서를 평탄화
            self.resnet50.fc  # 마지막 FC 레이어
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features

model = ResNet50FeatureExtractor()
model.eval()

def preprocess_image(img_path, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가
    return img_tensor

def predict_image(model, img_path):
    img_tensor = preprocess_image(img_path)
    features = model(img_tensor)
    features = features.squeeze().numpy()  # 배치 차원 제거 및 numpy 배열로 변환
    return features

if __name__ == "__main__":
    img_path = './static/top100/id2_AGD3PT004.jpg'  # 예시 이미지 경로
    features = predict_image(model, img_path)
    print(features)