import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm

# CPU를 사용할 경우 장치 설정
device = torch.device('cpu')

# 클래스 라벨을 숫자 인덱스로 매핑 (기타 제외)
class_to_idx = {
    '매니시': 0, '모던': 1, '밀리터리': 2, '섹시': 3, '소피스트케이티드': 4, '스트리트': 5,
    '스포티': 6, '아방가르드': 7, '오리엔탈': 8, '웨스턴': 9, '젠더리스': 10, '컨트리': 11,
    '클래식': 12, '키치': 13, '톰보이': 14, '펑크': 15, '페미닌': 16, '프레피': 17, '히피': 18, '힙합': 19,
    '레트로': 20, '로맨틱': 21, '리조트': 22
}

# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_files = []
        self.labels = []
        
        for class_name in os.listdir(root_dir):
            if class_name not in class_to_idx:
                continue
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = class_to_idx[class_name]
                for img_file in os.listdir(class_dir):
                    if img_file.endswith('.jpg'):
                        self.img_files.append(os.path.join(class_dir, img_file))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 이미지 전처리 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 커스텀 데이터셋 및 데이터로더 생성 (학습 데이터)
train_dataset = CustomDataset(root_dir='data/Training/원천데이터', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 커스텀 데이터셋 및 데이터로더 생성 (검증 데이터)
val_dataset = CustomDataset(root_dir='data/Validation/원천데이터', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 모델 정의 및 수정 (ResNet50)
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=23):
        super(ResNet50Classifier, self).__init__()
        # 이전 pretrained=True 대신 weights=ResNet50_Weights.DEFAULT 사용
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

# 모델 인스턴스 생성 및 장치 설정
model = ResNet50Classifier(num_classes=23)
model = model.to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 검증 함수 정의
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False, position=1, ncols=100)
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(batch=batch_idx, val_loss=val_loss/(batch_idx+1), val_accuracy=100 * correct / total)
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

# 모델 학습 함수 정의
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]', position=0, ncols=100)
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(batch=batch_idx, train_loss=running_loss/(batch_idx+1))
        
        train_loss = running_loss / len(train_loader)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# 학습된 모델 저장 경로 설정
model_save_path = './saved_models/resnet50_classifier.pth'

# 경로가 존재하지 않으면 폴더 생성
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 모델 학습 실행
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# 학습된 모델 저장
torch.save(model.state_dict(), model_save_path)

# 예측 함수 정의
def predict_image(model, image_path, class_to_idx, device='cuda'):
    model.eval()  # 모델을 평가 모드로 설정
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 클래스 인덱스를 클래스 이름으로 변환
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        results = {idx_to_class[idx]: prob for idx, prob in enumerate(probabilities)}
        
        # 가장 높은 확률을 가진 클래스 추출
        predicted_class_name = max(results, key=results.get)
        top_prob_percentage = results[predicted_class_name] * 100
        
        return predicted_class_name, top_prob_percentage, results

# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 배치 차원을 추가합니다.
    return image

# 특정 이미지에 대해 예측 수행
image_path = '.12200046_eh.jpg'  # 예측할 이미지 파일 경로
predicted_class_name, top_prob_percentage, results = predict_image(model, image_path, class_to_idx, device)

print(f"Predicted class: {predicted_class_name}")
print(f"Confidence: {top_prob_percentage:.2f}%")
print("\nClass probabilities:")
for class_name, prob in results.items():
    print(f"{class_name}: {prob * 100:.2f}%")
