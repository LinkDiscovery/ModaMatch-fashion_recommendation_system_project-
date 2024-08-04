# K-digital training ModaMatch Project
---
#### 개발 기간 : **2024.07.04 ~ 2024.07.26**
 ---
#### 프로젝트 소개 :
 > Moda Match는 AI 기반의 의류 추천 웹 어플리케이션으로, 사용자들이 원하는 스타일의 
  의류를 쉽게 찾을 수 있도록 도와줍니다. ‘Business Service’, ‘Customer Service’ 두 개
  의 Main Service가 존재하며, ‘Business Service’는 의류업체를 대상으로 ‘Customer 
  Service’는 일반 사용자를 대상으로 서비스를 제공합니다.
---
#### 팀 소개
|이름|역할|GIT URL|
|------|---|---|
|신건영|데이터 분석|https://github.com/LinkDiscovery/ModaMatch-fashion_recommendation_system_project-/tree/main|
|장민우|프론트엔드||
|김보성|백엔드||
|채수철|백엔드보조||
---
#### 데이터 분석 진행일기(Notion)
> https://www.notion.so/ModaMatch-ec9f7a15a4f24dbbb080e141b7677f8c
--- 
#### 시연 영상
> https://youtu.be/99uts-qYaJM - customer service

> https://youtu.be/yHtFunHfXCg - business service
---
#### 웹 개발 페이지 
#### Main Page
![image02](https://github.com/user-attachments/assets/efe70e4d-9196-4370-af4a-77711914fb07)
#### Service Page
![main service](https://github.com/user-attachments/assets/945ffca2-1360-4f0b-b9e4-24d6223f8d0d)
#### Result & Like Page
![result like](https://github.com/user-attachments/assets/24082a2e-a566-4a80-9777-58e9e9526783)



---
#### 시스템 모듈 구성
![캡처](https://github.com/user-attachments/assets/41013127-f9ca-4b85-9489-feebceceb62e)
1c528)

---

#### 분석 결과 개선점 및 한계점

  1. 산업 특성별 추천코드 개수 적용의 어려움
     
     >해당 분석 결과에서는 모든 기업에 대하여 20개의 HS부호를 추천하도록 하였음. 하지만 산업의 고도화 정도에 따라 추천될 수 있는 HS부호 개수가 다른 것이 정확도가 높을 것으로 판단됨.
  
  2. 텍스트 데이터 주요 단어 추출 시 무역 전문가와의 협업 필요
     
     >텍스트 데이터 전처리 과정과 주요 단어 추출 및 벡터화 과정에서 하나의 텍스트 데이터에서 어떤 주요 단어가 추출되느냐에 따라 추천 HS코드에 영향이 있는 바, 해당업에서의 전문가와 협업한다면 정확도가 높아질 것임. 

  3. 정밀도는 높으나 정확도 개선 필요
      
     >추천 HS부호의 정밀도(몰려있는 정도)는 높으나 다양한 HS부호가 나올 수 있도록 정확도를 개선한다면 실용도가 더 높아질 수 있을 것이라 판단됨
---
#### EER Diagram
![image](https://github.com/LinkDiscovery/HScodeMappingProject/assets/154401566/e8672ee7-af22-4c7b-a045-190b08198fbe)
---

#### Architecture
![image](https://github.com/LinkDiscovery/HScodeMappingProject/assets/154401566/75ddedc0-472b-437c-a98c-962cadb03bb2)
---
#### Rest Api List
![image](https://github.com/LinkDiscovery/HScodeMappingProject/assets/154401566/046e19c8-a78f-4147-b571-4535aa2b0475)
---
### 기술 스택

- **프로그래밍 언어**: Python version: 3.11.9

- **데이터 처리 및 분석**:
  - Pandas version: 2.2.2
  - NumPy version: 1.26.4
 
- **머신러닝**:
  - Scikit-learn version: 1.5.0
    
- **데이터베이스**:
  - mySQL
 
- **개발 도구 및 환경**:
  - Jupyter Notebook
  - Visual Studio Code
    
- **버전 관리**:
  - Git
  - GitHub
    
- **Back End**:
![image](https://github.com/LinkDiscovery/HScodeMappingProject/assets/154401566/9b2c121b-c71e-465a-be75-831f77e91cb6)
