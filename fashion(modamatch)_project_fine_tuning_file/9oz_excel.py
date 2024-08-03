import os
import pandas as pd

# 이미지 파일이 있는 디렉토리 경로 설정
directory = 'data/9oz'

# 디렉토리에서 파일 이름과 경로를 수집
file_data = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            file_path = os.path.join(root, file)
            file_data.append({'File Name': file, 'File Path': file_path})

# 데이터 프레임 생성
df = pd.DataFrame(file_data)

# 엑셀 파일로 저장
output_path = 'data/9oz_image_data.xlsx'
df.to_excel(output_path, index=False)

print(f"엑셀 파일이 '{output_path}'에 저장되었습니다.")