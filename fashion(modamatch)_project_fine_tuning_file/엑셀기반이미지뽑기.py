import pandas as pd
import os
from shutil import copy2
from tqdm import tqdm

# 엑셀 파일 경로
excel_path = './data/top1000.xlsx'
# 이미지 파일이 있는 디렉토리 경로
image_dir = './data/9oz'
# 저장할 디렉토리 경로
output_dir = './data/9oz(top1000)'

# 엑셀 파일 로드
df = pd.read_excel(excel_path)

# 상품코드와 id 리스트 추출
product_codes = df['상품코드'].astype(str).tolist()
ids = df['id'].astype(str).tolist()

# output_dir이 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# tqdm을 사용하여 파일명 비교 및 파일 복사
for product_code, id_value in tqdm(zip(product_codes, ids), desc="Processing files", total=len(product_codes)):
    image_path = os.path.join(image_dir, f'{product_code}.jpg')
    if os.path.exists(image_path):
        new_image_path = os.path.join(output_dir, f'id{id_value}_{product_code}.jpg')
        copy2(image_path, new_image_path)
        tqdm.write(f'Copied: {image_path} to {new_image_path}')
    else:
        tqdm.write(f'File not found: {image_path}')


