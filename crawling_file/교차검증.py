import pandas as pd
import os
import shutil

# 엑셀 파일 경로와 이미지 디렉토리 경로
excel_file_path = "./data/excel/상품정보원피스_1.xlsx"
image_folder_path = "./data/images/원피스/"
backup_folder_path = "./data/backup/"

# 백업 디렉토리 생성
if not os.path.exists(backup_folder_path):
    os.makedirs(backup_folder_path)

# 엑셀 파일 읽기
df = pd.read_excel(excel_file_path)

# 엑셀 파일의 상품 이름 리스트 생성
product_names = df['상품 이름'].tolist()

# 이미지 디렉토리의 파일 이름 리스트 생성
image_files = os.listdir(image_folder_path)
image_files = [file for file in image_files if file.endswith('.jpg') or file.endswith('.png')]  # 이미지 파일 확장자에 따라 조정

# 이미지 파일 이름에서 확장자를 제거한 상품 이름 리스트 생성
image_product_names = [os.path.splitext(file)[0] for file in image_files]

# 삭제 카운트 초기화
deleted_product_count = 0
deleted_image_count = 0

# 삭제된 엑셀 행 백업을 위한 데이터프레임
deleted_rows_df = pd.DataFrame(columns=df.columns)

# 엑셀의 상품 이름은 있으나 이미지 파일 이름이 없으면 삭제 (백업)
for product_name in product_names:
    if product_name not in image_product_names:
        deleted_rows_df = pd.concat([deleted_rows_df, df[df['상품 이름'] == product_name]], ignore_index=True)
        df = df[df['상품 이름'] != product_name]
        deleted_product_count += 1

# 삭제된 엑셀 행 백업 파일 저장
backup_excel_file_path = os.path.join(backup_folder_path, "삭제된_상품정보원피스.xlsx")
deleted_rows_df.to_excel(backup_excel_file_path, index=False)

# 엑셀의 상품 이름이 없으나 이미지 파일 이름이 있으면 삭제 (백업)
for image_file in image_files:
    image_product_name = os.path.splitext(image_file)[0]
    if image_product_name not in product_names:
        shutil.move(os.path.join(image_folder_path, image_file), os.path.join(backup_folder_path, image_file))
        deleted_image_count += 1

# 수정된 엑셀 파일 저장
df.to_excel(excel_file_path, index=False)

# 삭제된 항목 수 출력
print(f"삭제된 상품 수: {deleted_product_count}")
print(f"삭제된 이미지 수: {deleted_image_count}")
print(f"백업된 이미지 파일은 {backup_folder_path}에 저장되었습니다.")
print(f"삭제된 엑셀 행은 {backup_excel_file_path}에 저장되었습니다.")