import pandas as pd
import os

# 엑셀 파일들이 있는 디렉토리 경로와 병합된 파일을 저장할 경로
excel_files_dir = "./data/excel/"
merged_excel_file_path = "./data/excel/병합된_상품정보.xlsx"

# 병합할 엑셀 파일 목록
excel_files = [
    "상품정보아우터_1.xlsx",
    "상품정보상의_1.xlsx",
    "상품정보하의_1.xlsx",
    "상품정보원피스_1.xlsx"
]

# 모든 엑셀 파일을 읽어 데이터프레임 리스트로 저장
dfs = []
for file in excel_files:
    file_path = os.path.join(excel_files_dir, file)
    df = pd.read_excel(file_path)
    dfs.append(df)

# 데이터프레임들을 하나로 병합
merged_df = pd.concat(dfs, ignore_index=True)

# 병합된 데이터프레임을 엑셀 파일로 저장
merged_df.to_excel(merged_excel_file_path, index=False)

print(f"병합된 엑셀 파일이 {merged_excel_file_path}에 저장되었습니다.")