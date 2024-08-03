import os
import re
import pandas as pd

# 이미지 파일이 있는 디렉토리 경로 설정
image_folder_path = './data/9oz(top1000)'

# 폴더 내의 파일 이름을 리스트로 가져오기
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# 파일 이름에서 'id' 뒤에 있는 숫자 부분을 추출하는 함수 정의
def extract_number(filename):
    match = re.search(r'id(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# 파일 이름을 숫자 부분 기준으로 정렬
image_files_sorted = sorted(image_files, key=extract_number)

# 파일 이름을 데이터프레임으로 변환
image_files_df = pd.DataFrame(image_files_sorted, columns=['파일명'])

# 엑셀 파일로 저장
output_file_path = './data/image_files_list_sorted.xlsx'
image_files_df.to_excel(output_file_path, index=False)

output_file_path