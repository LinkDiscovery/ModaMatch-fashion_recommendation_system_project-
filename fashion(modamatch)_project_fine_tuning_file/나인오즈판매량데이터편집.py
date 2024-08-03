import pandas as pd
from tqdm import tqdm

# tqdm의 pandas 확장을 불러옴
tqdm.pandas()

# 엑셀 파일 경로 설정
file_path = './data/고객판매일보상세(세로).xlsx'
# 엑셀 파일 읽기
xls = pd.ExcelFile(file_path)

# '판매' 시트만 사용
sales_data = pd.read_excel(xls, sheet_name='판매', header=1)

# 관심 있는 열만 선택 (상품코드, 상품명, 판매금액, 수량)
filtered_sales_data = sales_data[['상품코드', '상품명', '판매금액', '수량']]

# 상품코드와 상품명을 기준으로 그룹화하고 수량 합계 계산
aggregated_sales_data = filtered_sales_data.groupby(['상품코드', '상품명']).progress_apply(lambda x: pd.Series({
    '판매금액': x['판매금액'].iloc[0],  # 같은 상품코드에 대해 첫 번째 판매금액을 사용
    '수량': x['수량'].sum()  # 수량을 합산
})).reset_index()

# 엑셀 파일을 작성하고 '판매' 시트만 저장
output_file_path = './data/aggregated_sales_data2.xlsx'
aggregated_sales_data.to_excel(output_file_path, index=False)

output_file_path