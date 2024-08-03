from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import urllib.request
import re

# 크롬 드라이버 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저를 백그라운드에서 실행
chrome_options.add_argument("--no-sandbox")  # 보안 모드 비활성화
chrome_options.add_argument("--disable-dev-shm-usage")  # 공유 메모리 사용 비활성화

# 크롬 드라이버 초기화
driver = webdriver.Chrome(options=chrome_options)

# 저장 디렉토리 설정
save_path = './data/excel'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 이미지 저장 디렉토리 설정
image_save_path = './data/images/기프티박스하의'
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

# 크롤링 결과를 저장할 리스트 초기화
image_links = []
product_names = []
product_prices = []
product_links = []
collected_links = set()  # 수집된 링크를 저장하는 집합

# 특수 문자를 제거하는 함수
def clean_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

# 데이터를 저장하는 함수
def make_df(image_links, product_names, product_prices, product_links):
    if len(image_links) == 0 or len(product_names) == 0 or len(product_prices) == 0 or len(product_links) == 0:
        print("No data to save.")
        return
    
    # 수집한 데이터를 데이터프레임으로 변환
    product_df = pd.DataFrame({
        '이미지 주소': image_links,
        '상품 이름': product_names,
        '상품 가격': product_prices,
        '상품 링크': product_links
    })
    # 데이터프레임을 엑셀 파일로 저장
    file_path = os.path.join(save_path, f"상품정보하의(기프티박스).xlsx")
    product_df.to_excel(file_path, index=False)
    print(f"{file_path} 파일이 저장되었습니다.")

# URL을 동적으로 생성하여 페이지를 순회하면서 크롤링 수행
base_url = 'https://gifteabox.com/product/list.html?cate_no=89&page='
page_num = 1
total_collected_count = 0  # 전체 수집된 데이터의 갯수

while page_num <= 13:
    url = base_url + str(page_num)
    driver.get(url)
    
    # 페이지가 완전히 로딩되도록 잠시 기다림
    time.sleep(3)
    
    # 페이지가 존재하지 않을 경우 루프 종료
    if "페이지를 찾을 수 없습니다" in driver.page_source or "Not Found" in driver.page_source:
        print(f"Page {page_num} does not exist. Stopping the crawl.")
        break
    
    # 페이지 소스를 BeautifulSoup으로 파싱
    html = driver.page_source
    bs_obj = BeautifulSoup(html, "html.parser")
    
    # 제품 정보를 포함하는 모든 요소를 찾음
    products = bs_obj.find_all('div', class_='box')
    
    if not products:
        print(f"No products found on page {page_num}. Stopping the crawl.")
        break
    
    current_collected_count = 0
    
    for product in products:
        try:
            # 필요한 데이터를 추출
            product_link_tag = product.find('a', class_='')
            if not product_link_tag:
                continue
            link_suffix = product_link_tag['href']
            full_link = 'https://gifteabox.com' + link_suffix
            
            # 이미 수집된 링크인지 확인
            if full_link in collected_links:
                continue
            
            img_tag = product.find('img', {'id': lambda x: x and x.startswith('eListPrdImage')})
            if img_tag:
                image_url = 'https:' + img_tag['src']
                product_name = img_tag['alt']
            else:
                continue
            
            # 가격 추출
            product_price_element = product.find('li', rel='판매가')
            if product_price_element:
                price_span = product_price_element.find_all('span', style='font-size:12px;color:#333333;')
                if len(price_span) > 1:
                    product_price = price_span[1].text.strip()
                else:
                    product_price = price_span[0].text.strip()
            else:
                continue
            
            # 파일 이름에 사용할 수 없는 문자를 제거
            cleaned_product_name = clean_filename(product_name)
            
            # 추출한 데이터를 리스트에 추가 (값이 세 개 다 있는 경우에만 추가)
            if full_link and product_name and product_price:
                image_links.append(image_url)
                product_names.append(cleaned_product_name)
                product_prices.append(product_price)
                product_links.append(full_link)
                
                # 이미지 저장 (값이 세 개 다 있는 경우에만)
                image_save_full_path = os.path.join(image_save_path, f"{cleaned_product_name}.jpg")
                urllib.request.urlretrieve(image_url, image_save_full_path)
                
                collected_links.add(full_link)  # 링크를 수집된 링크 집합에 추가
                current_collected_count += 1
            
        except Exception as e:
            print(f"Error parsing product on page {page_num}: {e}")
            continue
    
    total_collected_count += current_collected_count
    
    # 디버깅용 출력: 데이터 리스트 길이 확인
    print(f"Processed page {page_num}, collected {total_collected_count} images, {len(product_names)} names, {len(product_prices)} prices.")
    
    # 수집된 데이터의 갯수가 증가하지 않으면 크롤링 종료
    if current_collected_count == 0:
        print("No more data collected. Stopping the crawl.")
        break
    
    page_num += 1  # 다음 페이지로 이동

# 최종 데이터 저장
make_df(image_links, product_names, product_prices, product_links)

# 동작이 끝나면 크롬 드라이버 종료
driver.quit()



# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from bs4 import BeautifulSoup
# import time
# import pandas as pd
# import os
# import urllib.request
# import re

# # 크롬 드라이버 옵션 설정
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # 브라우저를 백그라운드에서 실행
# chrome_options.add_argument("--no-sandbox")  # 보안 모드 비활성화
# chrome_options.add_argument("--disable-dev-shm-usage")  # 공유 메모리 사용 비활성화

# # 크롬 드라이버 초기화
# driver = webdriver.Chrome(options=chrome_options)

# # 저장 디렉토리 설정
# save_path = './data/excel'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# # 이미지 저장 디렉토리 설정
# image_save_path = './data/images/기프티박스하의'
# if not os.path.exists(image_save_path):
#     os.makedirs(image_save_path)

# # 크롤링 결과를 저장할 리스트 초기화
# image_links = []
# product_names = []
# product_prices = []
# collected_links = set()  # 수집된 링크를 저장하는 집합
# num = 1  # 저장된 파일의 인덱스 번호

# # 특수 문자를 제거하는 함수
# def clean_filename(filename):
#     # 한글, 영문, 숫자, 공백 외의 모든 문자를 제거
#     return re.sub(r'[^가-힣a-zA-Z0-9\s]', "", filename)

# # 데이터를 저장하는 함수
# def make_df(image_links, product_names, product_prices, num):
#     if len(image_links) == 0 or len(product_names) == 0 or len(product_prices) == 0:
#         print("No data to save.")
#         return
    
#     # 수집한 데이터를 데이터프레임으로 변환
#     product_df = pd.DataFrame(list(zip(image_links, product_names, product_prices)), columns=['이미지 주소', '상품 이름', '상품 가격'])
#     # 데이터프레임을 엑셀 파일로 저장
#     file_path = os.path.join(save_path, f"상품정보하의(기프티박스)_{num}.xlsx")
#     product_df.to_excel(file_path, index=False)
#     print(f"{file_path} 파일이 저장되었습니다.")

# # URL을 동적으로 생성하여 페이지를 순회하면서 크롤링 수행
# base_url = 'https://gifteabox.com/product/list.html?cate_no=89&page='
# page_num = 1
# total_collected_count = 0  # 전체 수집된 데이터의 갯수

# while page_num <= 13:
#     url = base_url + str(page_num)
#     driver.get(url)
    
#     # 페이지가 완전히 로딩되도록 잠시 기다림
#     time.sleep(3)
    
#     # 페이지가 존재하지 않을 경우 루프 종료
#     if "페이지를 찾을 수 없습니다" in driver.page_source or "Not Found" in driver.page_source:
#         print(f"Page {page_num} does not exist. Stopping the crawl.")
#         break
    
#     # 페이지 소스를 BeautifulSoup으로 파싱
#     html = driver.page_source
#     bs_obj = BeautifulSoup(html, "html.parser")
    
#     # 제품 정보를 포함하는 모든 요소를 찾음
#     products = bs_obj.find_all('div', class_='box')
    
#     if not products:
#         print(f"No products found on page {page_num}. Stopping the crawl.")
#         break
    
#     current_collected_count = 0
    
#     for i, product in enumerate(products):
#         try:
#             # 필요한 데이터를 추출
#             product_link_tag = product.find('a', class_='')
#             if not product_link_tag:
#                 continue
#             link_suffix = product_link_tag['href']
#             full_link = 'https://gifteabox.com' + link_suffix
            
#             # 이미 수집된 링크인지 확인
#             if full_link in collected_links:
#                 continue
            
#             img_tag = product.find('img', {'id': lambda x: x and x.startswith('eListPrdImage')})
#             if img_tag:
#                 image_url = 'https:' + img_tag['src']
#                 product_name = img_tag['alt']
#             else:
#                 continue
            
#             # 가격 추출
#             product_price_element = product.find('li', rel='판매가')
#             if product_price_element:
#                 price_span = product_price_element.find_all('span', style='font-size:12px;color:#333333;')
#                 if len(price_span) > 1:
#                     product_price = price_span[1].text.strip()
#                 else:
#                     product_price = price_span[0].text.strip()
#             else:
#                 continue
            
#             # 파일 이름에 사용할 수 없는 문자를 제거
#             cleaned_product_name = clean_filename(product_name)
            
#             # 추출한 데이터를 리스트에 추가 (값이 세 개 다 있는 경우에만 추가)
#             if full_link and product_name and product_price:
#                 image_links.append(image_url)
#                 product_names.append(cleaned_product_name)
#                 product_prices.append(product_price)
                
#                 # 이미지 저장 (값이 세 개 다 있는 경우에만)
#                 image_save_full_path = os.path.join(image_save_path, f"{cleaned_product_name}.jpg")
#                 urllib.request.urlretrieve(image_url, image_save_full_path)
                
#                 collected_links.add(full_link)  # 링크를 수집된 링크 집합에 추가
#                 current_collected_count += 1
            
#         except Exception as e:
#             print(f"Error parsing product on page {page_num}: {e}")
#             continue
        
#         # 300개 단위로 데이터를 저장
#         if len(image_links) % 300 == 0:
#             make_df(image_links, product_names, product_prices, num)
#             num += 1  # 저장된 파일 번호 증가
#             # 리스트 초기화
#             image_links = []
#             product_names = []
#             product_prices = []
    
#     total_collected_count += current_collected_count
    
#     # 디버깅용 출력: 데이터 리스트 길이 확인
#     print(f"Processed page {page_num}, collected {total_collected_count} images, {len(product_names)} names, {len(product_prices)} prices.")
    
#     # 수집된 데이터의 갯수가 증가하지 않으면 크롤링 종료
#     if current_collected_count == 0:
#         print("No more data collected. Stopping the crawl.")
#         break
    
#     page_num += 1  # 다음 페이지로 이동

# # 최종 데이터 저장
# if len(image_links) > 0:
#     make_df(image_links, product_names, product_prices, num)

# # 동작이 끝나면 크롬 드라이버 종료
# driver.quit()

