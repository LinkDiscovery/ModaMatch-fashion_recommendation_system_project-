from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from vectorize import vectorize_image
from stylecheck import model, predict_image, get_prediction_result
from similarity import load_predictions, calculate_cosine_similarity, get_top_k_similar
from flask_cors import CORS
import time  # 소요 시간 측정을 위한 time 모듈

main = Blueprint('main', __name__)
CORS(main)  # 모든 도메인에서의 요청을 허용

# 업로드 가능한 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 파일 확장자 확인
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/businessService/recommendationStyle', methods=['POST'])
def business_service_recommendation_style():
    start_time = time.time()  # 시작 시간 기록
    if request.method == 'POST':
        img_url = request.form.get('imgURL')
        category = request.form.get('category')
        if not img_url:
            return jsonify({"error": "No imgURL provided"})
        if not category:
            return jsonify({"error": "No category provided"})

        if category not in ['상의', '하의', '아우터', '원피스']:
            return jsonify({"error": "Invalid category"})

        try:
            response = requests.get(img_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            return jsonify({"error": str(e)})
        except OSError:
            return jsonify({"error": "Invalid image format"})

        filename = secure_filename(os.path.basename(img_url))
        filepath = os.path.join(main.root_path, '..', './app/uploads', filename)
        img.save(filepath)

        prediction = predict_image(model, filepath, target_size=(256, 128))
        results = get_prediction_result(prediction)

        img_vector = vectorize_image(filepath)

        csv_path = os.path.join(main.root_path, '..', f'./vector_img/vectorimg/{category}.csv')
        df = load_predictions(csv_path)

        similarities = calculate_cosine_similarity(img_vector, df)
        top_files, top_scores = get_top_k_similar(similarities, df)

        excel_path = os.path.join(main.root_path, '..', 'vector_img', 'vectorimg', 'final_excel.xlsx')
        excel_df = pd.read_excel(excel_path, engine='openpyxl')

        similar_images_info = []
        for file, score in zip(top_files, top_scores):
            file_name_without_ext = os.path.splitext(file)[0]
            matched_rows = excel_df[excel_df['상품 이름'] == file_name_without_ext]
            if not matched_rows.empty:
                row = matched_rows.iloc[0]
                similar_images_info.append({
                    "file": file,
                    "similarity": score,
                    "file_path": f"/images/{file}",
                    "product_name": row['상품 이름'],
                    "product_link": row['상품 링크'],
                    "image_link": row['이미지 주소'],
                    "price": row['상품 가격'],
                    "brand": row['브랜드']
                })
            else:
                print(f"File {file} not found in Excel data.")

        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time  # 소요 시간 계산

        response = {
            "prediction_results": results,
            "similar_images": similar_images_info,
            "processing_time": elapsed_time  # 소요 시간을 응답에 포함
        }
        return jsonify(response)

@main.route('/customerService/recommendationStyle', methods=['POST'])
def upload_file():
    start_time = time.time()  # 시작 시간 기록
    if request.method == 'POST':
        if 'file' not in request.files:
            print('no file in request')
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(main.root_path, '..', './app/uploads', filename)
        file.save(filepath)

        prediction = predict_image(model, filepath, target_size=(256, 128))
        results = get_prediction_result(prediction)

        img_vector = vectorize_image(filepath)
        csv_path = os.path.join(main.root_path, '..', './vector_img/vectorimg/predictions.csv')
        df = load_predictions(csv_path)
        similarities = calculate_cosine_similarity(img_vector, df)
        top_files, top_scores = get_top_k_similar(similarities, df)

        excel_path = os.path.join(main.root_path, '..', 'vector_img', 'vectorimg', 'final_excel.xlsx')
        excel_df = pd.read_excel(excel_path, engine='openpyxl')

        similar_images_info = []
        for file, score in zip(top_files, top_scores):
            file_name_without_ext = os.path.splitext(file)[0]
            matched_rows = excel_df[excel_df['상품 이름'] == file_name_without_ext]
            if not matched_rows.empty:
                row = matched_rows.iloc[0]
                similar_images_info.append({
                    "file": file,
                    "similarity": score,
                    "file_path": f"/images/{file}",
                    "product_name": row['상품 이름'],
                    "product_link": row['상품 링크'],
                    "image_link": row['이미지 주소'],
                    "price": row['상품 가격'],
                    "brand": row['브랜드']
                })
            else:
                print(f"File {file} not found in Excel data.")

        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time  # 소요 시간 계산

        response = {
            "prediction_results": results,
            "similar_images": similar_images_info,
            "processing_time": elapsed_time  # 소요 시간을 응답에 포함
        }
        return jsonify(response)

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(main.root_path, '..', './app/uploads'), filename)

@main.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(main.root_path, '..', './vector_img/img'), filename)


# from flask import Blueprint, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# import os
# from vectorize import vectorize_image
# from stylecheck import model, predict_image, get_prediction_result
# from similarity import load_predictions, calculate_cosine_similarity, get_top_k_similar
# from flask_cors import CORS

# main = Blueprint('main', __name__)
# CORS(main)  # 모든 도메인에서의 요청을 허용

# # 업로드 가능한 파일 확장자
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # 파일 확장자 확인
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @main.route('/getImage', methods=['POST'])
# def upload_file():
#     # 클라이언트에서 POST 요청이 들어올 때 파일을 업로드하고 예측하는 부분
#     if request.method == 'POST':
#         # 파일이 요청에 포함되어 있는지 확인
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"})  # 파일이 없으면 에러 메시지 반환
#         file = request.files['file']  # 파일 객체 가져오기
#         if file.filename == '':  # 파일 이름이 비어 있는지 확인
#             return jsonify({"error": "No selected file"})  # 파일이 선택되지 않았으면 에러 메시지 반환
#         if file and allowed_file(file.filename):  # 파일이 존재하고, 허용된 파일 형식인지 확인
#             category = request.form.get('category')  # 폼 데이터에서 'category' 값 가져오기
#             if category not in ['상의', '하의', '아우터', '원피스']:  # 유효한 카테고리인지 확인
#                 return jsonify({"error": "Invalid category"})  # 유효하지 않으면 에러 메시지 반환

#             filename = secure_filename(file.filename)  # 파일 이름을 안전하게 처리
#             filepath = os.path.join(main.root_path, '..', './app/uploads', filename)  # 파일 저장 경로 설정
#             file.save(filepath)  # 파일을 지정된 경로에 저장

#             # 이미지 예측
#             prediction = predict_image(model, filepath, target_size=(256, 128))  # 이미지 예측 수행
#             results = get_prediction_result(prediction)  # 예측 결과 처리

#             # 1. 이미지 벡터화
#             img_vector = vectorize_image(filepath)

#             # 2. 기존 벡터값 로드
#             csv_path = os.path.join(main.root_path, '..', './vector_img/vectorimg/predictions.csv')
#             df = load_predictions(csv_path)

#             # 3. 코사인 유사도 계산
#             similarities = calculate_cosine_similarity(img_vector, df)

#             # 4. 유사도가 높은 상위 10개 값 추림
#             top_files, top_scores = get_top_k_similar(similarities, df)

#             # 결과를 JSON 형식으로 반환
#             response = {
#                 "category": category,
#                 "prediction_results": results,
#                 "similar_images": [
#                     {"file": file, "similarity": score, "file_path": f"/images/{file}"}
#                     for file, score in zip(top_files, top_scores)
#                 ]
#             }
#             return jsonify(response)

# @main.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(os.path.join(main.root_path, '..', './app/uploads'), filename)

# @main.route('/images/<filename>')
# def serve_image(filename):
#     return send_from_directory(os.path.join(main.root_path, '..', './vector_img/img'), filename)




#############################################################################유사도 잘해줌 
# from flask import Blueprint, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# import os
# from vectorize import model, vectorize_image
# from similarity import  load_predictions, calculate_cosine_similarity, get_top_k_similar

# main = Blueprint('main', __name__)

# # 업로드 가능한 파일 확장자
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # 파일 확장자 확인
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @main.route('/', methods=['GET', 'POST'])
# def upload_file():
#     # 클라이언트에서 POST 요청이 들어올 때 파일을 업로드하고 예측하는 부분
#     if request.method == 'POST':
#         # 파일이 요청에 포함되어 있는지 확인
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"})  # 파일이 없으면 에러 메시지 반환
#         file = request.files['file']  # 파일 객체 가져오기
#         if file.filename == '':  # 파일 이름이 비어 있는지 확인
#             return jsonify({"error": "No selected file"})  # 파일이 선택되지 않았으면 에러 메시지 반환
#         if file and allowed_file(file.filename):  # 파일이 존재하고, 허용된 파일 형식인지 확인
#             category = request.form.get('category')  # 폼 데이터에서 'category' 값 가져오기
#             if category not in ['상의', '하의', '아우터', '원피스']:  # 유효한 카테고리인지 확인
#                 return jsonify({"error": "Invalid category"})  # 유효하지 않으면 에러 메시지 반환

#             filename = secure_filename(file.filename)  # 파일 이름을 안전하게 처리
#             filepath = os.path.join(main.root_path, '..', './app/uploads', filename)  # 파일 저장 경로 설정
#             file.save(filepath)  # 파일을 지정된 경로에 저장

#             # 1. 이미지 벡터화
#             img_vector = vectorize_image(filepath)

#             # 2. 기존 벡터값 로드
#             csv_path = os.path.join(main.root_path, '..', './vector_img/vectorimg/predictions.csv')
#             df = load_predictions(csv_path)

#             # 3. 코사인 유사도 계산
#             similarities = calculate_cosine_similarity(img_vector, df)

#             # 4. 유사도가 높은 상위 5개 값 추림
#             top_files, top_scores = get_top_k_similar(similarities, df)

#             # 결과를 JSON 형식으로 반환
#             results = [
#                 {"file": file, "similarity": score, "file_path": f"/uploads/{file}"}
#                 for file, score in zip(top_files, top_scores)
#             ]
#             return jsonify({"category": category, "results": results})

#     # 클라이언트에서 GET 요청이 들어올 때 업로드 폼을 제공하는 부분
#     return '''
#     <!doctype html>
#     <title>Upload an Image</title>
#     <h1>Upload an Image for Style Prediction</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type="file" name="file">
#       <select name="category">
#         <option value="상의">상의</option>
#         <option value="하의">하의</option>
#         <option value="아우터">아우터</option>
#         <option value="원피스">원피스</option>
#       </select>
#       <input type="submit" value="Upload">
#     </form>
#     '''

# @main.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(os.path.join(main.root_path, '..', './app/uploads'), filename)

#######################################################################스타일 추천 잘해줌
# from flask import Blueprint, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# from stylecheck import model, predict_image, get_prediction_result

# main = Blueprint('main', __name__)

# # 업로드 가능한 파일 확장자
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # 파일 확장자 확인
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @main.route('/', methods=['GET', 'POST'])
# def upload_file():
#     # 클라이언트에서 POST 요청이 들어올 때 파일을 업로드하고 예측하는 부분
#     if request.method == 'POST':
#         # 파일이 요청에 포함되어 있는지 확인
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"})  # 파일이 없으면 에러 메시지 반환
#         file = request.files['file']  # 파일 객체 가져오기
#         if file.filename == '':  # 파일 이름이 비어 있는지 확인
#             return jsonify({"error": "No selected file"})  # 파일이 선택되지 않았으면 에러 메시지 반환
#         if file and allowed_file(file.filename):  # 파일이 존재하고, 허용된 파일 형식인지 확인
#             category = request.form.get('category')  # 폼 데이터에서 'category' 값 가져오기
#             if category not in ['상의', '하의', '아우터', '원피스']:  # 유효한 카테고리인지 확인
#                 return jsonify({"error": "Invalid category"})  # 유효하지 않으면 에러 메시지 반환

#             filename = secure_filename(file.filename)  # 파일 이름을 안전하게 처리
#             filepath = os.path.join(main.root_path, '..', './app/uploads', filename)  # 파일 저장 경로 설정
#             file.save(filepath)  # 파일을 지정된 경로에 저장

#             # 이미지 예측
#             prediction = predict_image(model, filepath, target_size=(256, 128))  # 이미지 예측 수행
#             results = get_prediction_result(prediction)  # 예측 결과 처리

#             # 예측 결과를 JSON 형식으로 반환
#             return jsonify({"category": category, "results": results})
    
#     # 클라이언트에서 GET 요청이 들어올 때 업로드 폼을 제공하는 부분
#     return '''
#     <!doctype html>
#     <title>Upload an Image</title>
#     <h1>Upload an Image for Style Prediction</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type="file" name="file">  # 파일 업로드 입력 필드
#       <select name="category">  # 카테고리 선택 드롭다운 메뉴
#         <option value="상의">상의</option>
#         <option value="하의">하의</option>
#         <option value="아우터">아우터</option>
#         <option value="원피스">원피스</option>
#       </select>
#       <input type="submit" value="Upload">  # 제출 버튼
#     </form>
#     '''