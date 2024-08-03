import os
from backgroundremover.bg import remove
from PIL import Image
import io
from tqdm import tqdm

# 원본 이미지 디렉토리와 저장할 디렉토리 설정
input_dir = './vector_img/img'
output_dir = './vector_img/img_backgroundremove'

# 저장할 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 배경 제거 함수
def remove_background(image_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    try:
        with open(image_path, "rb") as f:
            data = f.read()
            # alpha_matting으로 투명 배경 생성
            img = remove(data, model_name=model_choices[0], alpha_matting=True,
                         alpha_matting_foreground_threshold=240,
                         alpha_matting_background_threshold=10,
                         alpha_matting_erode_structure_size=10,
                         alpha_matting_base_size=1000)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    
# # 배경 제거 함수
# def remove_background(image_path):
#     model_choices = ["u2net", "u2net_human_seg", "u2netp"]
#     try:
#         with open(image_path, "rb") as f:
#             data = f.read()
#             img = remove(data, model_name=model_choices[0], alpha_matting=True,
#                          alpha_matting_foreground_threshold=240,
#                          alpha_matting_background_threshold=10,
#                          alpha_matting_erode_structure_size=10,
#                          alpha_matting_base_size=1000)
#         return img
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return None

# 배경 제거된 이미지를 저장하는 함수
def save_image(image_data, output_path):
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGBA")  # RGBA로 변환하여 투명 배경 지원
        img.save(output_path, format='PNG')  # PNG 포맷으로 저장하여 투명도 유지
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")

# 이미지 파일 확장자
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

# 모든 이미지에 대해 배경 제거 및 저장
files = [file for file in os.listdir(input_dir) if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]

batch_size = 32

for i in tqdm(range(0, len(files), batch_size), desc="Processing images"):
    batch_files = files[i:i + batch_size]
    for filename in batch_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")  # PNG 확장자로 저장
        img_data = remove_background(input_path)
        if img_data:
            save_image(img_data, output_path)
        else:
            print(f"Failed to process: {input_path}")

# import os
# from backgroundremover.bg import remove
# from PIL import Image
# import io
# from tqdm import tqdm

# # 원본 이미지 디렉토리와 저장할 디렉토리 설정
# input_dir = './vector_img/img'
# output_dir = './vector_img/img_backgroundremove'

# # 저장할 디렉토리가 없으면 생성
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 배경 제거 함수
# def remove_background(image_path):
#     model_choices = ["u2net", "u2net_human_seg", "u2netp"]
#     try:
#         with open(image_path, "rb") as f:
#             data = f.read()
#             img = remove(data, model_name=model_choices[0], alpha_matting=True,
#                          alpha_matting_foreground_threshold=240,
#                          alpha_matting_background_threshold=10,
#                          alpha_matting_erode_structure_size=10,
#                          alpha_matting_base_size=1000)
#         return img
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return None

# # 배경 제거된 이미지를 저장하는 함수
# def save_image(image_data, output_path):
#     try:
#         img = Image.open(io.BytesIO(image_data)).convert("RGB")
#         img.save(output_path, format='PNG')
#     except Exception as e:
#         print(f"Error saving image {output_path}: {e}")

# # 이미지 파일 확장자
# IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

# # 모든 이미지에 대해 배경 제거 및 저장
# files = [file for file in os.listdir(input_dir) if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]

# batch_size = 32

# for i in tqdm(range(0, len(files), batch_size), desc="Processing images"):
#     batch_files = files[i:i + batch_size]
#     for filename in batch_files:
#         input_path = os.path.join(input_dir, filename)
#         output_path = os.path.join(output_dir, filename)
#         img_data = remove_background(input_path)
#         if img_data:
#             save_image(img_data, output_path)
#         else:
#             print(f"Failed to process: {input_path}")

