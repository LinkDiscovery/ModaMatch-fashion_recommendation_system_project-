import os
from backgroundremover.bg import remove
from PIL import Image
import io

# 원본 이미지 디렉토리와 저장할 디렉토리 설정
input_dir = './top100'
output_dir = './data/9oz_top100_remove_background'

# 저장할 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 배경 제거 함수
def remove_background(image_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    try:
        with open(image_path, "rb") as f:
            data = f.read()
            img = remove(data, model_name=model_choices[0], alpha_matting=True,
                         alpha_matting_foreground_threshold=240,
                         alpha_matting_background_threshold=10,
                         alpha_matting_erode_structure_size=10,
                         alpha_matting_base_size=1000)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# 배경 제거된 이미지를 저장하는 함수
def save_image(image_data, output_path):
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img.save(output_path, format='PNG')
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")

# 모든 이미지에 대해 배경 제거 및 저장
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    img_data = remove_background(input_path)
    if img_data:
        save_image(img_data, output_path)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Failed to process: {input_path}")
