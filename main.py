## 필요 라이브러리 설치
import os
from pathlib import Path
import json
import sys
import subprocess # model 파일 수정하지 않고, 그대로 가져와서 사용 가능
import torch
import shutil

## 결과 저장 폴더 생성

# 저장 폴더 구조 정의
def create_folder_structure(base_path, user_name):
    folder_structure = {
        f"{base_path}/{user_name}": [
            "txt2img",
            "yolo",
            {
                "composition": [
                    "input/background",
                    "input/bbox",
                    "input/foreground",
                    "input/foreground_mask",
                    "output"
                ]
            }
        ]
    }
    
    def create_folders(path, structure):
        for item in structure:
            if isinstance(item, dict):
                for subfolder, substructure in item.items():
                    subfolder_path = os.path.join(path, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)
                    create_folders(subfolder_path, substructure)
            else:
                folder_path = os.path.join(path, item)
                os.makedirs(folder_path, exist_ok=True)
    
    for main_folder, subfolders in folder_structure.items():
        os.makedirs(main_folder, exist_ok=True)
        create_folders(main_folder, subfolders)

# 폴더 생성 실행
base_path = Path(__file__).parent  # main.py 상위 폴더에 저장
user_name = "minseo" # user 이름 수정

create_folder_structure(base_path, user_name)

## jason에서 caption 가져오기

# JSON 파일 경로 설정 (현재 경로: tailydoodly/final/{user_name}/response.json)
json_file_path = f'/home/tailydoodly/final/{user_name}/response.json'  # 실제 파일 이름으로 변경

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# caption 출력
captions = []
for page in data.get('pages', []):
    caption = page.get('caption', [])
    captions.extend(caption)

# txt2img_modify.py 실행

temp_prompts_path = base_path / 'temp_prompts.json'
with open(temp_prompts_path, 'w', encoding='utf-8') as file:
    json.dump(captions, file, ensure_ascii=False, indent=4)

txt2img_script_path = '/home/tailydoodly/model/txt2img_modify.py'  # 스크립트의 절대 경로
txt2img_output_dir = base_path / user_name / 'txt2img'

subprocess.run([
    'python3', str(txt2img_script_path),
    '--prompts', str(temp_prompts_path),
    '--output', str(txt2img_output_dir)
], check=True)

# YOLO 실행

yolo_output_dir = base_path / user_name / 'yolo'
yolo_model_path = '/home/tailydoodly/yolov5_best_0113.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)

for image_path in txt2img_output_dir.glob('*.png'):
    results = model(image_path)

    # 결과에서 bounding box 좌표 추출
    detections = results.xyxy[0].cpu().numpy()  # 탐지된 박스 정보 (x1, y1, x2, y2, confidence, class)

    person_class = 0
    bbox_o = False
    image_name = image_path.stem
    output_txt_path = yolo_output_dir / f"{image_name}.txt"
    with open(output_txt_path, 'w') as f:
        for detection in detections:
            x1, y1, x2, y2, confidence, cls = detection
            if int(cls) == person_class and confidence > 0.5:   
                f.write(f"{int(x1)} {int(y1)} {int(x2)} {int(y2)} {confidence:.2f} {int(cls)}\n")
                bbox_o = True
                break
        if bbox_o == False:
            f.write(f"128 128 384 512 {int(cls)}\n")

    # YOLO 결과를 composition 입력 폴더로 복사
    composition_input_dir = base_path / user_name / 'composition' / 'input'
    background_dir = composition_input_dir / 'background'
    bbox_dir = composition_input_dir / 'bbox'

    # 배경 이미지 복사
    shutil.copy(image_path, background_dir / image_path.name)

    # bbox 좌표 파일 복사
    shutil.copy(output_txt_path, bbox_dir / output_txt_path.name)

# Composition 실행
composition_script_path = '/home/tailydoodly/model/composition_modify.py'
composition_output_dir = base_path / user_name / 'composition' / 'output'

subprocess.run([
    'python3', str(composition_script_path),
    '--testdir', str(composition_input_dir),
    '--outdir', str(composition_output_dir)
], check=True)