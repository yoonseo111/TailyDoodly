import torch
from PIL import Image
import pathlib
import os
import argparse
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.PosixPath

def load_yolov5_model(model_path) :
    fine_tuning_model = pathlib.Path(model_path).resolve()
    fine_tuning_model = str(fine_tuning_model)
    model = torch.hub.load('ultralytics/yolov5', 'custom', force_reload=True, path = fine_tuning_model)
    return model

def detect_and_crop_people(model_path, image_path, output_dir):
    # 모델 불러오기
    model = load_yolov5_model(model_path)
    print('model load done')
    #path = background_path + '/' + image_path
    # 이미지 로드
    image = Image.open(image_path)

    # YOLOv5로 탐지
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()  # 탐지된 박스 정보 (x1, y1, x2, y2, confidence, class)

    # COCO 클래스에서 "person" 클래스는 0번 인덱스
    person_class = 0
    bbox_o = False
    #bbox 추출 코드 및 txt 저장
    name = image_path.split('.')[0]
    format_ = image_path.split('.')[1]
    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if int(cls) == person_class and confidence > 0.5:  # 신뢰도 필터링
            # 좌표를 정수로 변환
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            with open(f"{output_dir}" , "w") as f:
                f.write(f"{x1} {y1} {x2} {y2} {name}_GT.{format_}")
            bbox_o = True
            break
    if bbox_o == False :
        x1, y1, x2, y2 = map(int, [128, 128, 384, 512])
        with open(f"{output_dir}" , "w") as f:
            f.write(f"{x1} {y1} {x2} {y2} {name}_GT.{format_}")

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help ="custom yolo model path",
        default = "custom.pt"
    )
    parser.add_argument(
        "--imagedir",
        type=str,
        help="background image path",
        default="0_txt2img_output"
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        help = "background detection txt file ouput path",
        default = "1_yolo_output"
    )
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = argument_parse()
    image_txt = os.listdir(opt.imagedir)
    output_dir = opt.outputdir
    model_path = opt.model
    for image in image_txt :
        image_name = image.split('.')[0]
        image_name = image_name + ".txt"
        image_path = os.path.join(opt.imagedir, image)
        output = os.path.join(output_dir, image_name)
        detect_and_crop_people(model_path, image_path, output)