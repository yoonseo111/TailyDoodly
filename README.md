# TailyDoodly
# 어린이가 그린 크레파스 그림체 그림에 내 사진을 넣어 나만의 동화책 만들기

### 👦🏻 목표

- 어린이의 상상력 실제로 표현하기
- 아이가 원하는 이야기의 내용에 아이의 사진을 입력하여 더 생생한 동화책 생성

### 🎨 AI 알고리즘

1. **txt2img**
- 사용자에게 입력받은 동화 내용을 아동 그림체로 생성
- 별도의 fine-tuning 없이 여러 모델 test 후 선별 채택
- promt-tuning을 통해 입력하여 원하는 그림 도출

1. **YOLOv5**
- 생성된 이미지에서 사람 위치 탐지
- 아동 그림체 데이터셋으로 fine-tuning 진행
- diffusion 모델로 생성된 그림에서 사람을 직접 라벨링하여 사용 → train : validation : test = 8 : 1 : 1
- stable diffusion으로 생성된 이미지 데이터셋 150개 사용
- 16 batches, 25 epoches로 학습 진행

1. **Image Composition**
- 탐지된 사람 위치에 사용자 사진을 삽입하여 사용자 맞춤형 그림책 최종 완성

### 🧒🏻 결과

- Precision 99.4%의 결과
- 웹 서비스 구현 성공

  <img width="503" alt="image" src="https://github.com/user-attachments/assets/48c2d4ab-06ca-4679-a775-fa4dbb181e55" />
