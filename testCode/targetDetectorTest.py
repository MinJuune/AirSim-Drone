import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.yolo import targetDetector

# 1. YOLO 탐지기 로드
detector = targetDetector()

# 2. 테스트용 이미지 로드 (RGB 이미지)
image_path = "testcode/image.jpg"  # 테스트할 이미지 파일 경로
image = Image.open(image_path).convert("RGB")

# 3. 이미지 전처리 → 텐서화
transform = transforms.ToTensor()
image_tensor = transform(image)  # [3, H, W]

# 4. YOLO 탐지기 실행
result = detector.detect(image_tensor)

# 5. 결과 출력
print("YOLO 탐지 결과:")
for k, v in result.items():
    print(f"{k}: {v}")
