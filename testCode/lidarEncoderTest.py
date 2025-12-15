import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th
from models.globalFeatureExtractor import GlobalFeatureExtractor  # 모델 클래스 import

# 1. 라이다 인코더 모델 로드
encoder = GlobalFeatureExtractor(input_dim=3)
encoder.load_state_dict(th.load("weights/global_feature_extractor.pth", map_location=th.device('cpu')))
encoder.eval()  # 학습 아님, 추론 모드로 설정

# 2. 더미 입력 생성: 배치 4개, 각 샘플은 1024개의 (x, y, z) 포인트
batch_size = 4
num_points = 1024
dummy_lidar = th.rand(batch_size, num_points, 3)  # shape: [4, 1024, 3]

# 3. 모델에 입력 → 출력 feature 추출
with th.no_grad():  # 추론 시 gradient 계산 비활성화
    output = encoder(dummy_lidar)  # 예상 출력: [4, 1024]

# 4. 결과 출력
print("입력 shape:", dummy_lidar.shape)      # [4, 1024, 3] → 배치 4개, 포인트 1024개, 차원 3(x, y, z)
print("출력 shape:", output.shape)           # [4, 1024] → 배치 4개, 각 샘플은 1024차원 feature vector
