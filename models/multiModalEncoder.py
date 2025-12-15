import torch as th
import torch.nn as nn
import gymnasium as gym
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

from models.globalFeatureExtractor import GlobalFeatureExtractor
from models.yolo import targetDetector
from utils.lidarPolarCoordinate import lidarPolarCoordinate
import config.hyperparameter as hyperparameter
import config.config as config

# 채널 평균 모듈
class ChannelAveraging(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # [배치사이즈,채널수,높이,너비] -> [배치사이즈,1,높이,너비]
        return x.mean(dim=1, keepdim=True)

# PPO에서 카메라 + 라이다를 전처리 후 멀티모달 정책 네트워크에 전달하는 Feature Extractor
class MultiModalEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1364)  # 최종 Feature 크기

        # 1. CNN 인코더 정의(사전학습된 ResNet18 모델 기반)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 사전학습된 ResNet18 로드
        self.cnn_encoder = nn.Sequential(
            *list(resnet.children())[:-2],  # [B,512,15,20]. Avgpool전에 잘라 avgpool,fc 제거. 입력: [B,3,480,640]-> 출력: [B,512,15,20] 
            ChannelAveraging(),             # [B,1,15,20]. 채널 평균 
            nn.Flatten()                    # [B,300]. (B,1,15,20)을 (B,300) 형태의 1D 벡터로 변환
        )
        # [:-1]로 슬라이스 하면 출력 shape: [B,512,1,1]
        # [:-2]로 슬라이스 하면 출력 shape: [B,512,15,20]

        # 2. PointNet (라이다 데이터 처리)
        self.lidar_encoder = GlobalFeatureExtractor(input_dim=3)
        self.lidar_encoder.load_state_dict(th.load("weights/global_feature_extractor.pth", map_location=th.device('cpu')))  
        # 사전학습된 가중치 로드
        # 만약 GPU에서 학습한 모델을 CPU에서 사용하려면 map_location=th.device('cpu') 추가해줘야함
        self.lidar_encoder.eval()  # 학습 X, inference 모드
        # 입력: [B,1024,3] -> 출력: [B,1024]

        # 3. YOLO 클래스 초기화
        # 입력: 단일 이미지 1장 -> [3,480,640]
        # 출력: YOLO 탐지 결과 dict 형태 -> {"detected":float, "area_ratio":float, "offset_score":float}
        # [1,3] -> [B,3]으로 변환을 해야 하는데 이거는 forward에서 처리 
        self.targetDetector = targetDetector()

        # # 4. 최종 Feature Fusion (CNN Feature + MLP Feature)
        # self.final_layer = nn.Sequential(
        #     nn.Linear(300 + 1024 + 3, 512),  # 이미지(300)+라이다(1024)+YOLO(3) -> 512.
        #     # PPO 학습 과정에서 중요한 feature에 더 큰 가중치를 부여하도록 학습됨
        #     # CNN과 MLP중 더 중요한 특징을 학습을 통해 자동으로 비율 조정
        #     nn.ReLU()
        # )

        

    # 카메라(CNN) & 라이다(MLP) 데이터를 전처리 후 멀티모달 Feature로 변환
    def forward(self, observations):
        # 1. 이미지와 라이다 feature 추출
        # - 이미지: [B, 3, 480, 640] → CNN 인코더 → [B, 300]
        # - 라이다: [B, 1024, 3] → PointNet → [B, 1024]
        # - CNN 인코더는 RGB 3채널만 사용하므로 [:, :3, :, :] 슬라이싱 (4채널로 들어옴옴)
        img_features = self.cnn_encoder(observations["camera_image"][:, :3, :, :])  # env에서 가져온 카메라 이미지 입력
        lidar_global_features = self.lidar_encoder(observations["lidar_points"])  # env에서 가져온 LiDAR 데이터 입력
        yolo_tensor = observations["yolo_info"]
        lidar_local_features = observations["lidar_local_features"].squeeze(1)

        # 3. 모든 feature 결합
        # - 이미지 [B, 300] + 라이다 [B, 1024] + YOLO [B, 3] → [B, 1327]
        combined_features = th.cat([yolo_tensor, img_features, lidar_global_features, lidar_local_features], dim=1)  
        return combined_features

        # # 4. 최종 Feature 추출 (Linear + ReLU) → [B, 512]
        # return self.final_layer(combined_features)  