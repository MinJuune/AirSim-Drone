from ultralytics import YOLO
import numpy as np
import math

from utils.log import log
import config.config as config

class targetDetector:
    def __init__(self, model_path="C:/Users/rnrmf/Desktop/drone/code/AirSim-main/PythonClient/reinforcement_learning/AeroMind/weights/best_2.pt"):
        self.model = YOLO(model_path) # 모델 불러오기 
        # self.target_class_id = config.FLAG_CLASS_ID  # albania->3. 이거 좀있다가 바꾸기 
        self.target_class_id = 3 # 임시 

    def detect(self, image):
        """
        이미지를 받아 타겟 객체 중심 및 score 계산
        """
        result = self.model(image)    # 이미지 추론
        boxes = result[0].boxes       # 바운딩 박스 정보 

        # 0-1. 만약 바운딩 박스가 없을때? --> 이거는 생각을 해봐야 함
        if boxes is None or len(boxes) == 0:
            log("DEBUG", "No Target Detected.")
            return {
                "detected" : 0.0,
                "area_ratio" : 0.0,
                "center_offset_ratio" : 0.0,
                "theta_norm" : 0.0
            }
        
        # 클래스, 신뢰도, 좌표 추출
        cls_list = boxes.cls.cpu().numpy()
        conf_list = boxes.conf.cpu().numpy()
        xyxy_list = boxes.xyxy.cpu().numpy()

        # 원하는 클래스 ID(알바니아)의 박스 인덱스 추출
        # 만약 yolo가 여러개의 박스를 감지 했을때, 타겟 클래스가 있는 인덱스 고르기 
        # 만약 [0,3,2,3]이고 찾아야 하는것이 3임. 
        # 일단 where로 [1,3] 만들고
        # [0]으로 첫번째 거 위치 반환
        target_indices = np.where(cls_list == self.target_class_id)[0]

        # 0-2. 타겟 클래스가 없을때? --> 이것도 생각을 해봐야 함
        # if len(target_indices) == 0:
        #     log("DEBUG", "No Target Detected.")
        #     return {
        #         "detected" : 0.0, 
        #         "area_ratio" : 0.0,
        #         "center_offset_ratio" : 0.0,
        #         "theta_norm" : 0.0
        #     }

        # 0-3. 타겟 클래스가 여러개일때, 가장 신뢰도 높은 박스 선택
        # max_idx = target_indices[conf_list[target_indices].argmax()]
        # x1, y1, x2, y2 = xyxy_list[max_idx]

        x1, y1, x2, y2 = xyxy_list[0]

        # 바운딩 박스 중심 좌표 계산
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 이미지 면적 대비 바운딩 박스 면적 비율 계산
        width = x2 - x1
        height = y2 - y1
        area = width * height
        area_ratio = round(area / (config.CAMERA_HEIGHT * config.CAMERA_WIDTH), 2)

        # 이미지 중심과 박스 중심 간 거리(offset) 계산
        dx = center_x - (config.CAMERA_WIDTH / 2)
        dy = center_y - (config.CAMERA_HEIGHT / 2) 
        center_offset = np.sqrt(dx**2 + dy**2)

        half_cross_distance = np.sqrt((config.CAMERA_HEIGHT)**2 + (config.CAMERA_WIDTH)**2) / 2
        center_offset_ratio = round(1 - (center_offset / half_cross_distance), 2)

        x_shifted, y_shifted = center_x - (config.CAMERA_WIDTH / 2), center_y - (config.CAMERA_HEIGHT / 2) 

        theta = math.atan2(x_shifted, y_shifted)
        theta_norm = round((theta + math.pi) / (2 * math.pi), 2)

        log("DEBUG", "Target Detected.")
        log("DEBUG", f"Area Ratio: {area_ratio}")
        log("DEBUG", f"Center Offset: {center_offset_ratio}")
        log("DEBUG", f"Theta: {theta_norm}")

        # 결과 딕셔너리 반환
        return {
            "detected" : 1.0,            # 1. 타겟 감지 여부
            "area_ratio" : area_ratio,    # 2. 바운딩 박스 비율 (정규화)
            "center_offset_ratio" : center_offset_ratio, # 3. 중앙 오차 (정규화)
            "theta_norm" : theta_norm
        }