from models.yolo import targetDetector
import cv2
import os
import matplotlib.pyplot as plt

# 이미지 경로 (image.jpg가 models 폴더에 있으므로 상대경로 지정)
IMAGE_PATH = os.path.join("models", "image.jpg")

# 이미지 로드
image = cv2.imread(IMAGE_PATH)

# YOLO 클래스 초기화
yolo = targetDetector(model_path="weights/best.pt")

print(image.shape)
# 추론 실행
result = yolo.detect(image)

# 결과 출력
print("YOLO Detection Result:")
print(f"Detected: {result['detected']}")
print(f"Area Ratio: {result['area_ratio']:.4f}")
print(f"Offset Score: {result['offset_score']:.4f}")
# offset 테스트. 끝나면 지워워
print(f"Raw Offset Distance (pixels): {result['offset']:.2f}")  

# 바운딩 박스 시각화 (탐지되었을 때만)
if result["detected"]:
    # best_idx를 내부에서 저장하지 않아서 여기서 다시 구함
    result_raw = yolo.model(image)  # raw result 다시 받아옴
    boxes = result_raw[0].boxes

    cls_list = boxes.cls.cpu().numpy()
    conf_list = boxes.conf.cpu().numpy()
    xyxy_list = boxes.xyxy.cpu().numpy()

    # 타겟 클래스만 필터링
    target_indices = (cls_list == yolo.target_class_id)
    target_confs = conf_list[target_indices]
    target_boxes = xyxy_list[target_indices]

    # 가장 확률 높은 타겟 박스 선택
    best_idx = target_confs.argmax()
    x1, y1, x2, y2 = target_boxes[best_idx]

    # 바운딩 박스 그리기
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    image = cv2.putText(image, "albania", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 이미지 출력 (matplotlib)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("YOLO Detection Result")
plt.axis("off")
plt.show()
