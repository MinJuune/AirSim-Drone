import os
import cv2
import datetime

class ImageSaver:
    def __init__(self, save_dir="dataset/all_frames"):
        self.save_dir = save_dir
        # os.makedirs(self.save_dir, exist_ok=True)
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"폴더 생성됨 또는 이미 존재: {os.path.abspath(self.save_dir)}")
        except Exception as e:
            print(f"폴더 생성 실패: {e}")

        self.counter = 0

    def save(self, image):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"frame_{self.counter}_{timestamp}.jpg"
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, image)
        self.counter += 1
