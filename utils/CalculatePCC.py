import csv
import os
import numpy as np

class CalculatePCC:
    def __init__(self, log_path="weights/PCC_between_CO_A.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self.center_offsets = []
        self.theta_norms = []

        # # CSV 파일 초기화
        # with open(self.log_path, mode='w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["center_offset_ratio", "theta_norm", "pearson_corr"])

        if os.path.exists(self.log_path):
            # 기존 CSV 로드해서 리스트에 저장
            with open(self.log_path, mode='r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                for row in reader:
                    if len(row) >= 2:
                        try:
                            co = float(row[0])
                            tn = float(row[1])
                            self.center_offsets.append(co)
                            self.theta_norms.append(tn)
                        except ValueError:
                            continue  # 잘못된 데이터 무시
        else:
            # 새 파일이면 헤더 추가
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["center_offset_ratio", "theta_norm", "pearson_corr"])


    def log(self, center_offset, theta_norm):
        # 0,0 은 상관계수에서 제외
        if center_offset == 0 and theta_norm == 0:
            return  # 기록 안 함 (또는 따로 카운트만)

        self.center_offsets.append(center_offset)
        self.theta_norms.append(theta_norm)

        # 피어슨 상관계수 계산
        if len(self.center_offsets) >= 2:
            corr = np.corrcoef(self.center_offsets, self.theta_norms)[0, 1]
        else:
            corr = 0.0  # 최소 2개 필요

        # 값 저장
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([center_offset, theta_norm, corr])