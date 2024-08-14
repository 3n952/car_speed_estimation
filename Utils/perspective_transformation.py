import numpy as np
import cv2 

# polygon의 관심영역 좌표를 실제 거리 좌표에 매핑 (입력: polygon 좌표 numpy array, 출력: 실제 거리 좌표 numpy array)(np.float32)

class Perspective_transformer:
    def __init__(self, source: np.ndarray, target:np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # 원근 변환 행렬 계산
        self.m = cv2.getPerspectiveTransform(source, target)
    
    # tracking 되는 객체에 대한 bottom center 좌표를 변환
    def transform_points(self, points: np.array):
        # perspectiveTransform 입력 인자로 3채널 array가 필요
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1,2)
    
