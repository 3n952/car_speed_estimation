import argparse
import supervision as sv
import cv2
import torch
import numpy as np
from collections import defaultdict, deque
# from ultralytics import YOLO

import warnings
# FutureWarning 무시
warnings.simplefilter(action='ignore', category=FutureWarning)

from Utils import perspective_transformation

# polygon 영역(관심영역 좌표 지정)
SOURCE_test_video = np.array([[282, 150],[355, 154], [160, 390], [-55, 350]])

# 실제 거리 영역 좌표(m) lt, rt, rb, lb
TARGET_test = np.array([[0,0], [10, 0], [10, 55], [0, 55]])

# arguments 직접 설정
source_video_path = r'C:\Users\QBIC\Desktop\workspace\car_speed_estimation\data\test1.mp4'
weight_path = r'C:\Users\QBIC\Desktop\workspace\car_speed_estimation\weights\yolov5s.pt'
confidence_threshold = 0.35
iou_threshold = 0.5


# yolov5모델 로드를 위한 함수
def load_yolov5_model(weights_path):
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
    model.eval()  # Set model to evaluation mode
    return model
    
#video에 대한 정보 - 높이 너비 fps 등등..
video_info = sv.VideoInfo.from_video_path(source_video_path)

# yolov5 model load
model = load_yolov5_model(weight_path)

#tracking using supervision
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
)
# byte_track = sv.ByteTrack(frame_rate=video_info.fps,
#                           track_thresh=0.55,  # 추적 임계값
#                           match_thresh=0.8,  # 매칭 임계값
#                         )

thickness = sv.calculate_optimal_line_thickness(
    resolution_wh = video_info.resolution_wh
)
text_scale = sv.calculate_optimal_text_scale(
    resolution_wh = video_info.resolution_wh
)

bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
    
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER,
)

frame_generator = sv.get_video_frames_generator(source_video_path)

# polygon zone init
polygon_zone = sv.PolygonZone(SOURCE_test_video)

# 또 다른 polygon zone 만들기
# polygon_zone2 = sv.PolygonZone(SOURCE2, frame_resolution_wh=video_info.resolution_wh )

# perspective transform 행렬
transformer_m = perspective_transformation.Perspective_transformer(source = SOURCE_test_video, target = TARGET_test)


# speed 계산을 위한 죄표 초기화
# x_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))
y_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))

# 전체 프레임에 대한 각 프레임 별 객체의 속도 리스트
speed_list = []
frame_num = 0
for frame in frame_generator:
    speed_list2 = []

    # model infer (detect.py)
    result = model(frame)

    # yolov5 detect 반환 값에 대한 sv.Detections.from_yolov5 모듈 적용
    detections = sv.Detections.from_yolov5(result)

    # confidence 이상인 값만 추출
    detections = detections[detections.confidence > confidence_threshold]

    # polygon zone 
    detections = detections[polygon_zone.trigger(detections)]
    #detections = detections[polygon_zone2.trigger(detections)]

    detections = detections.with_nms(threshold=iou_threshold)

    # 객체 탐지 반환 값을 가지고 tracking
    detections = byte_track.update_with_detections(detections= detections)

    # 탐지된 객체의 bottom center 좌표
    car_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) 
    car_points = transformer_m.transform_points(points=car_points).astype(int)
            
    for tracker_id, [_, y] in zip(detections.tracker_id, car_points):
        y_coordinates[tracker_id].append(y)

    for tracker_id in detections.tracker_id:
        # 최소 0.5초 동안 탐지된 객체들만 tracking
        if len(y_coordinates[tracker_id]) < video_info.fps / 2:
            continue
        else:
            # deque 자료구조에서 반환
            y_coordinates_start = y_coordinates[tracker_id][-1]
            y_coordinates_end = y_coordinates[tracker_id][0]
            distance = abs(y_coordinates_start - y_coordinates_end)
            time = len(y_coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6
            speed_list2.append(int(speed))

    speed_list.append(speed_list2)
    frame_num += 1

# 원하는 구간 만큼의 전체 교통 std 

# 전체 프레임: frame_num
print(f'total frame numbers: {frame_num}')
# fps: video_info.fps
print(f'frame per second: {video_info.fps}')
input_second = int(input('std를 추출하고 싶은 시간 단위를 입력하시오(초): '))

input_frame_per_second = input_second * video_info.fps

# Assign group numbers to frames
for i in range(0,frame_num+1, input_frame_per_second):
    if i + input_frame_per_second > frame_num:
        print(f'--------- {i} ~ {frame_num} 번째 프레임 까지의 속도편차 -----------')
    else:
        print(f'--------- {i} ~ {i+input_frame_per_second -1} 번째 프레임 까지의 속도편차 -----------')
    sum_per_tracking = [sum(speed_tracking) for speed_tracking in speed_list[i:i+input_frame_per_second]]
    if len(sum_per_tracking) > 1:
        std_dev = np.std(sum_per_tracking, ddof=1)
        print("speed std:", std_dev)
    else:
        print("speed std: 0 (객체가 1개거나 없는 경우에 속합니다.)")
   