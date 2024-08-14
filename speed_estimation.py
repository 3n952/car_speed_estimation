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

# A polygon represented by a numpy array of shape (N, 2), containing the x, y coordinates of the points. -> SOURCE
# SOURCE labeling -> https://roboflow.github.io/polygonzone/

# 주의사항: 여러개 polygon 중 겹치는 부분에 대해서만 tracking 한다.
# lt, rt, rb, lb
SOURCE_vehicle1= np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
SOURCE_traffic1 = np.array([[452, 337],[800, 337],[1270, 690],[-100, 690]])
SOURCE_traffic2= np.array([[230, 160],[400, 160], [680, 340], [-50, 340]])

# 실제 거리 영역 좌표(m)
TARGET = np.array([[0,0], [24, 0], [24, 249], [0, 249]])

# arguments 설정
def parse_arguments():
    parser = argparse.ArgumentParser(
        description= 'vehicle speed estimation'
    )

    parser.add_argument(
        "--source_video_path", required= True,
        help = "path to the source video file",
        type = str,

    )
    
    return parser.parse_args()

# yolov5모델 로드를 위한 함수
def load_yolov5_model(weights_path):
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    model.eval()  # Set model to evaluation mode
    return model

def main():
    args = parse_arguments()

    #video에 대한 정보 - 높이 너비 fps 등등..
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # yolov5 model load
    model = load_yolov5_model(r'..\car_model_recogize\Yolov5\yolov5_run\yolov5s.pt')

    #tracking using supervision
    byte_track = sv.ByteTrack(frame_rate=video_info.fps,
                              track_thresh=0.55,  # 추적 임계값
                              match_thresh=0.8,  # 매칭 임계값
                            )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh = video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh = video_info.resolution_wh
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale)
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    # polygon zone init
    polygon_zone = sv.PolygonZone(SOURCE_traffic2, frame_resolution_wh=video_info.resolution_wh )

    # 또 다른 polygon zone 만들기
    # polygon_zone2 = sv.PolygonZone(SOURCE2, frame_resolution_wh=video_info.resolution_wh )

    # perspective transform 행렬
    transformer_m = perspective_transformation.Perspective_transformer(source = SOURCE_traffic1, target = TARGET)
    print(transformer_m)

    # speed 계산을 위한 죄표 초기화
    x_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))
    y_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))


    for frame in frame_generator:

        # model infer (detect.py)
        result = model(frame)

        # yolov5 detect 반환 값에 대한 sv.Detections.from_yolov5 모듈 적용
        detections = sv.Detections.from_yolov5(result)

        # polygon zone 
        detections = detections[polygon_zone.trigger(detections)]
        #detections = detections[polygon_zone2.trigger(detections)]

        # 객체 탐지 반환 값을 가지고 tracking
        detections = byte_track.update_with_detections(detections= detections)

        # 탐지된 객체의 bottom center 좌표
        car_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) 
        # perspective transformation 적용 
        car_points = transformer_m.transform_points(points=car_points).astype(int)

        labels = []

        for tracker_id, [x, y] in zip(detections.tracker_id, car_points):
            # 직선도로의 경우 y 좌표값만 사용.
            y_coordinates[tracker_id].append(y)
            if len(y_coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # deque 자료구조에서 반환
                y_coordinates_start = y_coordinates[tracker_id][-1]
                y_coordinates_end = y_coordinates[tracker_id][0]
                distance = abs(y_coordinates_start - y_coordinates_end)
                time = len(y_coordinates[tracker_id] / video_info.fps)
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")





        # labels = [
        #     f"#{tracker_id}" for tracker_id in detections.tracker_id
        # ]

        annotated_frame = frame.copy()

        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE_traffic2, color=sv.Color.RED)
        #annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE2, color=sv.Color.BLUE)

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels = labels
        )

        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

    