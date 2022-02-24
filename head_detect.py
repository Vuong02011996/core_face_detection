import sys
import time

sys.path.append("models_local/head_detection/yolov5_detect")
# sys.path.append("../../core/models_local/head_detection/yolov5_detect")
from models_local.head_detection.yolov5_detect.yolov5_detect_image import Y5Detect


y5_model = Y5Detect(weights="models_local/head_detection/yolov5_detect/model_head/y5headbody_v2.pt")
# y5_model = Y5Detect(weights="../../core/models_local/head_detection/yolov5_detect/model_head/y5headbody_v2.pt")
class_names = y5_model.class_names


def head_detect(cam, frame_detect_queue, detections_queue):
    while cam.cap.isOpened():
        frame_rgb, frame_count = frame_detect_queue.get()
        start_time = time.time()
        boxes, labels, scores, detections_sort = y5_model.predict_sort(frame_rgb, label_select=["head"])
        # print("head_detect cost: ", time.time() - start_time)

        detections_queue.put([boxes, labels, scores, frame_rgb, detections_sort, frame_count])

    cam.cap.release()


if __name__ == '__main__':
    pass