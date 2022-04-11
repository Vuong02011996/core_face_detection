import sys
import time

import numpy as np
import requests

# sys.path.append("models_local/head_detection/yolov5_detect")
# from models_local.head_detection.yolov5_detect.yolov5_detect_image import Y5Detect

#
# y5_model = Y5Detect(weights="models_local/head_detection/yolov5_detect/model_head/y5headbody_v2.pt")
# class_names = y5_model.class_names
from shm.writer import SharedMemoryFrameWriter


def head_detect(cam, frame_detect_queue, detections_queue):
    process_id = "604ef817ef7c20fc5e52a20d"
    shm_w1 = SharedMemoryFrameWriter(process_id)
    while cam.cap.isOpened():
        frame_rgb, frame_count = frame_detect_queue.get()
        start_time = time.time()

        # boxes, labels, scores, detections_sort = y5_model.predict_sort(frame_rgb, label_select=["head"])
        # print("head_detect cost: ", time.time() - start_time)

        shm_w1.add(frame_rgb)
        url = "http://0.0.0.0:7000/yolov5/predict/share_memory"
        payload = {}
        headers = {}
        response = requests.request("POST", url, headers=headers, data=payload)
        data_out = response.json()
        boxes = np.array(data_out["boxes"])
        labels = data_out["labels"]
        scores = data_out["scores"]
        detections_sort = np.array(data_out["detections_sort"])
        print("head_detect cost: ", time.time() - start_time)

        detections_queue.put([boxes, labels, scores, frame_rgb, detections_sort, frame_count])

    cam.cap.release()


if __name__ == '__main__':
    pass