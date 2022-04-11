from kthread import KThread
import threading
import cv2
import time
import multiprocessing
import sys

from main import main as main1
# sys.path.append("models_local/head_detection/yolov5_detect")
# from models_local.head_detection.yolov5_detect.yolov5_detect_image import Y5Detect

sys.path.append("models_local/face_detection/DSFDPytorchInference")
from models_local.face_detection.face_test import detect_face_bbox_head_batch

# input_path = '/home/vuong/Videos/test_phu.mp4'
input_path = "/storages/data/DATA/Video_Test/test_phu.mp4"
# y5_model = Y5Detect(weights="models_local/head_detection/yolov5_detect/model_head/y5headbody_v2.pt")
# class_names = y5_model.class_names
# print(class_names)
# a =0


def mult_cam_3():
        jobs = []
        multiprocessing.set_start_method('spawn')
        for i in range(1):
            process = multiprocessing.Process(target=main1, args=(input_path, detect_face_bbox_head_batch,  True, "window_" + str(i)))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()


if __name__ == '__main__':
    mult_cam_3()