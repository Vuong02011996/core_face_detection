from kthread import KThread
import threading
import cv2
import time
import multiprocessing
import sys

from main import main as main1
sys.path.append("models_local/head_detection/yolov5_detect")
from models_local.head_detection.yolov5_detect.yolov5_detect_image import Y5Detect

sys.path.append("models_local/face_detection/DSFDPytorchInference")
from models_local.face_detection.face_test import detect_face_bbox_head_batch

input_path = "/storages/data/DATA/Video_Test/5555_17_3_action1.mp4"
y5_model = Y5Detect(weights="models_local/head_detection/yolov5_detect/model_head/y5headbody_v2.pt")
# class_names = y5_model.class_names
# print(class_names)
# a =0


def mult_cam_1():
    thread_1 = KThread(target=main1, args=(input_path, y5_model, True, "window1"))
    thread_2 = KThread(target=main1, args=(input_path, y5_model, True, "window2"))
    thread_3 = KThread(target=main1, args=(input_path, y5_model, True, "window3"))
    thread_4 = KThread(target=main1, args=(input_path, y5_model, True, "window4"))

    thread_main = []
    thread_1.daemon = True  # sẽ chặn chương trình chính thoát khi thread còn sống.
    thread_1.start()
    thread_main.append(thread_1)

    thread_2.daemon = True
    thread_2.start()
    thread_main.append(thread_2)

    thread_3.daemon = True
    thread_3.start()
    thread_main.append(thread_3)

    thread_4.daemon = True
    thread_4.start()
    thread_main.append(thread_4)

    for t in thread_main:
        if t.is_alive():
            t.terminate()
    cv2.destroyAllWindows()


def mult_cam_3():
        jobs = []
        multiprocessing.set_start_method('spawn')
        for i in range(4):
            process = multiprocessing.Process(target=main1, args=(input_path, y5_model, detect_face_bbox_head_batch,  True, "window_" + str(i)))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()


if __name__ == '__main__':
    mult_cam_3()