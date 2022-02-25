import os
import subprocess
import cv2
from queue import Queue
import time
from kthread import KThread
import numpy as np
from video_capture import video_capture
from head_detect import head_detect
from tracking import tracking
from detect_face import detect_face_bbox_head
from recognize_face import get_face_features
from drawing import drawing


class InfoCam(object):
    def __init__(self, cam_name):
        self.cap = cv2.VideoCapture(cam_name)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frame_video = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_start = 0
        # coordinates = get_config_from_process_name(process_name, self.width, self.height)
        # self.region_track = [coordinates[0], coordinates[2], coordinates[3], coordinates[1]]
        self.region_track = np.array([[0, 0],
                                      [2560, 0],
                                      [2560, 1440],
                                      [0, 1440]])
        # test region
        # ret, frame_ori = self.cap.read()
        # frame_ori = cv2.rectangle(frame_ori, tuple(coordinates[0]), tuple(coordinates[2]), (0, 0, 255), 2)
        # frame_ori = cv2.circle(frame_ori, tuple(self.region_track[0]), radius=5, color=(255, 0, 255), thickness=10)
        # frame_ori = cv2.circle(frame_ori, tuple(self.region_track[1]), radius=5, color=(255, 255, 255), thickness=10)
        # frame_ori = cv2.circle(frame_ori, tuple(self.region_track[2]), radius=5, color=(255, 0, 0), thickness=10)
        # frame_ori = cv2.circle(frame_ori, tuple(self.region_track[3]), radius=5, color=(0, 0, 255), thickness=10)
        # frame_ori = draw_region(frame_ori, self.region_track)
        # cv2.imshow('output_roll_call', cv2.resize(frame_ori, (800, 500)))
        # cv2.waitKey(0)

        self.frame_step_after_track = 0
        self.show_all = True


def main(input_path, cv2_show=True, name_window=None):
    start_time = time.time()
    frame_detect_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    frame_final_queue = Queue(maxsize=1)
    face_embedding_queue = Queue(maxsize=1)
    head_bbox_queue = Queue(maxsize=1)
    show_all_queue = Queue(maxsize=1)
    show_queue = Queue(maxsize=1)

    cam = InfoCam(input_path)
    # -------------------------------------------------------------------------

    thread1 = KThread(target=video_capture, args=(cam, frame_detect_queue))
    thread2 = KThread(target=head_detect, args=(cam, frame_detect_queue, detections_queue))
    thread3 = KThread(target=tracking, args=(cam, detections_queue, show_all_queue, head_bbox_queue))
    thread4 = KThread(target=detect_face_bbox_head, args=(cam, head_bbox_queue, face_embedding_queue))
    thread5 = KThread(target=get_face_features, args=(cam, face_embedding_queue, show_queue))
    # thread6 = KThread(target=matching_identity, args=(cam, matching_queue, database_queue, show_queue))
    # thread7 = KThread(target=export_data, args=(cam, database_queue, object_dal))
    thread8 = KThread(target=drawing, args=(cam, show_queue, show_all_queue, frame_final_queue))

    thread_roll_call_manager = []
    thread1.daemon = True  # sẽ chặn chương trình chính thoát khi thread còn sống.
    thread1.start()
    thread_roll_call_manager.append(thread1)
    thread2.daemon = True
    thread2.start()
    thread_roll_call_manager.append(thread2)
    thread3.daemon = True
    thread3.start()
    thread_roll_call_manager.append(thread3)
    thread4.daemon = True
    thread4.start()
    thread_roll_call_manager.append(thread4)
    thread5.daemon = True
    thread5.start()
    thread_roll_call_manager.append(thread5)
    thread8.daemon = True
    thread8.start()
    thread_roll_call_manager.append(thread8)

    while cam.cap.isOpened():
        image, frame_count = frame_final_queue.get()
        print("frame_count: ", frame_count)
        image = cv2.resize(image, (500, 300))

        if cv2_show:
            if name_window is not None:
                cv2.imshow(name_window, image)
            else:
                cv2.imshow('output_roll_call', image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow('output')
                break

    total_time = time.time() - start_time
    print("FPS video: ", cam.fps_video)
    print("Total time: {}, Total frame: {}, FPS all process : {}".format(total_time, cam.total_frame_video,
                                                                         1 / (total_time / cam.total_frame_video)), )

    for t in thread_roll_call_manager:
        if t.is_alive():
            t.terminate()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_path = "/storages/data/DATA/Video_Test/5555_17_3_action1.mp4"
    # input_path = "/home/vuong/Videos/test_phu.mp4"

    main(input_path, cv2_show=True, name_window="window2")
    # thread_1 = KThread(target=main, args=(input_path, True, "window1"))
    # thread_2 = KThread(target=main, args=(input_path, True, "window2"))
    # thread_3 = KThread(target=main, args=(input_path, True, "window3"))
    # thread_4 = KThread(target=main, args=(input_path, True, "window4"))

    # thread_main = []
    # thread_1.daemon = True  # sẽ chặn chương trình chính thoát khi thread còn sống.
    # thread_1.start()
    # thread_main.append(thread_1)

    # thread_2.daemon = True
    # thread_2.start()
    # thread_main.append(thread_2)
    #
    # thread_3.daemon = True
    # thread_3.start()
    # thread_main.append(thread_3)
    #
    # thread_4.daemon = True
    # thread_4.start()
    # thread_main.append(thread_4)

    # for t in thread_main:
    #     if t.is_alive():
    #         t.terminate()
    # cv2.destroyAllWindows()
