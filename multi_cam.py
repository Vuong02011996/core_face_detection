from kthread import KThread
import threading
import cv2
import time
import multiprocessing

from main import main as main1
from main2 import main as main2

input_path = "/storages/data/DATA/Video_Test/5555_17_3_action1.mp4"


def mult_cam_1():
    thread_1 = KThread(target=main1, args=(input_path, True, "window1"))
    thread_2 = KThread(target=main1, args=(input_path, True, "window2"))
    thread_3 = KThread(target=main1, args=(input_path, True, "window3"))
    thread_4 = KThread(target=main1, args=(input_path, True, "window4"))

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


def mult_cam_2():
    my_thread = threading.Thread(target=main1, args=(input_path, True, "window1"))
    my_thread.start()
    # time.sleep(1)
    my_second_thread = threading.Thread(target=main2, args=(input_path, True, "window2"))
    my_second_thread.start()
    # my_second_thread.join()  # Wait until thread finishes to exit


def mult_cam_3():
        jobs = []
        for i in range(4):
            process = multiprocessing.Process(target=main1, args=(input_path, True, "window1"))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()


if __name__ == '__main__':
    mult_cam_3()