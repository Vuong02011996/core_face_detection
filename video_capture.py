import cv2
import time


def video_capture(cam, frame_detect_queue, save_video=False):
    frame_count = cam.frame_start
    # frame_step = 2
    # frame_using = 0
    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    if save_video:
        # https://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
        # https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html#gsc.tab=0
        # https://stackoverflow.com/questions/30103077/what-is-the-codec-for-mp4-videos-in-python-opencv
        # VideoWriter (const String &filename, int fourcc, double fps, Size frameSize, bool isColor=true)
        size = (cam.width, cam.height)
        result = cv2.VideoWriter(cam.path_image + cam.process_name + ".mp4",
                                 cv2.VideoWriter_fourcc(*'MP4V'),  # X264, MJPG, MPEG
                                 cam.fps, size)
    else:
        result = None

    while cam.cap.isOpened():
        start_time = time.time()
        ret, frame_ori = cam.cap.read()
        # if frame_using != 0 and frame_count % frame_using != 0:
        #     frame_count += 1
        #     continue
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)

        # print("##################################")
        # print("video_capture cost", time.time() - start_time)
        frame_detect_queue.put([frame_rgb, frame_count])
        # print("frame_count: ", frame_count)
        if frame_count == 22:
            a = 0
        frame_count += 1
        # frame_using += frame_step

        # Write the frame into the
        # file 'filename.avi'
        if result is not None:
            result.write(frame_ori)

        if frame_count == 1500:
            break

    cam.cap.release()
    if result is not None:
        result.release()


if __name__ == '__main__':
    pass