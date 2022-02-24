import time
import cv2
import numpy as np
from mot_tracking import untils_track
from head_detect import class_names
from main_utils.draw import draw_boxes_tracking, draw_det_when_track, show_stream, draw_region


def drawing(cam, show_queue, show_all_queue, frame_final_queue):
    while cam.cap.isOpened():
        track_bbs_ids,  frame_count, frame_rgb, boxes_face = show_queue.get()
        track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count = show_all_queue.get()
        image_show = frame_rgb.copy()
        image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)

        # delete track anh list name have bbox out of region
        track_bbs_ids, list_idx_bbox_del = untils_track.select_bbox_inside_polygon(track_bbs_ids, cam.region_track)

        if image_show is not None:
            image_show = draw_region(image_show, cam.region_track)
            image_show = draw_boxes_tracking(image_show, track_bbs_ids, boxes_face, list_name=None,
                                             track_bbs_ext=unm_trk_ext)
            image_show = draw_det_when_track(image_show, boxes, scores=scores, labels=labels,
                                             class_names=class_names)
        if frame_final_queue.full() is False:
            frame_final_queue.put([image_show, frame_count])
        else:
            time.sleep(0.0001)
        # print("drawing cost", time.time() - start_time)
        # print("##################################")

    cam.cap.release()


if __name__ == '__main__':
    pass
