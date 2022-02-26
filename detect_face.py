import time
import numpy as np
import sys
import cv2

from main_utils.box_utils import extend_bbox
# sys.path.append("models_local/face_detection/DSFDPytorchInference")
# from models_local.face_detection.face_test import detect_face_bbox_head_batch


def detect_face_bbox_head(cam, head_bbox_queue, face_embedding_queue, detect_face_bbox_head_batch):
    max_size_head = 200
    while cam.cap.isOpened():

        track_bbs_ids, frame_rgb, frame_count = head_bbox_queue.get()

        batch_image_head = None
        x_offset = []
        y_offset = []
        w_scale = []
        h_scale = []
        for i, box in enumerate(track_bbs_ids):
            box = list(map(int, box[:4]))
            box = extend_bbox(box, frame_rgb.shape)
            image_head = frame_rgb[box[1]:box[3], box[0]:box[2]]

            w_scale.append(image_head.shape[1] / max_size_head)
            h_scale.append(image_head.shape[0] / max_size_head)
            x_offset.append(box[0])
            y_offset.append(box[1])

            if 0 in image_head.shape:
                image_head = np.zeros((max_size_head, max_size_head, 3), dtype=int)
            else:
                image_head = cv2.resize(image_head, (max_size_head, max_size_head), interpolation=cv2.INTER_AREA)

            image_head = image_head[:, :, ::-1]
            if batch_image_head is None:
                batch_image_head = image_head[None, :, :, :]
            else:
                batch_image_head = np.vstack((batch_image_head, image_head[None, :, :, :]))

        start_time = time.time()
        if batch_image_head is not None:
            boxes_face, landmarks_face = detect_face_bbox_head_batch(batch_image_head, x_offset, y_offset, w_scale, h_scale)
        else:
            # Case no track_bbs_ids (no head, no track)
            boxes_face = []
            landmarks_face = []

        # print("boxes_face: ", boxes_face)
        # print("detect_face_bbox_head cost: ", time.time() - start_time)
        assert len(boxes_face) == len(track_bbs_ids) == len(landmarks_face)

        face_embedding_queue.put([boxes_face, landmarks_face, frame_rgb, track_bbs_ids, frame_count])

    cam.cap.release()


if __name__ == '__main__':
    pass