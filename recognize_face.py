import time
import numpy as np
import cv2
import requests
from main_utils.helper import convert_np_array_to_base64, align_face


def get_face_features(cam, face_embedding_queue, show_queue):
    align = True
    while cam.cap.isOpened():
        boxes_face, landmarks_face, frame_rgb, track_bbs_ids, frame_count = face_embedding_queue.get()
        face_embeddings = np.zeros((len(boxes_face), 512))
        list_face_base64 = []
        start_time = time.time()
        index_have_face = []
        for idx, box_face in enumerate(boxes_face):
            box_face = list(map(int, box_face[:4]))
            if np.sum(box_face) == 0:
                continue
            else:
                image_face = frame_rgb[box_face[1]:box_face[3], box_face[0]:box_face[2]]
                if align:
                    image_face = align_face(frame_rgb, box_face, landmarks_face[idx])
                else:
                    image_face = cv2.resize(image_face, (112, 112), interpolation=cv2.INTER_AREA)
                face_base64 = convert_np_array_to_base64(image_face)
                list_face_base64.append(face_base64)
                index_have_face.append(idx)
        if len(list_face_base64) > 0:
            req = {"images": {"data": list_face_base64}, "embed_only": True}
            resp = requests.post('http://localhost:18081/extract', json=req)
            data = resp.json()
            for idx in range(len(data["data"])):
                face_embeddings[index_have_face[idx]] = data["data"][idx]["vec"]
                print("__________________________________________")
                print("vector", data["data"][idx]["vec"])
                print("__________________________________________")

        assert len(track_bbs_ids) == face_embeddings.shape[0] == len(boxes_face)
        # print("Reponse cost: ", time.time() - start_time)

        # matching_queue.put([face_embeddings, track_bbs_ids, frame_count, frame_rgb, boxes_face])
        # print(face_embeddings, track_bbs_ids, frame_count, frame_rgb, boxes_face)
        show_queue.put([track_bbs_ids, frame_count, frame_rgb, boxes_face])

    cam.cap.release()


if __name__ == '__main__':
    pass