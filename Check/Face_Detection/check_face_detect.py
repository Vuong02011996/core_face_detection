import glob
import os
import cv2
import time
from models_local.face_detection.DSFDPytorchInference import face_detection
import numpy as np
import base64
import requests
import shutil

detector = face_detection.build_detector(
    "RetinaNetResNet50",  # DSFDDetector, RetinaNetResNet50, RetinaNetMobileNetV1
    max_resolution=200
)


path_image_head = '/DATA/data/DATA/Clover_data/image_test/test1'
path_result_dsfd = '/DATA/data/DATA/Clover_data/image_test/dsfd/'
path_result_scrfd = '/DATA/data/DATA/Clover_data/image_test/scrfd/'

if os.path.exists(path_result_dsfd):
    shutil.rmtree(path_result_dsfd)
os.mkdir(path_result_dsfd)

if os.path.exists(path_result_scrfd):
    shutil.rmtree(path_result_scrfd)
os.mkdir(path_result_scrfd)


"""____________USING SCRFD in Insight-Face-REST_________"""


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def extract_vecs(ims, max_size=[200, 200]):
    target = [file2base64(im) for im in ims]
    # req = {"images": {"data": target}, "max_size": max_size, "embed_only": True}
    req = {"images": {"data": target}, "embed_only": True}
    start_time = time.time()
    resp = requests.post('http://localhost:18081/extract', json=req)
    data = resp.json()
    print("Reponse cost: ", time.time() - start_time)
    return data


def convert_np_array_to_base64(image):
    """

    :param image: np array image
    :return: string image base64
    """
    success, encoded_image = cv2.imencode('.png', image)
    image_face = encoded_image.tobytes()
    image_base64 = base64.b64encode(image_face).decode('ascii')
    return image_base64


def detect_face_one_image():
    pass


def detect_face_batch_scrfd():
    """
    @app.post('/extract', tags=['Detection & recognition'])
       - **images**: dict containing either links or data lists. (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **embed_only**: Treat input images as face crops (112x112 crops required), omit detection step. Default: False (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **return_landmarks**: Return face landmarks. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **verbose_timings**: Return all timings. Default: False (*optional*)
       - **msgpack**: Serialize output to msgpack format for transfer. Default: False (*optional*)
       - **api_ver**: Output data serialization format. (*optional*)
    """

    impaths = glob.glob(os.path.join(path_image_head, "*.png"))
    # impaths = impaths[:10]
    target = []
    list_image = []
    list_name = []
    for i, impath in enumerate(impaths):
        im = cv2.imread(impath)
        # im = cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA)
        im = im[:, :, ::-1]
        target.append(convert_np_array_to_base64(im))

        # using default
        # target.append(file2base64(impath))

        # for save
        list_image.append(im)
        list_name.append(impath.split("/")[-1].split(".")[0])

    req = {"images": {"data": target}, "threshold": 0.55, "return_landmarks": True, "embed_only": False,
           "extract_embedding": True}

    start_time = time.time()
    resp = requests.post('http://localhost:18081/extract', json=req)
    print("Reponse cost: ", time.time() - start_time)
    data = resp.json()["data"]

    for i in range(len(list_image)):
        boxes = data[i]["faces"]
        for j, box_face in enumerate(boxes):
            acc = str(int(box_face["prob"] * 100))
            box_face = box_face["bbox"]
            if any(n < 0 for n in box_face):
                continue
            box_face = list(map(int, box_face[:4]))
            image = list_image[i]
            image_head = image[box_face[1]:box_face[3], box_face[0]:box_face[2]]
            cv2.imwrite(path_result_scrfd + list_name[i] + "_box_" + str(j) + "_" + acc + "_scrfd.png", image_head)


"""_____________USING DSFD_________"""


def detect_face_batch_dsfd():
    impaths = glob.glob(os.path.join(path_image_head, "*.png"))
    # impaths = impaths[:10]
    # impaths = [impaths[0]]
    list_image = []
    list_name = []
    for i, impath in enumerate(impaths):
        im = cv2.imread(impath)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
        im = im[:, :, ::-1]
        if i == 0:
            batch_im = im[None, :, :, :]
        else:
            # batch_im = np.concatenate((batch_im, im), axis=0)
            batch_im = np.vstack((batch_im, im[None, :, :, :]))

        # for save
        list_image.append(im)
        list_name.append(impath.split("/")[-1].split(".")[0])

    start_time = time.time()
    result_detect = detector.batched_detect_with_landmarks(batch_im)
    print(f"Detection time dsfd: {time.time() - start_time:.3f}")

    boxes_det, landmarks = result_detect[0], result_detect[1]

    for i in range(len(list_image)):
        boxes = boxes_det[i]
        for j, box_face in enumerate(boxes):
            if any(n < 0 for n in box_face):
                continue
            acc = str(int(box_face[-1] * 100))
            box_face = list(map(int, box_face[:4]))
            image = list_image[i]
            image_head = image[box_face[1]:box_face[3], box_face[0]:box_face[2]]
            cv2.imwrite(path_result_dsfd + list_name[i] + "_box_" + str(j) + "_" + acc + "_dsfd.png", image_head)


if __name__ == '__main__':

    detect_face_batch_dsfd()
    detect_face_batch_scrfd()