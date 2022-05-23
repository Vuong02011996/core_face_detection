# FACE Detection
+ Run file test: check_face_dectect.py

# Perfomance
## Hardware
+ CPU: AMD® Ryzen threadripper 2950x 16-core processor × 32 
+ GPU: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A] (RTX 2080 Ti GAMING X TRIO)
+ RAM: 
  + Type: DDR4
  + Speed: 2133 MT/s

## Model
### Model DSFD
+ Nhận diện đúng 8 face.
+ 10 image : 0.037
+ 30 image: 0.08
### Model SRCFD
1. Config 1
      ```commandline
      n_workers=1
      max_size=224,224
      force_fp16=False
      det_model=scrfd_500m_bnkps
      det_batch_size=1
      rec_model=glintr100
      rec_batch_size=1
      mask_detector=None
      ga_model=None
      triton_uri='localhost:8001'
      return_face_data=False
      extract_embeddings=False
      detect_ga=False
      det_thresh=0.6
      ```

+ Model: scrfd_500m_bnkps, scrfd_500m_gnkps(the same time inference)
  + det_batch_size=1
    + Nhận diện đúng 8 face. 
    + 10 image: 0.037s
    + Model: det_batch_size=1 cũng giống det_batch_size=10(không chạy batch)
  + det_batch_size=10
    + 30 image: 0.06s
    + 10 image: 0.03
+ Model: scrfd_10g_bnkps, scrfd_10g_gnkps
  + Thời gian chậm hơn một xíu với 500m
  + Độ chính xác thấp, nhận sai rất nhiều.(30 image head -> 59face.)
+ Model: yolov5s-face
  + [05/17/2022-04:38:32] [TRT] [W] Output type must be INT32 for shape outputs


# Note 
+ [Commercial usage](https://github.com/deepinsight/insightface/tree/master/python-package)
+ DSFD model save in : /home/gg_greenlab/.cache/torch/hub/checkpoints/WIDERFace_DSFD_RES152.pth
+ retinaface_mnet025_v1, retinaface_mnet025_v2, retinaface_r50_v1, centerface no support batch inference.
+ (scrfd_500m_gnkps): _KPS means the model includes 5 keypoints prediction.
+ bn(batch normalize) vs gn(group normalize): custom models retrained for this repo. 
Original SCRFD models have bug (deepinsight/insightface#1518) with detecting large faces occupying >40% of image. 
These models are retrained with Group Normalization instead of Batch Normalization, which fixes bug,
though at cost of some accuracy.


# REF
+ [scrfd_2021](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
+ [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST#detection)
+ [DSFD-Pytorch-Inference-batch size](https://github.com/hukkelas/DSFD-Pytorch-Inference/tree/2bdd997d785e20ea39a911e9b3c451b7cdd3b152)