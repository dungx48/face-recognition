# Face Recognize for roll call
![](https://img.shields.io/static/v1?label=python&message=3.7.9&color=pink)

***

## Pre-Requisites
* Linux or macOS
* Python 3.7.9
* [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-1-download-archive)
* [Pytorch 1.11.0+cu113] 
    * ```$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113```
* [TensorRT 8.2.5.1](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.5.1/local_repos/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505_1-1_amd64.deb)
    * ```$ sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505_1-1_amd64.deb```
    * ```$ sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505/82307095.pub```
    * ```$ sudo apt-get update```
    * ```$ sudo apt install tensorrt```
* Some lib:
    * ``` $ pip install -r requirements.txt```
* [Torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
    * ```$ git clone https://github.com/NVIDIA-AI-IOT/torch2trt```
    * ```$ cd torch2trt ```
    * ```$ python3 setup.py install```


***

## Model Pretrain
* [BH-IR50](https://drive.google.com/drive/folders/11TI4Gs_lO-fbts7cgWNqvVfm9nps2msE)

***

## Data preprocessing
```$ python3 ./align/align_dataset_mtcnn.py```

***

## Extract feature face
```$ python3 ./run_extract_feature.py ```

***

## Run

```$ python3 ./run_face_cam.py```