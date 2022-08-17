# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os
from torch2trt import torch2trt
import matplotlib.pyplot as plt

def load_model(model_root, backbone):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    assert (os.path.exists(model_root))
    print('Backbone Model Root:', model_root)
    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)
    return backbone

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img,
                    backbone,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    tta=True):

    # load image
    # img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a = int((128 - 112) / 2)  # x start
    b = int((128 - 112) / 2 + 112)  # x end
    c = int((128 - 112) / 2)  # y start
    d = int((128 - 112) / 2 + 112)  # y end
    ccropped = resized[a:b, c:d]  # center crop the image
    ccropped = ccropped[..., ::-1]  # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype=np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)

    # extract features
    backbone.eval()  # set to evaluation mode
    ir50_trt_cropped = torch2trt(backbone, [torch.ones((1,3,112,112)).cuda()], use_onnx=True)
    with torch.no_grad():
        if tta:
            emb_batch = ir50_trt_cropped(ccropped.to(device)) + ir50_trt_cropped(flipped.to(device))
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(ir50_trt_cropped(ccropped.to(device)))


#     np.save("features.npy", features)
#     features = np.load("features.npy")

    return features
