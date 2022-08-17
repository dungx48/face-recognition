import sys
# Extract embeddings of dataset and save to a npy file
import numpy as np
from evolveface.util import extract_feature_v1
from evolveface.backbone import model_irse

# sys.settrace()

input_size = [112, 112]
backbone = model_irse.IR_50(input_size)
data_root = '/home/vdungx/Desktop/face-recognition/dataset/processed'
pretrain_model = '/home/vdungx/Desktop/face-recognition/pretrain/backbone_ir50_asia.pth'

features = extract_feature_v1.extract_feature(data_root=data_root, model_root=pretrain_model, backbone=backbone)
np.save("/home/vdungx/Desktop/face-recognition/data_aia.npy", features)

# data = np.load('/home/vdungx/Desktop/face-recognition/data.npy')
# print(data)