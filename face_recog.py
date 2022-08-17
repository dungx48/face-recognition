import imp
import sys
import os
import math
import cv2
import time
sys.path.append("evolveface")
import os
import json
import random

# sys.path.append("F:\\vinAI\\face-recog\\face.evoLVe.PyTorch\\applications\\align")
from PIL import Image
from evolveface.align.detector import detect_faces
from evolveface.align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
from evolveface.util import extract_feature_v2, extract_feature_v1, extract_feature_v2_test
from evolveface.backbone import model_irse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


#--------------------------------------------------
# Function to detect and align face img 
def detect_align(img, crop_size):
  warped_list=[]
  scale = crop_size / 112.
  reference = get_reference_facial_points(default_square = True) * scale
  bbs, landmarks = detect_faces(img)
  # print(landmarks)
  for bb, landmark in zip(bbs, landmarks):
    # print(landmark)
    facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
    # print(facial5points, len(facial5points), len(reference))
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    
    warped_list.append([bb,warped_face])
  return warped_list
#--------------------------------------------------

#--------------------------------------------------
# Function defines 2 distance caculation methods: Euclide or Cosine
def distance(embeddings1, embeddings2, distance_metric=0):
        # print(embeddings1.shape, embeddings2.shape)
        if distance_metric == 0:
            # Euclidian distance
            # print(embeddings1)
            embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=0, keepdims=True)
            # print(embeddings2)
            embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=0, keepdims=True)
            dist = np.sqrt(np.sum(np.square(np.subtract(embeddings1, embeddings2))))
            # diff = np.subtract(embeddings1, embeddings2)
            # dist = np.sum(np.square(diff), 1)
            return dist
        elif distance_metric == 1:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=0)
            norm = np.linalg.norm(embeddings1, axis=0) * np.linalg.norm(embeddings2, axis=0)
            similarity = dot/norm
            # print("similarity", similarity)
            dist = np.arccos(similarity) / math.pi
            return dist
        else:
            raise 'Undefined distance metric %d' % distance_metric
#--------------------------------------------------

#--------------------------------------------------
# Extract feature of a single image
def extract_feature_test(img, backbone):
  # pre_time = time.time()
  aligned_imgs = detect_align(img, crop_size=112)
  # print(time.time() - pre_time)
  bb_and_features = []
  for aligned in aligned_imgs:
    feature = extract_feature_v2_test.extract_feature(img=aligned[1], backbone=backbone)
    bb_and_features.append([aligned[0], feature])
  return bb_and_features
#--------------------------------------------------

#--------------------------------------------------
# Extract feature of a single image
def extract_feature(img, backbone):
  # pre_time = time.time()
  aligned_imgs = detect_align(img, crop_size=112)
  # print(time.time() - pre_time)
  bb_and_features = []
  for aligned in aligned_imgs:
    feature = extract_feature_v2.extract_feature(img=aligned[1], backbone=backbone)
    bb_and_features.append([aligned[0], feature])
  return bb_and_features
#--------------------------------------------------

#--------------------------------------------------
# Extract feature of a single image
def extract_feature_fast(aligned_imgs, backbone):
  bb_and_features = []
  for aligned in aligned_imgs:
    feature = extract_feature_v2.extract_feature(img=aligned[1], backbone=backbone)
    bb_and_features.append([aligned[0], feature])
  return bb_and_features
#--------------------------------------------------

#--------------------------------------------------
#Calculate avg embedding for each person: Require folder of imgs of 
# that person (embedding == feature)
def avg_embeddings(img_folder_path, backbone):
  img_cnt = 0
  avg_emb = np.zeros((1, 512))
  for img_path in os.listdir(img_folder_path):
    img_cnt+=1
    img = Image.open(img_folder_path+"/"+img_path)
    bb_and_features = extract_feature(img, backbone)
    avg_emb += bb_and_features[0][1].cpu().detach().numpy()
  avg_emb = avg_emb / img_cnt
  if "/" in img_folder_path:
    class_label = img_folder_path.split("/")[-1]
  elif "\\" in img_folder_path:
    class_label = img_folder_path.split("\\")[-1]
  # print(avg_emb.shape)
  avg_emb = avg_emb.flatten()
  return class_label, avg_emb
#--------------------------------------------------

#--------------------------------------------------
#Make a dict reference to class label
def make_cls_label_dict(facebank_path, data_root, input_size):
  facebank_feature = np.load(facebank_path)
  rgb_mean = rgb_std = [0.5, 0.5, 0.5]
  transform = transforms.Compose([
      transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
      transforms.CenterCrop([input_size[0], input_size[1]]),
      transforms.ToTensor(),
      transforms.Normalize(mean = rgb_mean, std = rgb_std)])
  dataset = datasets.ImageFolder(data_root, transform)
  loader = torch.utils.data.DataLoader(dataset, batch_size = 512, shuffle = False, pin_memory = True, num_workers = 0)
  total_len=0
  cls_label_dict = {}
  for each_class in loader.dataset.classes:
    each_len=len([fn for fn in os.listdir(data_root+"/"+each_class)])
    total_len = total_len+each_len
    if total_len-each_len==0:
      cls_label_dict[each_class] = str(total_len-each_len) + "-" +str(total_len)
    else:
      cls_label_dict[each_class]= str(total_len-each_len+1)+"-"+str(total_len)
  # print(cls_label_dict)
  avg_cls_label_dict = {}
  i=0
  for key, value in cls_label_dict.items():
    img_range = value.split("-")
    avg_emb = facebank_feature[int(img_range[0])]
    # print(img_range)
    for i in range(int(img_range[0]), int(img_range[1])):
        avg_emb += facebank_feature[i]
    avg_cls_label_dict[key] = avg_emb / ((int(img_range[1])-int(img_range[0])))
    i+=1
  return cls_label_dict, avg_cls_label_dict
#--------------------------------------------------

#--------------------------------------------------
def recog(img, bb_and_features, cls_label_dict):
  for bb_and_feature in bb_and_features:
    bb = bb_and_feature[0]
    feature = bb_and_feature[1]
    npFeature = feature.cpu().detach().numpy()
    max_dist = 0
    min_dist = 999
    dist_list = []
    for key, value in cls_label_dict.items():
        dist = distance(value, npFeature[0], distance_metric=0)
        dist_list.append(dist)
        if max_dist < dist:
          max_dist = dist
        if min_dist>dist:
          min_dist = dist
          min_dist_name = key
    # print(min_dist)
    if min_dist < 1:
      name = min_dist_name
      bb_and_feature.extend([min_dist,name])
    else:
      name = "Unknown"
    img = np.array(img)
    cv2.rectangle(img, (int(bb [0]),int(bb[1])) , (int(bb[2]), int(bb[3])), (0,0,0), 2)
    cv2.putText(img, name, (int(bb[0]),int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
  return img
#--------------------------------------------------

#--------------------------------------------------
#Recog for each bounding box
def recog_each_bbox(feature, cls_label_dict, distance_metric=0):
  # npFeature = feature.cpu().detach().numpy()
  max_dist = 0
  min_dist = 999
  dist_list = []
  cnt = 0
  for key, value in cls_label_dict.items():
      dist = distance(value, feature[0], distance_metric=distance_metric)
      # print(dist)
      # print("value", value)
      # print("feature", feature[0])
      dist_list.append(dist)
      if max_dist < dist:
        max_dist = dist
      if min_dist>dist:
        min_dist = dist
        min_dist_name = key
        idx = cnt
      cnt+=1
  if distance_metric == 0:
    dist_threshold = 1
  elif distance_metric == 1:
    dist_threshold = 0.4
  if min_dist < dist_threshold:
    name = min_dist_name
  else:
    name = "Unknown"
    idx = len(cls_label_dict)+1
  # img = np.array(img)
  # cv2.rectangle(img, (int(bb [0]),int(bb[1])) , (int(bb[2]), int(bb[3])), (0,0,0), 2)
  # cv2.putText(img, name, (int(bb[0]),int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
  return min_dist, name, idx
#--------------------------------------------------