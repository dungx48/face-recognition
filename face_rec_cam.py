from numpy.linalg import det
import sys
import cv2
import time
import imutils
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
sys.path.append("applications\\align")
from PIL import Image
import numpy as np
from evolveface.util import extract_feature_v2
from evolveface.backbone import model_irse
from face_recog import extract_feature, make_cls_label_dict, recog_each_bbox
from imutils.video import VideoStream
import matplotlib.pyplot as plt


model_root = "data\\logs\\backbone_ir50_asia\\backbone_ir50_asia.pth"
data_root = "data\\images\\aligned"
facebank_path = "data\\features_extracted\\features.npy"
input_size =[112, 112]
backbone = model_irse.IR_50(input_size)
backbone = extract_feature_v2.load_model(model_root, backbone)
cls_label_dict = make_cls_label_dict(facebank_path, data_root, input_size)
# calculate cosine distance metric
max_cosine_distance = 0.4
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

# cap = VideoStream(src=0).start()
cap = cv2.VideoCapture("output.avi")
# prev_frame_time = 0
# new_frame_time = 0
fr_cnt = 0
dets=[]

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 800)
    # frame = cv2.flip(frame, 1)
    frame = Image.fromarray(frame)
    bb_and_features = extract_feature(frame, backbone)
    frame = np.array(frame)
    if fr_cnt==4:
        fr_cnt=0
    if fr_cnt==0:
        dets=[]
        for bb_and_feature in bb_and_features:
            bbox = np.array((bb_and_feature[0][0],bb_and_feature[0][1],bb_and_feature[0][2]-bb_and_feature[0][0],bb_and_feature[0][3]-bb_and_feature[0][1]))
            feature = bb_and_feature[1].cpu().detach().numpy()
            min_dist, name, _ = recog_each_bbox(feature, cls_label_dict)
            print(min_dist, name)
            dets.append(Detection(bbox, name, feature.flatten()))
    #     print("frame 0")
    # fr_cnt+=1
    # if fr_cnt==3:
    #     dets=[]
    #     for bb_and_feature in bb_and_features:
    #         bbox = np.array((bb_and_feature[0][0],bb_and_feature[0][1],bb_and_feature[0][2]-bb_and_feature[0][0],bb_and_feature[0][3]-bb_and_feature[0][1]))
    #         feature = bb_and_feature[1].cpu().detach().numpy()
    #         min_dist, name, _ = recog_each_bbox(feature, cls_label_dict)
    #         print(min_dist, name)
    #         dets.append(Detection(bbox, name, feature.flatten()))
    #     fr_cnt=0
    # fr_cnt+=1
    #initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # Call the tracker
    tracker.predict()
    tracker.update(dets)
    # new_frame_time = time.time()

    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        name = track.get_name()
        color = [0,0,0]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(name))*15, int(bbox[1])), color, -1)
        cv2.putText(frame, name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
    # fps = 1/(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
    # fps = str(int(fps))
    # cv2.putText(frame, fps, (7, 70), 0, 1, (100, 255, 0), 2)
    frame = np.array(frame)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


