# from numpy.linalg import det
import sys
from turtle import rt
import cv2
import time
import imutils
import scipy as sp
sys.path.append("applications\\align")
from PIL import Image
import numpy as np
from evolveface.util import extract_feature_v2
from evolveface.backbone import model_irse
from face_recog import extract_feature, make_cls_label_dict, recog_each_bbox
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import torch
from torch2trt import torch2trt, TRTModule
# from torch2trt_dynamic import torch2trt_dynamic
# from nlp.speak import speak_name, speak


def progress(frame):
    global name_person, min_dist
    bb_and_features = extract_feature(frame, backbone_trt)
    frame = np.array(frame)
    color = [0,0,0]
    if len(bb_and_features)!=0:
        for bb_and_feature in bb_and_features:
            bbox = np.array((bb_and_feature[0][0],bb_and_feature[0][1],bb_and_feature[0][2]-bb_and_feature[0][0],bb_and_feature[0][3]-bb_and_feature[0][1]))
            feature = bb_and_feature[1].cpu().detach().numpy()
            min_dist, name_person, _ = recog_each_bbox(feature, avg_cls_label_dict, distance_metric=0)
            # print(min_dist, name)
            bbox[2:] += bbox[:2]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(name_person))*15, int(bbox[1])), color, -1)
            cv2.putText(frame, name_person,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
    return name_person, min_dist, frame


if __name__ == "__main__":
    
    model_root = "./pretrain/backbone_ir50_asia.pth"
    data_root = "./dataset/processed"
    facebank_path = "./data.npy"
    input_size = [112, 112]
    backbone = model_irse.IR_50(input_size)
    backbone = extract_feature_v2.load_model(model_root, backbone)
    backbone.eval()
    data_sample = torch.ones((1,3,112,112)).cuda()
    # backbone_trt = torch2trt(backbone, [data_sample])
    # torch.save(backbone_trt.state_dict(), "ir50_trt.pth")
    backbone_trt = TRTModule()
    backbone_trt.load_state_dict(torch.load('./ir50_trt.pth'))
    cls_label_dict, avg_cls_label_dict = make_cls_label_dict(facebank_path, data_root, input_size)

    """
    'rtsp': if use camera rtsp
    '0' : if use webcam personal
    """
    rtsp = "rtsp://admin:TCPOJH@192.168.1.3:554/"
    # rtsp = "rtsp://admin:XEJVQU@10.37.239.113:554"
    cap = VideoStream(rtsp).start()
    prev_frame_time = 0
    # out = cv2.VideoWriter('test_face_rec.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (800,800))
    while(True):
        frame = cap.read()
        frame = imutils.resize(frame, width = 800)
        frame = Image.fromarray(frame)
        try:
            frame = progress(frame)[2]
            new_frame_time = time.time()
            fps = str(int(1/(new_frame_time-prev_frame_time)))
            prev_frame_time = new_frame_time
            cv2.putText(frame, fps, (7, 70), 0, 1, (100, 255, 0), 2)
            # out.write(frame)
            # frame = np.array(frame)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print("no face")
    cap.release()
    # out.relese()
    cv2.destroyAllWindows()