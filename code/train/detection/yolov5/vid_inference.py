import numpy as np
import pandas as pd
import time
import torch
import json
import time
import os
import sys
import os.path as pt
from ensemble_boxes import weighted_boxes_fusion
import warnings
from yolo_inference import init_yolo_detector, simple_infer
warnings.filterwarnings('ignore')
import argparse
from tqdm import tqdm

IMAGE_ROOT = '../../../../user_data/tmp_data/round-2/video_'
CKPT_PATH = './runs/train/fbody3/weights/best.pt'
OUT_ROOT = '../../../../user_data/tmp_data/tracking/MOT-PANDA/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='for train or test')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu device to use')
    opt = parser.parse_args()
    IMAGE_ROOT+=opt.mode
    img_paths = {0:{},1:{},2:{},3:{}}
    vid_nms = [x for x in os.listdir(IMAGE_ROOT) if not x.endswith('zip')]
    for vid_nm in vid_nms:
        gpu_id = int(vid_nm.split('_')[0]) % 4
        img_root = os.path.join(IMAGE_ROOT, vid_nm)
        img_nms = os.listdir(img_root)
        img_paths[gpu_id][vid_nm] = sorted([os.path.join(img_root, img_nm) for img_nm in img_nms])

    device = 'cuda:{:}'.format(opt.gpu_id)
    print("work on device : ", device)
    model = init_yolo_detector(CKPT_PATH, ('full_body',), device=device)

    ts = time.time()
    cnt = 0
    for vid_nm, paths in img_paths[opt.gpu_id].items():
        print("process video : ", vid_nm)
        det_lst = []
        for path in tqdm(paths, total=len(paths)):
            frame = int(path.split("/")[-1].split("_")[-1].split(".")[0])
            yolov5_dets = simple_infer(path, model, device)
            for det in yolov5_dets:
                x1, y1, x2, y2, score = det
                det_lst.append([frame, -1, x1, y1, x2-x1, y2-y1, score])

            te = time.time()
            cnt += 1
            time_cost = te - ts
            print("average time : ", time_cost / cnt)
        
        det_out_root = os.path.join(OUT_ROOT, vid_nm, 'det')
        if not os.path.exists(det_out_root):
            print("create folder : {:}".format(det_out_root))
            os.makedirs(det_out_root)
        
        det_out_path = det_out_root + '/yolov5_det.txt'
        
        det_df = pd.DataFrame(det_lst)
        det_df.to_csv(det_out_path, index=False,header=False)
    filename=opt.mode+'_{}.txt'.format(opt.gpu_id)
    print(filename)
    with open(filename,'w') as f:
        f.write(filename)
