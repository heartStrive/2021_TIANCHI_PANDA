import re
import cv2
import json
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import create_folder, reid_cls_annotation_process

root = "../../user_data/tmp_data/tracking/MOT-PANDA"
reid_root = '../../user_data/tmp_data/reid/ReID-PANDA'
fast_reid_data_path = '../../code/train/reid/fast-reid/datasets/panda_reid'
cls_root = '../../user_data/tmp_data/classification/CLS-PANDA'
vid_root = '../../user_data/tmp_data/round-2/video_train/'
vid_nms = os.listdir(root)
create_folder(fast_reid_data_path)
create_folder(reid_root)
create_folder(cls_root)

# create reid & classification annotations
for vid_nm in vid_nms:
    mot_root = os.path.join(root, vid_nm)
    det_path = os.path.join(mot_root, 'det/yolov5_det.txt')
    gt_path = os.path.join(mot_root, "gt/gt.txt")
    
    cls_vid_root = os.path.join(cls_root, vid_nm)
    create_folder(cls_vid_root)
    reid_vid_root = os.path.join(reid_root, vid_nm)
    create_folder(reid_vid_root)
    
    cls_vid_txt_root = os.path.join(cls_vid_root, 'txt')
    create_folder(cls_vid_txt_root)
    reid_vid_txt_root = os.path.join(reid_vid_root, 'txt')
    create_folder(reid_vid_txt_root)    
    
    det_df = pd.read_csv(det_path, names=['fid', 'pid', 'x', 'y', 'w', 'h','score'])
    det_df.insert(det_df.shape[1], 'did', det_df.index.values)
    gt_df = pd.read_csv(gt_path, names=['fid', 'pid', 'x', 'y', 'w', 'h', 'score', 'xx', 'yy', 'zz', 'occ'])
    
    reid_cls_annotation_process(det_df, gt_df, cls_vid_txt_root, reid_vid_txt_root)

    print('\n\n\n')

# generate reid images
for vid_nm in vid_nms:
    vid_id = vid_nm.split('_')[0]

    reid_vid_root = os.path.join(reid_root, vid_nm)
    reid_img_root = reid_vid_root + '/img'
    reid_df_path = reid_vid_root + '/txt/reid.txt'
    
    cls_img_root = os.path.join(cls_root, vid_nm) + '/img'
    cls_df_path = os.path.join(cls_root, vid_nm) + '/txt/cls.txt'
    
    create_folder(reid_img_root)
    create_folder(cls_img_root)

    reid_df = pd.read_csv(reid_df_path)
    cls_df = pd.read_csv(cls_df_path)
    
    num_fid = len(reid_df.fid.unique())
    for fid in tqdm(reid_df.fid.unique(), total = num_fid):
        img_root = os.path.join(vid_root, vid_nm)
        img_path = os.path.join(img_root, 'SEQ_{:}_{:}.jpg'.format(vid_id, str(fid).zfill(3)))
        img = cv2.imread(img_path)
        sub_reid_df = reid_df[reid_df.fid == fid]
        sub_cls_df = cls_df[cls_df.fid == fid]
        
        for idx, bbox in sub_reid_df.iterrows():
            _, pid, x, y, w, h, iou, did = bbox
            reid_img_out_path = os.path.join(reid_img_root, str(int(pid)).zfill(6))
            create_folder(reid_img_out_path, quiet=True)
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            reid_img = img[y1:y2, x1:x2, :]
            img_nm = '_'.join([str(x) for x in [int(pid), int(fid), x1, y1, x2, y2]]) 
            reid_img_nm = os.path.join(reid_img_out_path, img_nm +'.jpg')
            cv2.imwrite(reid_img_nm, reid_img)
        
        for idx, bbox in sub_cls_df.iterrows():
            _, _, x, y, w, h, score, did, iou, gid, cls = bbox
            cls_img_out_path = os.path.join(cls_img_root, str(int(fid)).zfill(6))
            create_folder(cls_img_out_path, quiet=True)
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            cls_img = img[y1:y2, x1:x2, :]
            img_nm = '_'.join([str(x) for x in [int(did), int(fid), x1, y1, x2, y2]]) 
            cls_img_nm = os.path.join(cls_img_out_path, img_nm +'.jpg')
            cv2.imwrite(cls_img_nm, cls_img)

# # create link to fast-reid data path
if not os.path.exists(fast_reid_data_path):
    os.symlink(os.path.abspath(reid_root),
               os.path.abspath(fast_reid_data_path), target_is_directory=True)
