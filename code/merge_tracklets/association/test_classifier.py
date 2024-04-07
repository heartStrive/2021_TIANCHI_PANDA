import os
import numpy as np
import pandas as pd
import seaborn as sns
from utils import *
from model import *

cls_root = "../../../user_data/tmp_data/classification/CLS-PANDA/"
vid_nms = os.listdir(cls_root)

data_dict = {}
for vid_id, vid_nm in enumerate(vid_nms):
    # load data
    dets_path = os.path.join(cls_root, vid_nm, 'txt/cls.txt')
    feats_path = os.path.join(cls_root, vid_nm, 'feat/feat.npy')
    feats_info_path = os.path.join(cls_root, vid_nm, 'feat/feat_info.npy')
    dets, feats, gts = load_data(dets_path, feats_path, feats_info_path, train = False)
    data_lst = split_data(dets, feats, gts, step = 50)
    
    # feature extraction
    sub_tp_feats_lst = []
    sub_st_feats_lst = []
    for data in data_lst:
        sub_dets, sub_feats = data
        sub_tp_feats, sub_st_feats = extract_det_features(sub_dets, sub_feats)
        sub_tp_feats_lst.append(sub_tp_feats)
        sub_st_feats_lst.append(sub_st_feats)

    # format train-val data for tp_classifier & st_classifier classification    
    tp_feats_df = pd.concat(sub_tp_feats_lst)
    st_feats_df = pd.concat(sub_st_feats_lst)
    data_dict[vid_nm] = {
        'tp_feat' : tp_feats_df,
        'st_feat' : st_feats_df
    }

print("save feats data for tp classification and st classification")
np.save('./data/feats/feats_data.npy', data_dict)
