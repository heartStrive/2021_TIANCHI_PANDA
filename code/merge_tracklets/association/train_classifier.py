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
    dets, feats, gts = load_data(dets_path, feats_path, feats_info_path, train = True)
    data_lst = split_data(dets, feats, gts, step = 50)
    
    # feature extraction
    sub_gts_lst = []
    sub_tp_feats_lst = []
    sub_st_feats_lst = []
    for data in data_lst:
        sub_dets, sub_feats, sub_gts = data
        sub_tp_feats, sub_st_feats = extract_det_features(sub_dets, sub_feats)
        sub_gts_lst.append(sub_gts)
        sub_tp_feats_lst.append(sub_tp_feats)
        sub_st_feats_lst.append(sub_st_feats)

    # format train-val data for tp_classifier & st_classifier classification    
    gts_df = pd.concat(sub_gts_lst)
    tp_feats_df = pd.concat(sub_tp_feats_lst)
    st_feats_df = pd.concat(sub_st_feats_lst)
    data_dict[vid_nm] = {
        'gt' : gts_df,
        'tp_feat' : tp_feats_df,
        'st_feat' : st_feats_df
    }

print("save feats data for tp classification and st classification")
np.save('./data/feats/feats_data.npy', data_dict)

# load train data
data = np.load("./data/feats/feats_data.npy", allow_pickle=True).item()

# train-val data for TP classifier
train_gts, train_tp_feats, train_st_feats = data['10_Huaqiangbei']['gt'], data['10_Huaqiangbei']['tp_feat'], data['10_Huaqiangbei']['st_feat']
val_gts, val_tp_feats, val_st_feats = data['02_OCT_Habour']['gt'], data['02_OCT_Habour']['tp_feat'], data['02_OCT_Habour']['st_feat']

X_train = train_tp_feats.values[:,1:]
y_train = train_gts.values[:,3].astype('int')
X_val = val_tp_feats.values[:,1:]
y_val = val_gts.values[:,3].astype('int')

# train TP classifier
clf = TruePositiveClassifier()
clf.train(X_train, y_train)
clf.save_model('./data/model/tpclf.joblib')
y_pred = clf.infer(X_val, proba=False)
clf.eval(y_val, y_pred)

# train-val data for ST classifier
X_train = train_st_feats.values[:,1:]
y_train = train_gts.values[:,4].astype('int')
X_val = val_st_feats.values[:,1:]
y_val = val_gts.values[:,4].astype('int')

# train ST classifier
clf = StatusClassifier()
clf.train(X_train, y_train)
clf.save_model('./data/model/stclf.joblib')
y_pred = clf.infer(X_val, proba=False)
clf.eval(y_val, y_pred)
