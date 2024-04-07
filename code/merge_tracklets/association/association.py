import os
import numpy as np
import pandas as pd
import seaborn as sns
from utils import *
from model import *
from pycinda.src_python.algo import mcc4mot
import motmetrics as mm
from evaluation_panda import Evaluator
import time

cls_root = "../../../user_data/tmp_data/classification/CLS-PANDA/"
motgt_root = '../../../user_data/tmp_data/tracking/MOT-PANDA/'
tp_model_path = './data/model/tpclf.joblib'
st_model_path = './data/model/stclf.joblib'
vid_nms = sorted(os.listdir(cls_root))
time_step = 80
eval_mot = False

tp_clf = TruePositiveClassifier(model_path=tp_model_path)
st_clf = StatusClassifier(model_path=st_model_path)

result_dict = {}

for vid_id, vid_nm in enumerate(vid_nms):
    t0 = time.time()
    # load data
    #motgt_path = os.path.join(motgt_root, vid_nm, 'gt/gt.txt')
    dets_path = os.path.join(cls_root, vid_nm, 'txt/cls.txt')
    feats_path = os.path.join(cls_root, vid_nm, 'feat/feat.npy')
    feats_info_path = os.path.join(cls_root, vid_nm, 'feat/feat_info.npy')
    dets, feats, gts = load_data(dets_path, feats_path, feats_info_path, train = False)
    data_lst = split_data(dets, feats, gts, step = time_step)

    # association
    result_lst = []
    for data_id, data in enumerate(data_lst):
        sub_dets, sub_feats = data
        sub_tp_feats, sub_st_feats = extract_det_features(sub_dets, sub_feats)

        tp_prob = get_tp_prob(tp_clf, sub_tp_feats)
        st_prob = get_st_prob(st_clf, sub_st_feats)
        trans_prob = get_trans_prob(sub_dets, sub_feats)
        
        tp_prob = get_tp_prob(tp_clf, sub_tp_feats)
        st_prob = get_st_prob(st_clf, sub_st_feats, )
        trans_prob = get_trans_prob(sub_dets, sub_feats, time_range=time_step)
        transition_arcs, detection_arcs, code_book, icode_book = prob2arcs(tp_prob, st_prob, trans_prob)
        trklets, cost = mcc4mot(detection_arcs, transition_arcs)
        
        result = trj2result(trklets, code_book, sub_dets)
        result_lst.append(result)

    # combine tracklets
    for r_id in range(len(result_lst)):
        if r_id > 0:
            r_id_max = result_lst[r_id-1].id.max()
            result_lst[r_id].id += r_id_max

    merged_result = pd.concat(result_lst)
    merged_result.to_csv('sample_result.txt', header=None, index=False)
    
    t1 = time.time()
    print("association {:} cost : {:}s".format(vid_nm, t1 - t0))
    if eval_mot:
        motgt_path = os.path.join(motgt_root, vid_nm, 'gt/gt.txt')
        accs = []
        data_type = 'mot'
        seqs=(vid_nm,)
        ###
        gt_path = motgt_path # 'sample_gt.txt' # grount truth文件位置 .txt
        result_filename = 'sample_result.txt' # prediction文件位置 .txt , 格式为：save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

        evaluator = Evaluator(gt_path, data_type)
        accs.append(evaluator.eval_file(result_filename))

        # get summary
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(accs, seqs, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)

    result_dict[vid_nm] = merged_result

print("save tracking results ")
np.save('track_result.npy', result_dict)