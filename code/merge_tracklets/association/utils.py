from operator import sub
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from termcolor import colored
import torch
import torchvision
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------------------------
# functions related to data loading
def load_data(dets_path, feats_path, feats_info_path, train = True):
    if train:
        dets = pd.read_csv(dets_path)        
        gts = pd.DataFrame({'did' : dets.did, 'fid': dets.fid, 'gid' : dets.gid, 'tp' : dets.cls}).copy()
        dets = dets[['did', 'fid', 'x', 'y', 'w', 'h', 'score']].copy()

        feats = np.load(feats_path)
        feats_info = np.load(feats_info_path)        
        dids = [int(info.split('/')[-1].split('_')[0]) for info in feats_info]
        feats = pd.DataFrame(feats)
        feats['did'] = dids
        feats.sort_values('did', inplace=True, ignore_index=True)
    else:
        dets = pd.read_csv(dets_path)        
        gts = None
        dets = dets[['did', 'fid', 'x', 'y', 'w', 'h', 'score']].copy()

        feats = np.load(feats_path)
        feats_info = np.load(feats_info_path)        
        dids = [int(info.split('/')[-1].split('_')[0]) for info in feats_info]
        feats = pd.DataFrame(feats)
        feats['did'] = dids
        feats.sort_values('did', inplace=True, ignore_index=True)
        return dets, feats, gts

def split_data(dets, feats, label=None, step = 30):

    s = dets.fid.unique().min()
    e = dets.fid.unique().max()
    sfids = np.arange(s, e, step).tolist()
    efids = sfids[1:] + [e+1]

    data_lst = []
    print('++ split data with {} frames step : '.format(step))
    for i in tqdm(range(len(sfids))):
        sfid = sfids[i]
        efid = efids[i]
        sub_dets = dets[(dets.fid >= sfid) & (dets.fid <= efid)].copy()
        sub_feats = feats.iloc[sub_dets.index.values].copy()
        if label is None:
            data_lst.append([sub_dets, sub_feats])
        else:
            gts_part1 = label.iloc[sub_dets.index.values].copy()
            gts_part2 = cal_status(gts_part1.copy())
            sub_label = pd.merge(gts_part1, gts_part2, on='did')
            data_lst.append([sub_dets, sub_feats, sub_label])
    
    return data_lst

# -----------------------------------------------------------------------------------------------
# detection status (en-ex) related function
def det_status(sub_gt):    
    did = sub_gt.copy().index.values
    fid = sub_gt.copy().fid.values

    is_start = (fid == fid.min()).astype(int)
    is_end = (fid == fid.max()).astype(int)
    is_inside = ( (fid > fid.min()) & (fid < fid.max())).astype(int)
    status = np.argmax(np.c_[is_start, is_end, is_inside], axis=1)

    gt_status = pd.DataFrame({
                        'did' : did,
                        'status' : status, # 0-start, 1-end, 2-inside
                    })
    return gt_status

def cal_status(gt):
    gt_status = pd.concat([det_status(sub_gt) for _, sub_gt in gt.groupby('gid')])
    gt_status.sort_index(inplace=True)
    return gt_status

# -----------------------------------------------------------------------------------------------
# det-tp related function
def calc_crowdness(boxes):
    num_boxes = boxes.shape[0]
    boxes_np = boxes[:,:4]
    score_np = boxes[:,4]

    boxes_tensor = torch.from_numpy(boxes_np)
    boxes_tensor = torchvision.ops.box_convert(boxes_tensor, 'xywh','xyxy')
    score_tensor = torch.from_numpy(score_np)
    
    nms_feat = np.zeros((num_boxes, 10))
    for i, thr in enumerate(np.linspace(0, 1, 11)):
        if (thr !=1):
            keep_idx = torchvision.ops.nms(boxes_tensor, score_tensor.reshape(-1), thr)
            nms_feat[keep_idx,i] = 1

    crowdness = 1- nms_feat.sum(axis=1)/10 # add weights
       
    return crowdness

def extract_tp_features(dets):
    fids = sorted(dets.fid.unique())

    crowdness_lst = []
    for fid in fids:
        det = dets[dets.fid == fid]
        did = det.did.values
        score = det.score.values
        boxes = det[['x', 'y', 'w', 'h', 'score']].values
        crowdness = calc_crowdness(boxes)
        tp_feat = np.c_[did, score, crowdness]
        crowdness_lst.append(tp_feat)

    tp_feats = pd.DataFrame(np.vstack(crowdness_lst), columns=['did', 'score', 'crowdness'])
    tp_feats.did = tp_feats.did.astype('int')
    return tp_feats

# -----------------------------------------------------------------------------------------------
# det-status related
def fetch_time_local(det, fid, sfid, efid, time_range = 10):
    # fetch scand det inside time range
    if fid > sfid:
        if fid < time_range:
            scand_det = det[det.fid < fid]
        else:
            scand_det = det[(det.fid < fid) & (det.fid >= (fid - time_range))]
    else:
        scand_det = None
    
    # fetch ecand det inside time range
    if fid < efid:
        if fid+time_range > efid:
            ecand_det = det[det.fid > fid]
        else:
            ecand_det = det[(det.fid > fid) & (det.fid <= (fid + time_range))]
    else:
        ecand_det = None
    
    return scand_det, ecand_det


def extract_status_features(dets, feats, time_range=10, ktop=10, bins=10):
    sfid = dets.fid.min()
    efid = dets.fid.max()
    cur_efeat_lst = []
    cur_sfeat_lst = []
    cur_did_lst = []
    for fid, cur_det in tqdm(dets.groupby('fid'), total = dets.fid.unique().size):
        scand_det, ecand_det = fetch_time_local(dets, fid, sfid, efid, time_range)
        cur_feat = feats.loc[cur_det.index].values[:,:-1]
        
        if ecand_det is not None:
            ecand_feat = feats.loc[ecand_det.index].values[:,:-1]    
            app_dist_mat = 1 - (cosine_similarity(cur_feat, ecand_feat) + 1) / 2    
            cur_efeat = np.sort(app_dist_mat, axis=1)[:,:ktop]
        else:
            cur_efeat = np.ones((cur_feat.shape[0], ktop))
        
        if scand_det is not None:
            scand_feat = feats.loc[scand_det.index].values[:,:-1]
            app_dist_mat = 1 - (cosine_similarity(cur_feat, scand_feat) + 1) / 2    
            cur_sfeat = np.sort(app_dist_mat, axis=1)[:,:ktop]
        else:
            cur_sfeat = np.ones((cur_feat.shape[0], ktop))
        
        cur_efeat_lst.append(cur_efeat)
        cur_sfeat_lst.append(cur_sfeat)
        cur_did_lst.append(feats.loc[cur_det.index].values[:,-1].astype(int))
    
    # return cur_efeat_lst, cur_sfeat_lst, cur_idx_lst
    start_feat = pd.DataFrame(np.concatenate(cur_sfeat_lst)).apply(lambda x : (np.histogram(x, bins=bins, range=[0,1])[0]/ktop).tolist(), axis=1)
    end_feat = pd.DataFrame(np.concatenate(cur_efeat_lst)).apply(lambda x : (np.histogram(x,bins=bins, range=[0,1])[0]/ktop).tolist(), axis=1)
    # return np.concatenate(cur_idx_lst), start_feat, end_feat
    scol_nm = ['did'] + ['s'+str(i) for i in range(bins)]
    ecol_nm = ['did'] + ['e'+str(i) for i in range(bins)]
    
    status_sfeat = pd.DataFrame(np.c_[np.concatenate(cur_did_lst), np.array(start_feat.values.tolist())], columns = scol_nm)
    status_efeat = pd.DataFrame(np.c_[np.concatenate(cur_did_lst), np.array(end_feat.values.tolist())], columns = ecol_nm)

    status_sfeat.did = status_sfeat.did.astype('int')
    status_efeat.did = status_efeat.did.astype('int')
    
    return status_efeat, status_sfeat

# -----------------------------------------------------------------------------------------------
# extract detection related feat extraction
def extract_det_features(dets, feats, time_range = 5, ktop=20):
    tp_feats = extract_tp_features(dets)
    ste_feats, sts_feats = extract_status_features(dets, feats, time_range=time_range, ktop=ktop)
    st_feats = pd.merge(ste_feats, sts_feats, on='did')
    return tp_feats, st_feats

# ----------------------------------------------------------------------------------------------
# trans prob related 

def inverse_decay(x):
    return 1/x

def get_trans_prob(det, feats, thr = 0.7, time_range=30, width_thr=3):
    feat = feats.values[:,:-1]
    adist_mat = (cosine_similarity(feat, feat) + 1) / 2
    cons_prob_lst = []
    x = det.x.values
    width = det.w.values

    fids = det.fid.values
    for i, adist in tqdm(enumerate(adist_mat), total = adist_mat.shape[0]):
        tdist = fids[i] - fids
        tdist[tdist <= 0] = time_range
        w_t = inverse_decay(tdist)
        w_d = ((np.abs((x[i] - x) / width[i])) <= width_thr).astype(float)   
        cons_prob = adist * w_t * w_d
        cons_prob_lst.append(cons_prob)

    cons_prob_mat = np.array(cons_prob_lst)
    indices = np.argwhere(cons_prob_mat > thr)
    prob = cons_prob_mat[indices[:, 0], indices[:,1]]
    return np.c_[indices, prob]

# -------------------------------------------------------------------------------------------
def prob2cost(prob):
    return -np.log(prob)

def tpprob2cost(prob):
    return np.log((1-prob)/prob)

def get_tp_prob(tp_clf, tp_feats):
    return tp_clf.infer(tp_feats.values[:,1:])

def get_st_prob(st_clf, st_feats):
    return st_clf.infer(st_feats.values[:,1:])[:,:2]

def prob2arcs(tp_prob, st_prob, trans_prob):
    ij = trans_prob[:,:2]
    valid_did = np.unique(ij.reshape(-1)).astype('int')
    tp_cost = tpprob2cost(tp_prob[valid_did])
    st_cost = prob2cost(st_prob[valid_did])
    trans_cost = prob2cost(trans_prob[:,2])

    # generate codebook for did
    code_book = {}
    icode_book = {}
    for a, b in enumerate(valid_did):
        code_book[a+1] = b
        icode_book[b] = a+1

    nij = []
    for x in ij:
        nij.append([icode_book[x[0]], icode_book[x[1]]])
    nij = np.array(nij)

    transition_arcs = np.c_[nij, trans_cost]
    detection_arcs = np.c_[np.arange(valid_did.shape[0])+1, st_cost, tp_cost]

    return transition_arcs, detection_arcs, code_book, icode_book

def fetch_tracklets_det(code_book, trklets, dets):
    return dets.iloc[np.array([code_book[i] for i in trklets])]

def fetch_tracklets_gt(code_book, trklets, gts):
    return gts.iloc[np.array([code_book[i] for i in trklets])]

def trj2result(traj, code_book, sub_dets):
    result_lst = []
    for trj_id, trj in enumerate(traj):
        tmp_dets = fetch_tracklets_det(code_book, trj, sub_dets)
        trj_result = sub_dets.loc[tmp_dets.did].iloc[:,1:-1]
        trj_result.insert(1, 'id', trj_id+1)
        trj_result.insert(6,'score',1)
        trj_result.insert(7,'xx',-1)
        trj_result.insert(8,'yy',-1)
        trj_result.insert(9,'zz',-1)
        result_lst.append(trj_result)
    return pd.concat(result_lst)
