import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

from tracker.multitracker import JDETracker, JDE_PANDA_Tracker
import visualization as vis
from log import logger
from timer import Timer
# from utils.evaluation_panda import Evaluator
from utils.evaluation import Evaluator
# import utils.datasets as datasets
# import torch
# from utils.utils import *
import zipfile
import numpy as np

import os
import sys
import pandas as pd

def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
    print('save in {}'.format(output_filename))

def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)
def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
    print('save in {}'.format(output_filename))


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


# def eval_panda_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30,num_deal=300):
def eval_panda_seq( cls_df, did_feat, opt, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30,num_deal=300):
    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = JDE_PANDA_Tracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    cnt = 0

    num_frame = cls_df.loc[:,"fid"].max()
    for frame_id in range(0, num_frame):
        if frame_id % 5 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1.0 / max(1e-5, timer.average_time)))
        cnt += 1
        if cnt > num_deal:
            break
        # 取当前帧数据
        frame_feat = []
        bbox = []
        frame_cls_df = cls_df[(cls_df["fid"] == frame_id+1)]
        for i in range(len(frame_cls_df)):
            det = frame_cls_df.iloc[i]
            did = det['did']
            f = did_feat[did]
            bbox.append([det['x'],det['y'],det['w'],det['h'],det['score']])
            frame_feat.append(f)
        bbox = np.array(bbox)
        frame_feat = np.array(frame_feat)
        # print(len(bbox),len(frame_cls_df))
        # print('bbox:',bbox.shape,frame_feat.shape)
        # print(bbox[0], frame_feat[0])
        # input()
        # idx = cls_df[(cls_df["fid"] == frame_id+1)].index.tolist()
        # frame_feat = feat[idx]
        # run tracking
        timer.tic()
        # running tracking
        online_targets = tracker.update(bbox, frame_feat)
        online_tlwhs = []
        online_ids = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            #vertical = tlwh[2] / tlwh[3] > 1.6 # 过滤宽高比不满足条件的框
            # min_box_area = 50
            #if tlwh[2] * tlwh[3] > opt.min_box_area: #and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
        timer.toc()
        print('Total cost time: {} s'.format(timer.interval()))
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        # save_dir = '/home/fair/Desktop/zls/tianchi/panda_det_reid_feat/panda_project/code/test/show_results'
        if show_image or save_dir is not None:
            video_name = result_filename.split('.')[0].split('/')[1]
            src_giga_img_path = '/raid/panda_track/image_train/'+video_name+'/SEQ_'+video_name.split('_')[0]+\
                '_'+str(frame_id+1).zfill(3)+'.jpg'
            src_giga_img = cv2.imread(src_giga_img_path)
            online_im = vis.plot_tracking_panda(
                src_giga_img, online_tlwhs, online_ids, frame_id=frame_id, fps=1.0 / timer.average_time, scales=0.2
            )
        if show_image:
            cv2.imshow('online_im', online_im)
        
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        # print('show:',frame_id)
        # input()
        # if frame_id>=9:
        #     break
        # frame_id += 1
        
    # save results
    # result_filename = '/home/fair/Desktop/zls/tianchi/PANDA-round2/results/'
    write_results(result_filename, results, data_type)

    return frame_id, timer.average_time, timer.calls


def main_for_panda(
    # feat,
    cls_df,
    did_feat,
    opt,
    data_root='test_frames',
    det_root=None,
    seqs=('IMG',),
    exp_name='demo',
    save_images=False,
    save_videos=False,
    show_image=True,
):
    logger.setLevel(logging.INFO)
    # result_root = os.path.join('results', exp_name)
    result_root = 'results'
    output_root = 'vis_results'

    mkdir_if_missing(result_root)
    mkdir_if_missing(output_root)

    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    logger.info('start seq: {}'.format(exp_name))

    # dataloader = datasets.LoadPandaImages(data_root, opt.img_size)
    result_filename = os.path.join(result_root, '{}.txt'.format(exp_name))
    output_dir = os.path.join(output_root, exp_name) if save_images or save_videos else None

    frame_rate = 30

    nf, ta, tc = eval_panda_seq(
        # feat,
        cls_df,
        did_feat,
        opt,
        # dataloader,
        data_type,
        result_filename,
        save_dir=output_dir,
        show_image=show_image,
        frame_rate=frame_rate,
        # num_deal=50
    )
    n_frame += nf
    timer_avgs.append(ta)
    timer_calls.append(tc)
    # eval
    seq = exp_name
    # seq = '01_University_Canteen_first10'
    # logger.info('Evaluate seq: {}'.format(seq))
    # evaluator = Evaluator(data_root, seq, data_type)
    # accs.append(evaluator.eval_file(result_filename))

    if save_videos:
        output_video_path = osp.join(output_dir, '{}.mp4'.format(exp_name))
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
        os.system(cmd_str)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    # metrics = mm.metrics.motchallenge_metrics
    # mh = mm.metrics.create()
    # summary = Evaluator.get_summary(accs, seqs, metrics)
    # strsummary = mm.io.render_summary(
    #     summary,
    #     formatters=mh.formatters,
    #     namemap=mm.io.motchallenge_metric_names
    # )
    # print(strsummary)
    # Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    
    parser.add_argument('--img-size', type=int, default=(1088, 608), help='size of each image dimension')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=50, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=300, help='tracking buffer')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    root = '../../user_data/tmp_data/classification/CLS-PANDA/'
    for video in os.listdir(root):
        # video = '01_University_Canteen'
        video_path = os.path.join(root, video)
        feat_path = os.path.join(video_path, 'feat/feat.npy')
        feat_info_path = os.path.join(video_path, 'feat/feat_info.npy')
        cls_df_path = os.path.join(video_path, 'txt/cls.txt')
        feat = np.load(feat_path)
        cls_df = pd.read_csv(cls_df_path)
        feat_info = np.load(feat_info_path)
        # 先转换成 {did: feat}
        did_feat = {}
        for i in range(len(feat)):
            info = feat_info[i]
            img_name = info.split('/')[-1].split('.')[0]
            did = int(img_name.split('_')[0])
            did_feat[did] = feat[i]

        main_for_panda(
            # feat,
            cls_df,
            did_feat,
            opt,
            data_root=video_path,
            exp_name=video,
            show_image=False,
            save_images=False,
            save_videos=False,
        )
        # print(video)
        # input()
        # break

