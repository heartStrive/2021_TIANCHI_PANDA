# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import os
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PandaReID(ImageDataset):

    dataset_dir = 'panda_reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        set_nms = sorted(os.listdir(self.dataset_dir))
        bboxes = self.fetch_bboxes(set_nms)
        train, query, gallery = self.split_bboxes(bboxes)
        super(PandaReID, self).__init__(train, query, gallery, **kwargs)
        

    def fetch_bboxes(self, set_nms):
        bboxes = []
        pid = 0
        camid = 0
        for set_nm in set_nms:
            set_root = os.path.join(self.dataset_dir, set_nm) + '/img'
            track_nms = sorted(os.listdir(set_root))
            for track_nm in track_nms:
                img_root = os.path.join(set_root, track_nm)
                img_nms = sorted(os.listdir(img_root))
                img_paths = [os.path.join(img_root, x) for x in img_nms]
                for img_path in img_paths:
                    bboxes.append((img_path, pid, camid))
                pid += 1
            camid += 1
        
        return bboxes
    
    def split_bboxes(self, bboxes):
        train = []
        test = []
        for bbox in bboxes:
            pid = bbox[1]
            if pid < 7000:
                train.append(bbox)
            else:
                test.append(bbox)
        
        query = []
        gallery = []
        for bbox in test:
            img = bbox[0]
            pid = bbox[1]
            camid = bbox[2]

            fid = int(img.split("/")[-1].split("_")[1])
            if fid % 3 == 0:
                query.append((img, pid, 0))
            else:
                gallery.append((img, pid, 1))
        
        return train, query, gallery