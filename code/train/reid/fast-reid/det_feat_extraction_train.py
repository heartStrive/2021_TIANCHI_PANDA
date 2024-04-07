import os
from termcolor import colored

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager
from fastreid.data import build_reid_test_loader
from fastreid.data.transforms import build_transforms
from fastreid.data.common import CommDataset
import sys
sys.path.append('./demo/')
from predictor import FeatureExtractionDemo
from demo import setup_cfg
import easydict
import tqdm
import torch
import numpy as np

def create_folder(folder_path, quiet=False):
    if not os.path.exists(folder_path):
        if not quiet:
            print(colored("++ create folder : {:}".format(folder_path), 'red'))
        os.makedirs(folder_path)
    else:
        if not quiet:
            print(colored("++ {:} exists".format(folder_path), 'green'))

if __name__ == '__main__':

    # load ReID model
    config_params = easydict.EasyDict()
    config_params.config_file = "logs/Panda/bagtricks_R50/config.yaml"
    config_params.opts = ['MODEL.WEIGHTS', "logs/Panda/bagtricks_R50/model_best.pth"]
    config_params.dataset_name = 'PandaReID'
    config_params.parallel = True
    cfg = setup_cfg(config_params)
    extractor = FeatureExtractionDemo(cfg, parallel=config_params.parallel)


    # data transform
    transforms = build_transforms(cfg, is_train=False)
    # extraction
    root =  "../../../../user_data/tmp_data/classification/CLS-PANDA/"
    
    vid_nms = [x for x in os.listdir(root) if not x.endswith('zip')]
    test_batch_size = 256
    for vid_nm in vid_nms:
        print(colored("++ extract feats of {:} ".format(vid_nm), 'green'))
        # fetch img paths
        img_root = os.path.join(root, vid_nm, 'img')
        feat_root = os.path.join(root, vid_nm, 'feat')
        create_folder(feat_root)

        fids = os.listdir(img_root)
        feat_infos = []
        for fid in fids:
            img_fid_root = os.path.join(img_root, fid)
            feat_infos += [os.path.join(img_fid_root, x) for x in os.listdir(img_fid_root)]
        det_items = [(x, 0, 0) for x in feat_infos]
            
        # create det images dataloader
        test_set = CommDataset(det_items, transforms, relabel=False)
        test_loader, num_query = build_reid_test_loader(test_set=test_set, test_batch_size=test_batch_size, num_query=0)

        # extract features
        feats = []
        pids = []
        camids = []
        for (feat, pid, camid) in tqdm.tqdm(extractor.run_on_loader(test_loader), total=len(test_loader)):
            feats.append(feat)
            pids.extend(pid)
            camids.extend(camid)   
        det_feats = torch.cat(feats).numpy()
        print(colored("++ save feats at : {:}".format(os.path.join(feat_root, 'feat.npy')), 'red'))
        np.save(os.path.join(feat_root, 'feat.npy'), det_feats)
        np.save(os.path.join(feat_root, 'feat_info.npy'), feat_infos)

