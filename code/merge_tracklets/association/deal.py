import numpy as np
import os
def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

cls_root = "../../../user_data/tmp_data/classification/CLS-PANDA/"
data = np.load('track_result.npy',allow_pickle=True).item()
results_root = 'results'
mkdir_if_missing(results_root)
vid_nms = sorted(os.listdir(cls_root))
for vid_id, vid_nm in enumerate(vid_nms):
    item = data[vid_nm]
    item.to_csv(os.path.join(results_root, vid_nm+'.txt'), index=False, header=None)

