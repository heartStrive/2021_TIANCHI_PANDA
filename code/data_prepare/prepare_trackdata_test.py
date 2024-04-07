import os
import json
import pandas as pd
from tqdm import tqdm

root="../../user_data/tmp_data/round-2/video_test/"
ann_root = '../../user_data/tmp_data/round-2/video_annos/'
out_root = "../../user_data/tmp_data/tracking/MOT-PANDA/"
vid_nms = [x for x in os.listdir(root) if not x.endswith("zip")]

vid_root = "../../user_data/tmp_data/tracking/MOT-PANDA"
if not os.path.exists(vid_root):
    print("create folder : {:}".format(vid_root))
    os.makedirs(vid_root)

for vid_nm in vid_nms:
    vid_path = os.path.join(vid_root, vid_nm)
    if not os.path.exists(vid_path):
        print("create folder : {:}".format(vid_path))
        os.makedirs(vid_path)

for vid_nm in vid_nms:
    det_path = os.path.join(vid_root, vid_nm, "det")
    #gt_path = os.path.join(vid_root, vid_nm, "gt")
    img1_path = os.path.join(vid_root, vid_nm, "img1")
    
    if not os.path.exists(det_path):
        print("create folder : {:}".format(det_path))
        os.makedirs(det_path)
    # if not os.path.exists(gt_path):
    #     print("create folder : {:}".format(gt_path))
    #     os.makedirs(gt_path)
    if not os.path.exists(img1_path):
        os.symlink(os.path.abspath(os.path.join(root, vid_nm)),
                   os.path.abspath(img1_path), target_is_directory=True)

# for vid_nm in tqdm(vid_nms, total = len(vid_nms)):
#     seqinfo_path = os.path.join(ann_root, vid_nm, 'seqinfo.json')
#     tracks_path = os.path.join(ann_root, vid_nm, "tracks.json")
#     with open(seqinfo_path) as f:
#         seqinfo = json.load(f)
#     with open(tracks_path) as f:
#         tracks = json.load(f)
    
#     imWidth = seqinfo['imWidth']
#     imHeight = seqinfo['imHeight']
#     gt = []
#     for track in tracks:
#         trk_id = track['track id']
#         dets = track['frames']
#         for det in dets:
#             frame = det['frame id']
#             rect = det['rect']
#             occ = det['occlusion']

#             bb_left = max(0, rect['tl']['x'] * imWidth)
#             bb_top = max(0, rect['tl']['y'] * imHeight)
#             bb_right = min(imWidth - 1, rect['br']['x'] * imWidth)
#             bb_bottom = min(imHeight - 1, rect['br']['y'] * imHeight)

#             bb_width = bb_right - bb_left
#             bb_height = bb_bottom - bb_top

#             if (bb_width > 0) & (bb_height > 0) & (bb_width * bb_height >= 25):
#                 ann_gt = [frame, trk_id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1, occ]
#                 gt.append(ann_gt)

#     gt_df = pd.DataFrame(gt)
#     gt_path = os.path.join(out_root, vid_nm, 'gt/gt.txt')
#     gt_df.to_csv(gt_path, index=False, header=False)