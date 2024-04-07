import os
from distutils.dir_util import copy_tree

def create_folders(folders):
    for folder_path in folders:
        if not os.path.exists(folder_path):
            print("create folder : {:}".format(folder_path))
            os.makedirs(folder_path)
        else:
            print("exists : {:}".format(folder_path))

def copy_folders(src_lst, dst):
    for src in src_lst:
        print("copy folder from {:} to {:}".format(src, dst))
        copy_tree(src, dst)

'''
stucture dataset to:

|--user_data
    |--tmp_data
        |--round-1
            |--Train
                |--image_train
                    |--scene 1
                        |-- XXX.png
                        ...
                        |-- XXX.png
                    ...
                    |--scene N
                        |-- XXX.png
                        |-- XXX.png
                        ...
                        |-- XXX.png
                |--image_annos
                    |-- human_bbox_train.json
                    |-- vehicle_bbox_train.json
        |--round-2
            |--video_train
                |--scene 1
                    |-- XXX.jpg
                    ...
                    |-- XXX.jpg
                ...
                |--scene N
                    |-- XXX.jpg
                    ...
                    |-- XXX.jpg
            |--video_annos
                |--scene 1
                    |-- seqinfo.json
                    |-- tracks.json
                ...
                |--scene N
                    |-- seqinfo.json
                    |-- tracks.json
            |--video_test
                |--scene 1
                    |-- XXX.jpg
                    ...
                    |-- XXX.jpg
                ...
                |--scene N
                    |-- XXX.jpg
                    ...
                    |-- XXX.jpg
        |--detection
        |--reid
        |--tracking
'''

# prepare round 1 folder structure
round1_root = "../../user_data/tmp_data/round-1"
round1_img_train_path = os.path.join(round1_root, "image_train")
round1_ann_train_path = os.path.join(round1_root, "image_annos")
round1_folders = [round1_root, round1_img_train_path, round1_ann_train_path]
create_folders(round1_folders)

# move round 1 image data
round1_train_img_part1 = "../../tcdata/panda_round1_train_202104_part1/"
round1_train_img_part2 = "../../tcdata/panda_round1_train_202104_part2/"
copy_folders([round1_train_img_part1, round1_train_img_part2], round1_img_train_path)

# move round 1 annotations
round1_train_annos = "../../tcdata/panda_round1_train_annos_202104/"
copy_folders([round1_train_annos], round1_ann_train_path)

# prepare round 2 folder structure
round2_root = "../../user_data/tmp_data/round-2"
round2_vid_train_path = os.path.join(round2_root, "video_train")
round2_ann_train_path = os.path.join(round2_root, "video_annos")

round2_folders = [round2_root, round2_vid_train_path, round2_ann_train_path]
create_folders(round2_folders)

# move round 2 video data
round2_train_vid = ["../../tcdata/panda_round2_train_20210331_part{:}".format(x) for x in range(1, 3)]
copy_folders(round2_train_vid, round2_vid_train_path)

# move round 2 annotations
round2_train_annos = "../../tcdata/panda_round2_train_annos_20210331/"
copy_folders([round2_train_annos], round2_ann_train_path)

# make folders for detection, reid, tracking
det_root = "../../user_data/tmp_data/detection"
reid_root = "../../user_data/tmp_data/reid"
trk_root = "../../user_data/tmp_data/tracking"
create_folders([det_root, reid_root, trk_root])

#!!!!!!
# round2_vid_train_path = os.path.join(round2_root, "video_test") 
