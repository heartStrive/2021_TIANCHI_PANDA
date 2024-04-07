import sys
import time
from termcolor import colored
from pycocotools.coco import COCO
import os
import utils

# data path
round1_root = "../../user_data/tmp_data/round-1/"
round1_img_root = os.path.join(round1_root, 'image_train')
round2_root = "../../user_data/tmp_data/round-2/"
round2_img_root = os.path.join(round2_root, 'video_train')

# create folder and paths
detection_root = "../../user_data/tmp_data/detection"
ann_root =  os.path.join(detection_root, 'annotations')
tile_root = os.path.join(detection_root, 'image_tiles')

img_train_root = os.path.join(detection_root, 'image_tiles_train') 
img_val_root = os.path.join(detection_root, 'image_tiles_val')

yolo_ann_root = os.path.join(detection_root, 'yolo_annotations')
yolo_fbody_ann_root = os.path.join(yolo_ann_root, "fbody")
fbody_ann_train_root = os.path.join(yolo_ann_root, "fbody_train")
fbody_ann_val_root = os.path.join(yolo_ann_root, "fbody_val")

yolov5_root = os.path.join(detection_root, "yolov5_data")
yolov5_fbody_root = os.path.join(yolov5_root, "fbody")
yolov5_fbody_img_root = os.path.join(yolov5_fbody_root, "images")
yolov5_fbody_img_train_root = os.path.join(yolov5_fbody_img_root, 'train')
yolov5_fbody_img_val_root = os.path.join(yolov5_fbody_img_root, 'val')

yolov5_fbody_label_root = os.path.join(yolov5_fbody_root, "labels")
yolov5_fbody_label_train_root = os.path.join(yolov5_fbody_label_root, 'train')
yolov5_fbody_label_val_root = os.path.join(yolov5_fbody_label_root, 'val')

folder_paths = [ann_root, tile_root,
                img_train_root, img_val_root, 
                yolo_ann_root, yolo_fbody_ann_root,
                fbody_ann_train_root, fbody_ann_val_root,
                yolov5_root, yolov5_fbody_root, 
                yolov5_fbody_img_root, yolov5_fbody_label_root]

utils.create_folders(folder_paths)

# ----------------------------------------------------------------------------------------------------
# convert round1 panda dataset to coco format
round1_tgtfile = os.path.join(ann_root, 'panda_coco_round1.json')
if os.path.isfile(round1_tgtfile):
    print(colored('++ panda_coco_round1.json exists', 'green'))
else:
    ts = time.time()
    print(colored("++ convert pandas round1 annotations to coco format", 'green'))
    utils.round1_coco(round1_root, round1_tgtfile)
    te = time.time()
    print(colored("-- conversion finished, cost : {:.2f}s".format(te - ts), 'red'))
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# convert round2 panda dataset to coco format
round2_tgtfile = os.path.join(ann_root, 'panda_coco_round2.json')
if os.path.isfile(round2_tgtfile):
    print(colored('++ panda_coco_round2.json exists', 'green'))
else:
    ts = time.time()
    print(colored("++ convert pandas round2 annotations to coco format", 'green'))
    utils.round2_coco(round2_root, round2_tgtfile)
    te = time.time()
    print(colored("-- conversion finished, cost : {:.2f}s".format(te - ts), 'red'))
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# generate trainval images and annotations in coco format
subwidth, subheight = 2560, 2560
gap = (400, 800)
scale_lst = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
#scale_lst = [0.25, 0.125]
print("\tscales : ", scale_lst)
print("\ttile width, tile height : ", subwidth, subheight)
print("\tgap width, gap height : ", gap[0], gap[1])

round1_tiles_annfile = os.path.join(ann_root, "round1_tiles_trainval.json")
round2_tiles_annfile = os.path.join(ann_root, "round2_tiles_trainval.json")

# split round1 data
if os.path.isfile(round1_tiles_annfile):
    print(colored('++ round1_tiles_trainval.json exists', 'green'))
else:
    round1_coco = COCO(round1_tgtfile)
    # round1_coco = utils.coco_subannos(round1_coco, [1,2], 5)
    print(colored("++ split round1 orignal pandas images into tiles : ", 'green'))
    ts = time.time()
    utils.tiles_annos(round1_coco, round1_img_root, 
                    scale_lst, subwidth, subheight, gap, 
                    tile_root, round1_tiles_annfile)
    te = time.time()
    print(colored("-- split round1 finished, cost : {:.2f}s".format(te - ts)), 'red')

# split round2 data
if os.path.isfile(round2_tiles_annfile):
    print(colored('++ round2_tiles_trainval.json exists', 'green'))
else:
    round2_coco = COCO(round2_tgtfile)
    round2_coco = utils.coco_subannos(round2_coco, [i+1 for i in range(10)], 30)
    print(colored("++ split round2 orignal pandas videos into tiles : ", 'green'))
    ts = time.time()
    utils.tiles_annos(round2_coco, round2_img_root, 
                    scale_lst, subwidth, subheight, gap, 
                    tile_root, round2_tiles_annfile)
    te = time.time()
    print(colored("-- split round2 finished, cost : {:.2f}s".format(te - ts)), 'red')
# ----------------------------------------------------------------------------------------------------

# convert coco to yolo format data
round1_tiles_coco = COCO(round1_tiles_annfile)
round2_tiles_coco = COCO(round2_tiles_annfile)
print(colored("++ split tile annotations by classes", 'green'))
ts = time.time()
yolo_outRoot = yolo_fbody_ann_root
utils.yolo_cat_anns(round1_tiles_coco, tile_root, yolo_outRoot)
utils.yolo_cat_anns(round2_tiles_coco, tile_root, yolo_outRoot)
te = time.time()
print(colored("-- yolo annotations finished, cost : {:.2f}s".format(te - ts), 'red'))
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# trainval split
print(colored("++ train-val split", 'green'))
ts = time.time()
utils.trainval_split(tile_root, img_val_root, img_train_root, pattern="_01_")
utils.trainval_split(yolo_fbody_ann_root, fbody_ann_val_root, fbody_ann_train_root, pattern="_01_")
te = time.time()
print(colored("-- train-val split finished, cost : {:.2f}".format(te - ts), 'red'))

# ----------------------------------------------------------------------------------------------------
print(colored("++ create data link to train yolov5", 'green'))
if not os.path.exists(yolov5_fbody_img_train_root):
    os.symlink(os.path.abspath(img_train_root),
            os.path.abspath(yolov5_fbody_img_train_root), target_is_directory=True)    

if not os.path.exists(yolov5_fbody_img_val_root):
    os.symlink(os.path.abspath(img_val_root),
            os.path.abspath(yolov5_fbody_img_val_root), target_is_directory=True)

if not os.path.exists(yolov5_fbody_label_train_root):
    os.symlink(os.path.abspath(fbody_ann_train_root),
            os.path.abspath(yolov5_fbody_label_train_root), target_is_directory=True)

if not os.path.exists(yolov5_fbody_label_val_root):
    os.symlink(os.path.abspath(fbody_ann_val_root),
            os.path.abspath(yolov5_fbody_label_val_root), target_is_directory=True)
print(colored("-- create data link finished", 'red'))