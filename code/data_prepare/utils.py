import pandas as pd
import json
import os
import os.path as osp
from tqdm import tqdm
import cv2
import os 
from copy import deepcopy
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import json
import PIL
import shutil
from termcolor import colored
import torch
import torchvision


# detection related
def df2coco(df, tgtfile):
    attrDict = dict()
    attrDict["categories"] = [{"supercategory": "none", "id": 1, "name": 'full body'},]
    images = list()
    annotations = list()
    imageids = list()

    objid = 1
    imgid = 0

    for _, sub_df in df.groupby('img_nm'):
        image = dict()
        image['file_name'] = sub_df.img_nm.unique()[0]
        imageids.append(imgid)
        imgwidth = sub_df.imgW.unique()[0]
        imgheight = sub_df.imgH.unique()[0]
        image['height'] = int(imgheight)
        image['width'] = int(imgwidth)
        image['id'] = imgid
        images.append(image)

        for _, ann in sub_df.iterrows():
            annotation = dict()
            annotation["image_id"] = imgid
            annotation["ignore"] = 0
            annotation["iscrowd"] = 0
            x,y,w,h = int(ann.x1), int(ann.y1), int(ann.w), int(ann.h)
            annotation["bbox"] = [x, y, w, h]
            annotation["area"] = float(w * h)
            annotation["category_id"] = 1
            annotation["id"] = objid
            objid += 1
            annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
            annotations.append(annotation)
        imgid += 1
        
    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"     

    # print attrDict
    jsonString = json.dumps(attrDict, indent=2)
    print("save file at : ", tgtfile)
    with open(tgtfile, "w") as f:
        f.write(jsonString)

def round1_coco(round1_root, tgtfile):

    # load annotations
    round1_ann_root = osp.join(round1_root, 'image_annos')
    person_bbox_train_path = osp.join(round1_ann_root, "person_bbox_train.json")
    with open(person_bbox_train_path) as f:
        person_bbox_train = json.load(f)
    
    # process annotations
    gt_anns = []
    for img_nm, anns in person_bbox_train.items():
        imgH = anns['image size']['height']
        imgW = anns['image size']['width']
        objs = anns['objects list']
        for obj in objs:
            if obj['category'] == 'person':
                rect = obj['rects']['full body']
                x1 = rect['tl']['x'] * imgW
                y1 = rect['tl']['y'] * imgH
                x2 = rect['br']['x'] * imgW
                y2 = rect['br']['y'] * imgH
                gt_anns.append([img_nm, x1, y1, x2, y2, imgH, imgW])

    gt_df = pd.DataFrame(gt_anns, columns=['img_nm', 'x1', 'y1', 'x2', 'y2', 'imgH', 'imgW'])

    # drop invalid bounding box
    gt_df.loc[gt_df.y2 > gt_df.imgH, ['y2']] = gt_df.loc[gt_df.y2 > gt_df.imgH, ['imgH']] - 1
    gt_df.loc[gt_df.y1 < 0, ['y1']] = 0
    gt_df.loc[gt_df.x2 > gt_df.imgW, ['x2']] = gt_df.loc[gt_df.x2 > gt_df.imgW, ['imgW']] - 1
    gt_df.loc[gt_df.x1 < 0, ['x1']] = 0

    gt_df.loc[gt_df.y1 > gt_df.imgH, ['y1']] = gt_df.loc[gt_df.y2 > gt_df.imgH, ['imgH']] - 1
    gt_df.loc[gt_df.y2 < 0, ['y2']] = 0
    gt_df.loc[gt_df.x1 > gt_df.imgW, ['x1']] = gt_df.loc[gt_df.x2 > gt_df.imgW, ['imgW']] - 1
    gt_df.loc[gt_df.x2 < 0, ['x2']] = 0

    gt_df['h'] = gt_df.y2 - gt_df.y1
    gt_df['w'] = gt_df.x2 - gt_df.x1
    gt_df['area'] = gt_df.h * gt_df.w
    gt_df = gt_df[gt_df['area'] > 25]

    # convert to coco format
    df2coco(gt_df, tgtfile)

def round2_coco(round2_root, tgtfile):
    vid_root = os.path.join(round2_root, 'video_train')
    ann_root = os.path.join(round2_root, 'video_annos')
    vid_nms = [x for x in os.listdir(vid_root) if not x.endswith('zip')]

    gt_anns = []
    for vid_nm in tqdm(vid_nms, total=len(vid_nms)):
        seqinfo_path = os.path.join(ann_root, vid_nm, 'seqinfo.json')
        tracks_path = os.path.join(ann_root, vid_nm, 'tracks.json')

        # load annotations        
        with open(seqinfo_path) as f:
            seqinfo = json.load(f)
        with open(tracks_path) as f:
            tracks = json.load(f)
        
        # process annotations
        imgH, imgW = seqinfo['imHeight'], seqinfo['imWidth']
        for track in tracks:
            pid = track['track id']
            bboxes = track['frames']
            for bbox in bboxes:
                fid = bbox['frame id']
                img_nm = os.path.join(vid_nm, seqinfo['imUrls'][fid-1])
                rect = bbox['rect']
                x1 = rect['tl']['x'] * imgW
                y1 = rect['tl']['y'] * imgH
                x2 = rect['br']['x'] * imgW
                y2 = rect['br']['y'] * imgH
                occ = bbox['occlusion']
                gt_anns.append([img_nm, fid, pid, x1, y1, x2, y2, imgH, imgW,occ])

    # drop invalid bounding box
    gt_df = pd.DataFrame(gt_anns, columns=['img_nm', 'fid', 'pid', 'x1', 'y1', 'x2', 'y2', 'imgH', 'imgW', 'occ'])
    gt_df.occ.fillna('disappear', inplace=True)
    gt_df.occ.replace('', 'disappear', inplace=True)

    gt_df.loc[gt_df.y2 > gt_df.imgH, ['y2']] = gt_df.loc[gt_df.y2 > gt_df.imgH, ['imgH']] - 1
    gt_df.loc[gt_df.y1 < 0, ['y1']] = 0
    gt_df.loc[gt_df.x2 > gt_df.imgW, ['x2']] = gt_df.loc[gt_df.x2 > gt_df.imgW, ['imgW']] - 1
    gt_df.loc[gt_df.x1 < 0, ['x1']] = 0

    gt_df.loc[gt_df.y1 > gt_df.imgH, ['y1']] = gt_df.loc[gt_df.y1 > gt_df.imgH, ['imgH']] - 1
    gt_df.loc[gt_df.y2 < 0, ['y2']] = 0
    gt_df.loc[gt_df.x1 > gt_df.imgW, ['x1']] = gt_df.loc[gt_df.x1 > gt_df.imgW, ['imgW']] - 1
    gt_df.loc[gt_df.x2 < 0, ['x2']] = 0

    gt_df['h'] = gt_df.y2 - gt_df.y1
    gt_df['w'] = gt_df.x2 - gt_df.x1
    gt_df['area'] = gt_df.h * gt_df.w
    gt_df = gt_df[gt_df['area'] > 25]

    # convert to coco format
    df2coco(gt_df, tgtfile)

def create_folders(folder_paths):
    for folder_path in folder_paths:
        create_folder(folder_path)

def create_folder(folder_path, quiet=False):
    if not os.path.exists(folder_path):
        if not quiet:
            print(colored("++ create folder : {:}".format(folder_path), 'red'))
        os.makedirs(folder_path)
    else:
        if not quiet:
            print(colored("++ {:} exists".format(folder_path), 'green'))

def crop_img(roi_bbox, I):
    x1, y1, x2, y2 = roi_bbox
    return I[y1:y2, x1:x2, :]

def crop_annos(roi_bbox, anns, thr = 0.7):
    x1, y1, x2, y2 = roi_bbox
    xr, yr, wr, hr = x1, y1, x2 - x1, y2 - y1
    roi_rect = Polygon([(xr,yr),(xr+wr,yr),(xr+wr, yr+hr),(xr,yr+hr)])
    
    # select bbox
    sub_anns = []
    for ann in anns:
        x,y,w,h=ann['bbox']
        obj_rect = Polygon([(x,y),(x+w,y),(x+w, y+h),(x,y+h)])
        ratio = obj_rect.intersection(roi_rect).area / obj_rect.area
        if ratio > thr:
            sub_anns.append(ann)
    
    # move bbox
    local_anns = []
    for ann in sub_anns:
        x,y,w,h=ann['bbox']

        if ann['category_id'] != 2:
            x1, y1, x2, y2 = max(0, x-xr), max(0, y-yr), min(wr, x+w-xr), min(hr, y+h-yr)
            
            assert x1 >= 0
            assert y1 >= 0
            assert (x2 - x1) > 0
            assert (y2 - y1) > 0
            assert (x2 <= wr)
            assert (y2 <= hr)
        else:
            x1, y1, x2, y2 = x-xr, y-yr, x-xr+w, y-yr+h
        
        x,y,w,h = x1, y1, x2 - x1, y2 - y1
        ann['bbox'] = [x,y,w,h]
        ann["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
        local_anns.append(ann)
    
    return local_anns

def resize_img(scale, img, I):
    img_w, img_h = img[0]['width'], img[0]['height']
    img_w_scale, img_h_scale = np.rint(img_w * scale).astype(int), np.rint(img_h * scale).astype(int)
    
    return cv2.resize(I, (img_w_scale, img_h_scale))

def resize_annos(scale, img, anns, small_objsize = 5):
    
    img_w, img_h = img[0]['width'], img[0]['height']
    img_w_scale, img_h_scale = np.rint(img_w * scale).astype(int), np.rint(img_h * scale).astype(int)
    
    sub_anns = []
    for ann in anns:
        x,y,w,h=ann['bbox']
        
        x1, y1 = x/img_w*img_w_scale, y/img_h*img_h_scale, 
        x2, y2 = (x+w)/img_w*img_w_scale, (y+h)/img_h*img_h_scale
        x1 = int(np.rint(x1)) 
        y1 = int(np.rint(y1)) 
        x2 = int(np.rint(x2)) 
        y2 = int(np.rint(y2))
        
        if (y2-y1)*(x2-x1) > small_objsize**2:
            
            assert x1 >= 0
            assert y1 >= 0
            assert (x2 - x1) > 0
            assert (y2 - y1) > 0
            assert (x2 <= img_w_scale)
            assert (y2 <= img_h_scale)
            
            xs, ys, ws, hs = x1, y1, x2-x1, y2-y1               
 
            ann["bbox"] = [xs, ys, ws, hs]
            ann["area"] = float(ws * hs)
            ann["segmentation"] = [[xs, ys, xs, (ys + hs), (xs + ws), (ys + hs), (xs + ws), ys]]
            sub_anns.append(ann)   
    
    return sub_anns

def subimg_coord(imgwidth, imgheight, subwidth, subheight, gap):
    sub_coords = []
    gap_w, gap_h = gap
    slidewidth = subwidth - gap_w
    slideheight = subheight - gap_h

    left, up = 0, 0
    while left < imgwidth:
        if left + subwidth >= imgwidth:
            left = max(imgwidth - subwidth, 0)
        up = 0
        while up < imgheight:
            if up + subheight >= imgheight:
                up = max(imgheight - subheight, 0)
            right = min(left + subwidth, imgwidth - 1)
            down = min(up + subheight, imgheight - 1)
            coordinates = left, up, right, down
            sub_coords.append(coordinates)

            if up + subheight >= imgheight:
                break
            else:
                up = up + slideheight
        if left + subwidth >= imgwidth:
            break
        else:
            left = left + slidewidth
    
    return sub_coords

def fetch_infos(coco, imgId):
    catIds = list(coco.cats.keys())
    img = coco.loadImgs(imgId)
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    return img, anns

def tiles_annos(coco, imgRoot, scale_lst, subwidth, subheight, gap, outImgRoot, tgtfile):
    # init coco format annotations
    attrDict = dict()
    attrDict["categories"] = [
        {"supercategory": "none", "id": 1, "name": 'full body'}]
    images = list()
    annotations = list()

    objid = 1
    imgid = 1

    for imgId, img_info in tqdm(coco.imgs.items(), total=len(coco.imgs)):
        # fetch original annotations
        img_name = img_info['file_name']
        img, anns = fetch_infos(coco, imgId)
        annIds = coco.getAnnIds(imgIds=imgId, catIds=[1], iscrowd=None)
        anns = coco.loadAnns(annIds)    

        # load image
        # print(os.path.join(imgRoot, img_info['file_name']))
        I = cv2.imread(os.path.join(imgRoot, img_info['file_name']))
        # crop image tiles at different scales
        for scale in scale_lst:
            thr = 0.5
            
            # resize image & annotations
            rI = resize_img(scale, img, deepcopy(I))
            img_h, img_w, _ = rI.shape
            rAnnos = resize_annos(scale, img, deepcopy(anns), small_objsize = 5)

            # generate sub coords at scale
            sub_coords = subimg_coord(img_w, img_h, subwidth, subheight, gap)
            outbasename = img_name.replace('/', '_').split('.')[0] + '___' + str(scale) + '__'
            
            for sub_coord in sub_coords:
                # crop image & annotations
                left, up, right, down = sub_coord
                subimgname = outbasename + str(left) + '__' + str(up) + "__"  + str(right) + "__" + str(down) + ".jpg"
                cI = crop_img(sub_coord, rI)
                cAnnos = crop_annos(sub_coord, deepcopy(rAnnos), thr=thr)

                if len(cAnnos) > 0:
                    # save subimages
                    cv2.imwrite(os.path.join(outImgRoot, subimgname), cI)

                    image = dict()
                    image['file_name'] = subimgname
                    # print(subimgname, len(cAnnos))
                    imgwidth = cI.shape[1]
                    imgheight = cI.shape[0]
                    image['height'] = imgheight
                    image['width'] = imgwidth
                    image['id'] = imgid
                    images.append(image)

                    # format sub annotations                    

                    for ann in cAnnos:
                        annotation = dict()
                        x, y, w, h = ann['bbox']
                        cateId = ann['category_id']
                        annotation["image_id"] = imgid
                        annotation["ignore"] = 0
                        annotation["iscrowd"] = 0
                        annotation["bbox"] = [x, y, w, h]
                        annotation["area"] = float(w * h)
                        annotation["category_id"] = cateId
                        annotation["id"] = objid
                        annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                        annotations.append(annotation)
                        objid += 1
                    imgid += 1             

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict, indent=2)
    with open(tgtfile, "w") as f:
        f.write(jsonString)

def yolo_cat_anns(coco, imgRoot, outRoot):
    yolo_anns = []
    print("filter annotations ")
    for idx, ann in tqdm(coco.anns.items(), total=len(coco.anns)):
        img_id = ann['image_id']
        img_info = coco.imgs[img_id]
        img = PIL.Image.open(os.path.join(imgRoot, img_info['file_name']))
        img_w, img_h = img.size
        x,y,w,h = ann['bbox']
        xc, yc = x+w/2, y+h/2
        xcn, ycn, wn, hn = xc/img_w, yc/img_h, w/img_w, h/img_h
        cate_id = 0
        yolo_anns.append([xcn, ycn, wn, hn, img_w, img_h, img_id, cate_id])
    
    yolo_ann_df = pd.DataFrame(yolo_anns, columns=['xc', 'yc', 'w', 'h', 'img_w', 'img_h', 'img_id', 'cate_id'])
    dfs = [group for _, group in yolo_ann_df.groupby('img_id')]
    print("saving annotations ")
    for df in tqdm(dfs, total=len(dfs)):
        img_info = coco.imgs[df['img_id'].iloc[0]]
        df_txt = df[['cate_id','xc','yc','w','h']]
        ann_np = df_txt.values
        ann_nm = img_info['file_name'].replace('jpg', 'txt')
        np.savetxt(os.path.join(outRoot, ann_nm), ann_np, fmt='%d %f %f %f %f')

def trainval_split(root, val_root, train_root, pattern="_05__"):
    for file_nm in os.listdir(root):
        src = os.path.join(root, file_nm)
        if pattern in file_nm:
            dst = val_root
        else:
            dst = train_root
        
        if not os.path.isfile(os.path.join(dst, file_nm)):
            shutil.copy(src, dst)

def coco_subannos(coco, sets, steps):
    img_dict = {}
    for imgId, img_info in coco.imgs.items():
        img_nm = img_info['file_name']
        set_id = int(img_nm.split('/')[0].split('_')[0])
        frame_id = int(img_nm.split('/')[1].split('_')[-1][:-4])

        if (set_id in sets) & (frame_id % steps == 0):
            img_dict[imgId] = img_info
    
    coco.imgs = img_dict
    return coco


# reid cls related
def iou_mat(det_df, gt_df):
    def df2tensor(df):
        bbox = df[['x','y','w','h']]
        tensor = torch.tensor(bbox.values)
        return torchvision.ops.box_convert(tensor, 'xywh', 'xyxy')
    
    det_tensor = df2tensor(det_df)
    gt_tensor = df2tensor(gt_df)
    
    return torchvision.ops.box_iou(det_tensor, gt_tensor)

def reid_cls_annotation_process(det_df, gt_df, cls_vid_txt_root, reid_vid_txt_root):
    cls_lst = []
    reid_lst = []
    num_frame = len(det_df.fid.unique())
    for fid in tqdm(det_df.fid.unique(), total=num_frame):
        sub_det = det_df[det_df.fid == fid].copy()
        sub_gt = gt_df[gt_df.fid == fid].copy()
        # calculate iou matrix
        imat = iou_mat(sub_det, sub_gt)   

        # generate reid bbox by filtering iou(keep 0.5 above) between gt and detection
        iou_values, det_ids = torch.max(imat, axis=0)
        sub_gt.insert(sub_gt.shape[1], 'iou', iou_values.numpy())
        sub_gt.insert(sub_gt.shape[1], 'did', sub_det.iloc[det_ids].did.values)        

        # assign detection class for keep or drop by iou with gt
        iou_values, gt_ids = torch.max(imat, axis=1)
        sub_det.insert(sub_det.shape[1], 'iou', iou_values.numpy())
        sub_det.insert(sub_det.shape[1], 'gid', sub_gt.iloc[gt_ids].pid.values)
        cls = sub_det.groupby('gid', group_keys=False).apply(lambda x : x.iou == x.iou.max()).sort_index().values
        sub_det.insert(sub_det.shape[1], 'cls', cls)    
        
        sub_gt = sub_gt[sub_gt.iou > 0.5][['fid', 'pid', 'x', 'y', 'w', 'h', 'iou', 'did']]
        
        reid_lst.append(sub_gt)
        cls_lst.append(sub_det)
    
    # save annotations
    cls_out_path = os.path.join(cls_vid_txt_root, 'cls.txt')
    print(colored("++ save cls annos at : {:}".format(cls_out_path), 'red'))
    pd.concat(cls_lst).to_csv(cls_out_path, index=False)

    reid_out_path = os.path.join(reid_vid_txt_root, 'reid.txt')
    print(colored("++ save reid annos at : {:}".format(reid_out_path), 'red'))
    pd.concat(reid_lst).to_csv(reid_out_path, index=False)    