import cv2
import mmcv
import numpy as np
import torch
from utils.general import non_max_suppression
from models.experimental import attempt_load
from ensemble_boxes import weighted_boxes_fusion

#----------parameter for read image funtion-----------------------------
HEIGHT = 5120 # Sliding Window Height
WIDTH = 5120 # Sliding Window Width
SCORE_THRES = 0.35
OVERLAP = 0.25
print("OVERLAP : ", OVERLAP)
BATCH_SIZE=4

SCALES=[1.0, 0.5,0.25,0.125,0.0625]

# 初始化模型
def init_yolo_detector(path,clses, device):
    model = attempt_load(path, map_location=device) # 加载模型权重
    model.half() # fps16
    model.CLASSES = clses #模型预测的类别名称
    model.to(device) # 放到gpu上
    model.eval() # 不启用 BatchNormalization 和 Dropout
    return model
    
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class LoadImages:  
    def __init__(self, imgs, img_size=640, stride=32):
        self.images =imgs
        self.img_size = img_size
        self.stride = stride
        self.nf =  len(self.images)

    def __len__(self):
        return self.nf

    def __getitem__(self,index):
        img = self.images[index]
        # Padded resize
        # img = letterbox(img, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img

def inference_yolo_detector(models, imgs,batch_size, device):
    num=len(models)
    # 四个模型的结果单独放在不同的列表
    results=[ [] for i in range(num) ]
    conf_thres=0.001
    iou_thres=0.65

    datas = LoadImages(imgs,img_size=5120)
    dataloader=torch.utils.data.DataLoader(datas,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)
    dataloader=iter(dataloader)

    for data in dataloader:
        data=torch.tensor(data,dtype=torch.float16).to(device, non_blocking=True)
        data/=255.0 # 归一化
        # forward the model
        with torch.no_grad():
            for i,model in enumerate(models):
                pred = model(data)[0]
                pred = non_max_suppression(pred, conf_thres,iou_thres, agnostic=True)
                results[i].extend(pred)
    results = [ [item.cpu().numpy() for item in result] for result in results ]
    return results

class ImgData():
    def __init__(self, raw_height, raw_width):
        self.imgs=[]
        self.ini_xy=[]
        self.scale_ids=[]
        self.img_scales=[]

        self.image_id = 0
        self.image_path = ""
        self.raw_height = raw_height
        self.raw_width = raw_width

    def put_data(self,sub_img,ini_x,ini_y,idx,scale):
        self.imgs.append(sub_img)
        self.ini_xy.append((ini_x,ini_y))
        self.scale_ids.append(idx)
        self.img_scales.append(scale)

def predict(detector_models,data_img,device,score_thres=SCORE_THRES):
    ini_xy = data_img.ini_xy
    scale_ids = data_img.scale_ids
    img_scales = data_img.img_scales

    bboxes_one=[[] for i in range(len(SCALES))]
    labels_one=[[] for i in range(len(SCALES))]
    scores_one=[[] for i in range(len(SCALES))]
    all_results= inference_yolo_detector(detector_models,data_img.imgs, BATCH_SIZE, device=device)
    for cate,results in enumerate(all_results):
        assert(len(results)==len(ini_xy) and len(ini_xy)==len(scale_ids) and len(scale_ids)== len(img_scales))
        cate_id=str(cate+1)
        for xy,result,sca_id,scale in zip(ini_xy, results,scale_ids,img_scales):
            if len(result)==0:
                continue
            ini_x,ini_y=xy
            src_width=scale*data_img.raw_width
            src_height=scale*data_img.raw_height
            # process result
            bboxes = np.vstack(result)
            for i, bbox in enumerate(bboxes):
                score = bbox[4]
                if score > score_thres:
                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = bbox[2]
                    y2 = bbox[3]
                    if ini_x > 0 and x2 < WIDTH*OVERLAP/2:
                        continue
                    if ini_y > 0 and y2 < HEIGHT*OVERLAP/2:
                        continue
                    if ini_x+WIDTH<src_width and x1>WIDTH-WIDTH*OVERLAP/2:
                        continue
                    if ini_y+HEIGHT<src_height and y1>HEIGHT-HEIGHT*OVERLAP/2:
                        continue
                    point=[(x1 + ini_x) / scale,
                           (y1 + ini_y) / scale,
                           (x2 + ini_x) / scale,
                           (y2 + ini_y) / scale]
                    if scale>0.6:
                        w=point[2]-point[0]
                        h=point[3]-point[1]
                        if w>1024 or h>1024:
                            continue

                    bboxes_one[sca_id].append(point)
                    labels_one[sca_id].append(int(cate_id))
                    scores_one[sca_id].append(score)
    return bboxes_one,labels_one,scores_one

def fine_det_full(img_path):
    raw_img = mmcv.imread(img_path)
    raw_height, raw_width = raw_img.shape[:2]
    # detect on images
    data_img = ImgData(raw_height, raw_width)

    x_step = int(WIDTH * (1 - OVERLAP))
    y_step = int(HEIGHT * (1 - OVERLAP))
    for idx,scale in enumerate(SCALES):
        scale=max(scale,max(1.0*WIDTH/raw_width,1.0*HEIGHT/raw_height))
        src_img = mmcv.imresize(
            raw_img, (int(raw_width * scale), int(raw_height * scale)),
            interpolation='area')
        src_height, src_width = src_img.shape[:2]
        # sub image generate
        ini_y = 0
        while ini_y < src_height:
            ini_x = 0
            while ini_x < src_width:
                sub_img = src_img[ini_y:ini_y + HEIGHT, ini_x:ini_x + WIDTH]

                data_img.put_data(sub_img,ini_x,ini_y,idx,scale)

                if ini_x == src_width - WIDTH  or src_width <= WIDTH:
                    break
                ini_x += x_step
                if ini_x + WIDTH > src_width:
                    ini_x = src_width - WIDTH

            if ini_y == src_height - HEIGHT  or src_height <= HEIGHT:
                break
            ini_y += y_step
            if ini_y + HEIGHT > src_height:
                ini_y = src_height - HEIGHT

    return data_img

def nms_after_det(bboxes_list,labels_list,scores_list):
    iou_thr=0.65
    det_bboxes,det_scores, det_labels = weighted_boxes_fusion(bboxes_list,scores_list,labels_list,
    weights=None,iou_thr=iou_thr,skip_box_thr=SCORE_THRES)
    return det_bboxes, det_scores, det_labels.astype(np.int32)

def show_after_nms(
        img_path,
        det_bboxes,
        det_scores,
        det_labels,
        cls_nm,
        save_dir,
        show_scale=0.05,
        score_thres=SCORE_THRES):
    # show the detection result and save it
    # load full image
    full_img = mmcv.imread(img_path)
    full_height, full_width = full_img.shape[:2]
    full_img = mmcv.imresize(
        full_img,
        (int(full_width * show_scale), int(full_height * show_scale)))

    det_scores=det_scores[np.newaxis,:]
    det_bboxes=np.concatenate((det_bboxes,det_scores.T),axis=1)
    
    # transfer scale of detection results
    det_bboxes[:, 0:4] *= show_scale

    # save result after NMS
    mmcv.imshow_det_bboxes(
        full_img.copy(),
        det_bboxes,
        det_labels,
        class_names=cls_nm,
        score_thr=score_thres,
        out_file=save_dir,
        show=False,
        wait_time=0,
    )
    return None

def simple_infer(img_path, model, device):
    data_img = fine_det_full(img_path)
    raw_height = data_img.raw_height
    raw_width = data_img.raw_width
    bboxes_list, labels_list,scores_list = predict([model], data_img, device)
    for k in range(len(bboxes_list)):
        bboxes_list[k]=np.array(bboxes_list[k])
        bboxes_list[k][:,0]/=raw_width
        bboxes_list[k][:,1]/=raw_height
        bboxes_list[k][:,2]/=raw_width
        bboxes_list[k][:,3]/=raw_height
        bboxes_list[k]=bboxes_list[k].tolist()
    det_bboxes, det_scores, det_labels = nms_after_det(bboxes_list, labels_list, scores_list)
    det_bboxes[:,0]*=raw_width
    det_bboxes[:,1]*=raw_height
    det_bboxes[:,2]*=raw_width
    det_bboxes[:,3]*=raw_height
    yolov5_det = np.c_[det_bboxes, det_scores]

    return yolov5_det


