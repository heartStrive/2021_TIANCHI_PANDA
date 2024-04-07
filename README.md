# 训练和推理
1. 训练的pipeline：train.sh
2. 推理的pipeline：test.sh

# 检测算法介绍
1. 使用[YOLOv5l](https://github.com/ultralytics/yolov5)作为目标检测模型，按照"head","visible body","full body","vehicles"四类分别训练四个检测模型。
2. 训练过程中将原始图像分别以[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]尺度进行缩小，再按照640x640大小，x方向overlap=100，y方向overlap=200进行裁剪。
3. 每类模型训练24轮，训练过程中使用[YOLOv5l](https://github.com/ultralytics/yolov5)默认学习率，最终选择效果最好的模型。
4. 测试过程中将原始图像分别以[1.0, 0.5, 0.25, 0.125, 0.0625]尺度进行缩小，再按照2560x2560大小，overlap=0.2进行切分。
5. 不同尺度的检测结果利用segmentation-and-fusion处理。
6. 将各尺度检测结果转换到原图像坐标系中，使用[weighted nms](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)进行结果融合。

# 跟踪算法介绍
1. 训练reid模型
2. 构建图与成本，使用费用流算法进行数据关联

# 准备数据

1. 首先运行code/data_prepare/format_data.py将数据集按照如下结构重新排布

```
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
        |--detection
        |--reid
        |--tracking
```

# 准备目标检测数据

2. 运行code/data_prepare/prepare_detdata.py将round1、round2数据处理成可以用于yolov5训练的数据，其中01_University_Canteen作为测试集，round2中数据以按照30帧采样
```
|--detection
    |--annotations
        |--panda_coco.json      # 全部4类coco格式的标签
        |--tiles_trainval.json  # full body 1类coco格式的标签
    |--image_tiles
        |--XXX.jpg 
        ...
    |--image_tiles_train
        |--XXX.jpg 
        ...
    |--image_tiles_val
        |--XXX.jpg 
        ...
    |--yolo_annotations
        |--fbody
            |--XXX.txt
        |--fbody_train
            |--XXX.txt
        |--fbody_val
            |--XXX.txt
    |--yolov5_data
        |--fbody
            |--images
                |--train (link to image_tiles_train)
                |--val   (link to image_tiles_val)
            |--labels
                |--train (link to fbody_train)
                |--val   (link to fbody_val)
```

# 训练yolov5l模型

3. 运行code/train/detection/yolov5/train.py训练yolov5l模型

# 在跟踪数据上检测fully body

4. 运行code/train/detection/yolov5/vid_inference.py

# 准备跟踪数据

5. 运行code/data_prepare/prepare_trackdata.py 将round2数据转化成MOT2015格式

```
|--tracking
    |--MOT-PANDA
        |--01_University_Canteen
            |--det
                |--yolov5-det.txt
            |--gt
                |--gt.txt
            |--img1 (link to video train frames)
                |-- XXX.jpg
                ...
                |-- XXX.jpg
        |--02_OCT_Habour
            |--det
                |--yolov5-det.txt
            |--gt
                |--gt.txt
            |--img1 (link to video train frames)
                |-- XXX.jpg
                ...
                |-- XXX.jpg
        ...
        |--10_Huaqiangbei
            |--det
                |--yolov5-det.txt
            |--gt
                |--gt.txt
            |--img1 (link to video train frames)
                |-- XXX.jpg
                ...
                |-- XXX.jpg
```

# 准备reid以及cls数据

```
|--reid
    |--ReID-PANDA
        |--01_University_Canteen
            |--img
                |--000001
                    |--pid_fid_x1_y1_x2_y2.jpg
                    ...
                |--000002
                    |--pid_fid_x1_y1_x2_y2.jpg
                    ...
            |--txt
                |--reid.txt
        ...
```

6. 运行code/data_prepare/prepare_clsreiddata.py

# 训练reid模型
pip install yacs
pip install scikit-learn
pip install faiss-cpu
pip install gdown
cd fastreid/evaluation/rank_cylib; make all

7. 运行train/reid/fast-reid/train_net.py进行reid模型训练

# detection feature extraction

8. 运行 code/train/reid/fast-reid/det_feat_extraction.py

# 数据关联
9. 运行merge_tracklets/association/train_classifier.py进行数据关联
