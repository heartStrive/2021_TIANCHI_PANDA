#！/bin/sh
cd code/data_prepare
python format_data_train.py
python prepare_detdata.py
python prepare_trackdata_train.py

cd ../train/detection/yolov5
python train.py --weights pretrained_models/yolov5l.pt  --name fbody3 --data data/panda_fbody.yaml --batch-size 16 --img 2560 --epochs 24
if [ -f "train_0.txt" ];then rm "train_0.txt";fi
if [ -f "train_1.txt" ];then rm "train_1.txt";fi
if [ -f "train_2.txt" ];then rm "train_2.txt";fi
if [ -f "train_3.txt" ];then rm "train_3.txt";fi
python vid_inference.py --mode train --gpu_id 0 &
python vid_inference.py --mode train --gpu_id 1 &
python vid_inference.py --mode train --gpu_id 2 &
python vid_inference.py --mode train --gpu_id 3 &

while [ ! -f "train_0.txt" ] || [ ! -f "train_1.txt" ] || [ ! -f "train_2.txt" ] || [ ! -f "train_3.txt" ]
do
sleep 5
# echo 'waiting...'	
done
cd ../../../data_prepare
python prepare_clsreiddata_train.py
cd ../train/reid/fast-reid
python tools/train_net.py --config-file ./configs/PandaReID/bagtricks_R50.yaml MODEL.DEVICE "cuda:0"
python det_feat_extraction_train.py

cd ../../../merge_tracklets/association
python train_classifier.py
