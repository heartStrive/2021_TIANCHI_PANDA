#！/bin/sh
if [ -d "user_data" ];then rm "user_data";fi
cd code/data_prepare
python format_data_test.py
cd ../train/detection/yolov5
if [ -f "test_0.txt" ];then rm "test_0.txt";fi
if [ -f "test_1.txt" ];then rm "test_1.txt";fi
if [ -f "test_2.txt" ];then rm "test_2.txt";fi
if [ -f "test_3.txt" ];then rm "test_3.txt";fi
python vid_inference.py --mode test --gpu_id 0 &
python vid_inference.py --mode test --gpu_id 1 &
python vid_inference.py --mode test --gpu_id 2 &
python vid_inference.py --mode test --gpu_id 3 &

while [ ! -f "test_0.txt" ] || [ ! -f "test_1.txt" ] || [ ! -f "test_2.txt" ] || [ ! -f "test_3.txt" ]
do
sleep 5
# echo 'waiting...'	
done

cd ../../../data_prepare
python prepare_trackdata_test.py
python prepare_clsreiddata_test.py
cd ../train/reid/fast-reid
python det_feat_extraction_test.py

cd ../../../merge_tracklets/association
python test_classifier.py
python association.py
python deal.py
zip -r ../../../results.zip results

#cd ../../../test
#python track_panda.py
#zip -r ../../results.zip results

