############## Measuring Flow Task ##############

it computes the roi of the last flow, and collects the all pixels that assigned to flow in that region of interest(roi)
this training set is tuned on 'video/v2.mp4'.

==============================================================================


#### for training ####

in root directoy of project:

git clone https://github.com/ultralytics/yolov5.git

cd yolov5

pip install -r requirements.txt
pip install roboflow

>> use video_to_imaage.py to create images from video and use it in roboflow.com to create dataset <<

>> download the dataset from roboflow using api_key <<


python /yolov5/train.py --batch 16 --epochs 30 --data /yolov5/<dataset_name>/data.yaml --weights yolov5s.pt --cache

If the error of the absence of the valid or Test folder occurs, edit the data.yaml by changing the valid and test directory address manually.

>> to test the model run these codes <<

!python /yolov5/detect.py --source <imgae_source> --weights <best.pt>
!python /yolov5/detect.py --source <video_source> --weights <best.pt>

>> the best.pt exists in yolov5/runs/train/exp<num>/weights/best.pt <<

>> create weights directory in root of prject and copy the best.pt in /weights folder <<


==============================================================================


#### to visualize the video realtime ####

>> in root directory of project <<

pip install -r requirements1.txt

>> in root of project run this command <<

python app.py <video source> <weight source> --save <output name>

 !note> ( --save <output name> )is optional

example>  
python app.py 'video/v2.mp4' 'weights/best.pt' --save output

open: http://127.0.0.1:5000/
==============================================================================
==============================================================================






