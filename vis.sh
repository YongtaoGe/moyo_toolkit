root='/home/geyongtao/data/moyo'

python scripts/ioi_vicon_frame_sync.py \
--img_folder $root/images/train/220926_yogi_body_hands_03596_Bow_Pose_or_Dhanurasana_-a// \
--cam_folder_first  $root/cameras/20220923/cameras_param.json \
--cam_folder_second $root/cameras/20220926/cameras_param.json \
--output_dir ../data/moyo_images_mocap_projected \
--frame_offset 1 \
--split train 

# --c3d_folder ../data/moyo/20220923_20220926_with_hands/vicon \