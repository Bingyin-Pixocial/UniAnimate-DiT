# python dwpose_alignment_scaled.py  --ref_name /home/ubuntu/bingyin-Vol/code/projects/datasets/dance_p3/BadPoseCase/4_corrected.png \
#  --video_char_image data/pix/test1_base_img.png \
#  --source_video_paths data/pix/test1.mp4 \
#  --saved_pose_dir data/saved_pose/dwpose_test1_BadPoseCase4_corrected_ratio \
#  --ratio_factor 0.8


python dwpose_alignment.py --ref_name data/images/complex_motions/ref/1_ref.jpg \
 --video_char_image data/images/complex_motions/video_char/1_char.png \
 --source_video_paths data/videos/complex_motions/1.mp4 \
 --saved_pose_dir data/saved_pose/dwpose_1_1 \
 --fps 16


