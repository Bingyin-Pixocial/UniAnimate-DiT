# python dwpose_alignment_scaled.py  --ref_name /home/ubuntu/bingyin-Vol/code/projects/datasets/dance_p3/BadPoseCase/4_corrected.png \
#  --video_char_image data/pix/test1_base_img.png \
#  --source_video_paths data/pix/test1.mp4 \
#  --saved_pose_dir data/saved_pose/dwpose_test1_BadPoseCase4_corrected_ratio \
#  --ratio_factor 0.8


# python dwpose_alignment.py --ref_name data/images/complex_motions/ref/hiphop7_ref.png \
#  --video_char_image data/images/complex_motions/video_char/hiphop69_char.png \
#  --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
#  --saved_pose_dir data/saved_pose/dwpose_hiphop7_hiphop69 \
#  --fps 16 \
#  --video_only


 python dwpose_alignment.py --ref_name data/images/complex_motions/ref/3_ref_edited.png \
 --video_char_image data/images/complex_motions/video_char/hiphop69_char.png \
 --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
 --saved_pose_dir data/saved_pose/dwpose_3_edited_hiphop69 \
 --fps 16 \
 --video_only


