# Improved DWpose alignment with angle-based retargeting
# Usage: bash run_dwpose_alignment_improved.sh
#
# New optional flags vs original:
#   --fps               Video FPS for smoothing (default: 30)
#   --smooth_min_cutoff One-Euro min cutoff (default: 1.7)
#   --smooth_beta       One-Euro beta (default: 0.3)

python dwpose_alignment_improved.py \
  --ref_name /home/ubuntu/bingyin-Vol/code/projects/datasets/dance_p3/BadPoseCase/4_corrected.png \
  --video_char_image data/pix/test1_base_img.png \
  --source_video_paths data/pix/test1.mp4 \
  --saved_pose_dir data/saved_pose/dwpose_improved_test1_BadPoseCase4 \
  --fps 30.0 \
  --smooth_min_cutoff 1.7 \
  --smooth_beta 0.3
