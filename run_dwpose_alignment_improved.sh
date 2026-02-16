# Improved DWpose alignment with angle-based retargeting
# Usage: bash run_dwpose_alignment_improved.sh
#
# New optional flags vs original:
#   --fps               Video FPS for smoothing (default: 30)
#   --smooth_min_cutoff One-Euro min cutoff (default: 1.7)
#   --smooth_beta       One-Euro beta (default: 0.3)

python dwpose_alignment_improved.py \
  --ref_name data/images/complex_motions/ref/1_ref.jpg \
  --video_char_image data/images/complex_motions/video_char/1_char.png \
  --source_video_paths data/videos/complex_motions/1.mp4 \
  --saved_pose_dir data/saved_pose/dwpose_improved_1_1 \
  --fps 16 \
  --smooth_min_cutoff 1.7 \
  --smooth_beta 0.3
