# Improved DWpose alignment with angle-based retargeting
# Usage: bash run_dwpose_alignment_improved.sh
#
# Optional flags:
#   --fps               Video FPS for smoothing (default: 30)
#   --smooth_min_cutoff One-Euro min cutoff (default: 1.7)
#   --smooth_beta       One-Euro beta (default: 0.3)
#   --max_bone_ratio    Max bone-length ratio clamp (default: 0 = auto)
#   --video_only        Only keep video, delete frame images
#   --edited_ref_name   Full-body edited ref image (enables partial-body mode)
#   --sam_checkpoint    Path to SAM model for visibility detection (optional)
#   --visibility_margin Margin around visible region (default: 0.05)

# --- Example 1: standard full-body mode ---
# python dwpose_alignment_improved.py \
#   --ref_name data/images/complex_motions/ref/3_ref.jpg \
#   --video_char_image data/images/complex_motions/video_char/1_char.png \
#   --source_video_paths data/videos/complex_motions/1.mp4 \
#   --saved_pose_dir data/saved_pose/dwpose_improved_3_1_responsiveness \
#   --fps 16 \
#   --smooth_min_cutoff 2 \
#   --smooth_beta 0.5 \
#   --max_bone_ratio 0 \
#   --video_only

# --- Example 2: partial-body mode (uncomment to use) ---
# When the ref image shows only part of the body (e.g., face/upper body),
# provide an edited full-body version via --edited_ref_name.
#
python dwpose_alignment_improved.py \
  --ref_name data/images/complex_motions/ref/101_ref.jpg \
  --edited_ref_name data/images/complex_motions/ref/101_ref_edited.png \
  --video_char_image data/images/complex_motions/video_char/39_char.png \
  --source_video_paths data/videos/complex_motions/39.mp4 \
  --saved_pose_dir data/saved_pose/dwpose_improved_101_39_responsiveness \
  --fps 30 \
  --smooth_min_cutoff 2 \
  --smooth_beta 0.5 \
  --max_bone_ratio 0 \
  --visibility_margin 0.05 \
  --video_only



  # --- Example 3: partial-body mode with SAM mask (uncomment to use) ---
# When the ref image shows only part of the body (e.g., face/upper body),
# provide an edited full-body version via --edited_ref_name.
#
# python dwpose_alignment_improved.py \
#   --ref_name data/images/complex_motions/ref/3_ref.jpg \
#   --edited_ref_name data/images/complex_motions/ref/3_ref_edited.png \
#   --video_char_image data/images/complex_motions/video_char/39_char.png \
#   --source_video_paths data/videos/complex_motions/39.mp4 \
#   --saved_pose_dir data/saved_pose/dwpose_improved_3_39_sam_mask \
#   --sam_checkpoint /picassox/intelligent-cpfs/pixocial/bingyin.zhao/code/tools/UniAnimate-DiT/checkpoints/sam_vit_h_4b8939.pth \
#   --fps 30 \
#   --smooth_min_cutoff 2 \
#   --smooth_beta 0.5 \
#   --max_bone_ratio 0 \
#   --visibility_margin 0.05 \
#   --video_only