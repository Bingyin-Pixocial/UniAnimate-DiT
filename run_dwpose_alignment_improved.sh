# Improved DWpose alignment with angle-based retargeting
# Usage: bash run_dwpose_alignment_improved.sh
#
# Optional flags:
#   --fps               Output video FPS (default: 30)
#   --temporal_smoothing Enable One Euro Filter smoothing (default: off)
#   --smooth_min_cutoff One-Euro min cutoff (only if --temporal_smoothing)
#   --smooth_beta       One-Euro beta (only if --temporal_smoothing)
#   --max_bone_ratio    Max bone-length ratio clamp (default: 0 = auto)
#   --video_only        Only keep video, delete frame images
#   --edited_ref_name   Full-body edited ref image (enables partial-body mode)
#   --sam_checkpoint    Path to SAM model for visibility detection (optional)
#   --visibility_margin Margin around visible region (default: 0.05)
#   --draw_face         Draw face keypoints on output (requires wholebody model)

# --- Example 1: standard full-body mode (no temporal smoothing by default) ---
# python dwpose_alignment_improved.py \
#   --ref_name data/images/complex_motions/ref/3_ref_front.png \
#   --video_char_image data/images/complex_motions/video_char/hiphop69_char_half_front.png \
#   --source_video_paths data/videos/complex_motions/hiphop69_half.mp4 \
#   --saved_pose_dir data/saved_pose/dwpose_improved_3_front_hiphop69_half_face \
#   --fps 16 \
#   --max_bone_ratio 0 \
#   --draw_face \
#   --video_only


python dwpose_alignment_improved.py \
  --ref_name data/images/complex_motions/ref/3_ref_edited.jpg \
  --video_char_image data/images/complex_motions/video_char/hiphop69_char.png \
  --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
  --saved_pose_dir data/saved_pose/dwpose_improved_3_edited_hiphop69 \
  --fps 16 \
  --max_bone_ratio 0 \
  --draw_face \
  --video_only

# --- Example 2: partial-body mode (uncomment to use) ---
# When the ref image shows only part of the body (e.g., face/upper body),
# provide an edited full-body version via --edited_ref_name.
# #
python dwpose_alignment_improved.py \
  --ref_name data/images/complex_motions/ref/3_ref_front.png \
  --edited_ref_name data/images/complex_motions/ref/3_ref_edited.png \
  --video_char_image data/images/complex_motions/video_char/hiphop69_char.png \
  --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
  --saved_pose_dir data/saved_pose/dwpose_improved_3_front_hiphop69_face \
  --max_bone_ratio 0 \
  --visibility_margin 0.05 \
  --draw_face \
  --video_only


# # --- Example 3: partial-body mode with SAM mask (uncomment to use) ---
# # When the ref image shows only part of the body (e.g., face/upper body),
# # provide an edited full-body version via --edited_ref_name.
# #
python dwpose_alignment_improved.py \
  --ref_name data/images/complex_motions/ref/3_ref_front.png \
  --edited_ref_name data/images/complex_motions/ref/3_ref_edited.png \
  --video_char_image data/images/complex_motions/video_char/hiphop69_char.png \
  --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
  --saved_pose_dir data/saved_pose/dwpose_improved_3_front_hiphop69_sam_mask_face \
  --sam_checkpoint /picassox/intelligent-cpfs/pixocial/bingyin.zhao/code/tools/UniAnimate-DiT/checkpoints/sam_vit_h_4b8939.pth \
  --max_bone_ratio 0 \
  --visibility_margin 0.05 \
  --draw_face \
  --video_only
