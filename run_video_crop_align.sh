# Video crop align: crop full-body video to match partial-body reference framing
# Usage: run from UniAnimate-DiT directory, e.g. bash run_video_crop_align.sh
#
# Optional flags:
#   --output_size             Output resolution WxH (default: same as driven video)
#   --fps                     Output video FPS (default: auto-detect from driven video)
#   --source_pose_video_paths Pose video (same length as driven); same crop -> second output
#   --output_pose_video       Output path for cropped pose video (required if above set)
#   --sam_checkpoint          Path to SAM model for visibility region (optional)
#   --visibility_margin       Margin around visible region (default: 0.05)
#   --save_crops              Save crop box to JSON

# --- Example 1: basic usage (reference = partial body, video = full body) ---
# python video_crop_align.py \
#   --ref_name data/images/complex_motions/video_char/hiphop69_char_half_front.png \
#   --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
#   --output_video data/cropped_videos/crop_align_hiphop69_half_front_hiphop69/cropped_video.mp4 \
#   --fps 16 \
#   --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth

# --- Example 2: with SAM and save_crops (uncomment to use) ---
# python video_crop_align.py \
#   --ref_name data/images/complex_motions/ref/3_ref.jpg \
#   --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
#   --output_video data/output/crop_align_3_hiphop69.mp4 \
#   --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
#   --visibility_margin 0.05 \
#   --save_crops

# --- Example 3: driven video + pose video (two outputs, same crop) ---
python video_crop_align.py \
  --ref_name data/images/complex_motions/ref/3_ref_front.png  \
  --source_video_paths data/videos/complex_motions/hiphop69.mp4 \
  --output_video data/cropped_videos/crop_align_3_front_hiphop69/cropped_video.mp4 \
  --source_pose_video_paths /picassox/intelligent-cpfs/pixocial/bingyin.zhao/code/tools/UniAnimate-DiT/data/saved_pose/dwpose_improved_3_edited_hiphop69/pose_sequence.mp4 \
  --output_pose_video data/cropped_videos/crop_align_3_front_hiphop69/cropped_retargeted_pose_video.mp4 \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --visibility_margin 0.05 \
