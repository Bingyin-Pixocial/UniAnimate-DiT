# Video crop align: crop full-body video to match partial-body reference framing
# Uses DWPose keypoints (and optional SAM) to align per-frame crop with reference.
# Reuses helpers from dwpose_alignment_improved.

import os
import sys
import math
import argparse
import logging
import json
import tempfile
import shutil

import cv2
import numpy as np

# Reuse from improved alignment script
from dwpose_alignment_improved import (
    is_valid_kp,
    load_sam_predictor,
    compute_visibility_region_sam,
    compute_visibility_region_keypoints,
    DWposeDetector,
)
from dwpose_alignment_improved import _SAM_AVAILABLE

# Body keypoint indices (same as in improved script)
# 0=nose, 1=neck, 2=Rshoulder, 3=Relbow, 4=Rwrist,
# 5=Lshoulder, 6=Lelbow, 7=Lwrist, 8=Rhip, 9=Rknee, ...
STABLE_PAIRS_FOR_SCALE = [(1, 0), (2, 5)]  # neck-nose, shoulders
ANCHOR_JOINT = 1  # neck preferred for center


def get_logger(name="video_crop_align"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _visible_keypoint_indices(ref_cand, visible_region):
    """Indices of keypoints that are valid and inside the visibility region (norm coords)."""
    y_min, y_max, x_min, x_max = visible_region
    indices = []
    for j in range(min(20, ref_cand.shape[0])):
        if not is_valid_kp(ref_cand[j]):
            continue
        x, y = ref_cand[j][0], ref_cand[j][1]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            indices.append(j)
    return indices if indices else [j for j in range(min(20, ref_cand.shape[0])) if is_valid_kp(ref_cand[j])]


def compute_crop_from_keypoints(
    ref_cand_norm,
    vid_cand_norm,
    visible_indices,
    W_ref,
    H_ref,
    W_vid,
    H_vid,
):
    """
    Compute crop (cx, cy, w, h) in video pixel coords so that visible keypoints
    align with reference framing. Ref/vid_cand are normalized [0,1].
    """
    R = W_ref / float(H_ref)  # aspect
    # Ref pixel coords
    ref_px = np.array([(ref_cand_norm[j][0] * W_ref, ref_cand_norm[j][1] * H_ref) for j in visible_indices])
    vid_px = np.array(
        [(vid_cand_norm[j][0] * W_vid, vid_cand_norm[j][1] * H_vid) for j in visible_indices]
    )
    if len(ref_px) < 2:
        return None

    # Scale w from distance ratio (average over stable pairs present)
    w_candidates = []
    for i, j in STABLE_PAIRS_FOR_SCALE:
        if i not in visible_indices or j not in visible_indices:
            continue
        idx_i, idx_j = visible_indices.index(i), visible_indices.index(j)
        d_ref = _distance(ref_px[idx_i], ref_px[idx_j])
        d_vid = _distance(vid_px[idx_i], vid_px[idx_j])
        if d_ref < 1e-6:
            continue
        w_candidates.append(d_vid * W_ref / d_ref)
    if not w_candidates:
        # fallback: any pair
        for ii in range(len(visible_indices)):
            for jj in range(ii + 1, len(visible_indices)):
                d_ref = _distance(ref_px[ii], ref_px[jj])
                d_vid = _distance(vid_px[ii], vid_px[jj])
                if d_ref >= 1e-6:
                    w_candidates.append(d_vid * W_ref / d_ref)
    if not w_candidates:
        return None
    w = float(np.median(w_candidates))
    h = w / R

    # Center: mean of cx, cy from each visible keypoint
    cx_list = [vid_px[i][0] + w * (0.5 - ref_px[i][0] / W_ref) for i in range(len(ref_px))]
    cy_list = [vid_px[i][1] + h * (0.5 - ref_px[i][1] / H_ref) for i in range(len(ref_px))]
    cx = float(np.mean(cx_list))
    cy = float(np.mean(cy_list))

    # Shrink to fit frame while keeping aspect
    w_max_x = 2 * min(cx, W_vid - cx) if W_vid > 0 else w
    w_max_y = R * 2 * min(cy, H_vid - cy) if H_vid > 0 else w
    w_clamp = min(w, w_max_x, w_max_y)
    if w_clamp < 1:
        w_clamp = w
    h_clamp = w_clamp / R
    if h_clamp > H_vid or w_clamp > W_vid:
        scale = min(W_vid / w_clamp, H_vid / h_clamp)
        w_clamp *= scale
        h_clamp *= scale
    return (cx, cy, w_clamp, h_clamp)


def clamp_crop_to_frame(cx, cy, w, h, W_vid, H_vid):
    """Return (x1, y1, x2, y2) integer crop box clamped to frame."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    x1 = max(0, min(x1, W_vid - 1))
    y1 = max(0, min(y1, H_vid - 1))
    x2 = max(0, min(x2, W_vid))
    y2 = max(0, min(y2, H_vid))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, W_vid, H_vid)
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def extract_crop(frame, x1, y1, x2, y2, out_w, out_h):
    """Extract rectangle and resize to (out_w, out_h)."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def run(args):
    if getattr(args, "source_pose_video_paths", "").strip() and not getattr(args, "output_pose_video", "").strip():
        logger.error("When --source_pose_video_paths is set, --output_pose_video is required.")
        return
    # Reference
    ref_frame = cv2.imread(args.ref_name, cv2.IMREAD_COLOR)
    if ref_frame is None:
        logger.error("Cannot read reference image: %s", args.ref_name)
        return
    H_ref, W_ref = ref_frame.shape[:2]

    logger.info("Loading DWPose ...")
    dwpose = DWposeDetector()
    pose_ref = dwpose(ref_frame)
    ref_cand = pose_ref["bodies"]["candidate"]

    # Visibility region and visible keypoints
    vis_margin = getattr(args, "visibility_margin", 0.05)
    sam_ckpt = getattr(args, "sam_checkpoint", None) or ""
    if sam_ckpt and _SAM_AVAILABLE:
        logger.info("Loading SAM for visibility ...")
        sam_pred = load_sam_predictor(sam_ckpt)
        yolox_sess = dwpose.pose_estimation.session_det
        visible_region = compute_visibility_region_sam(
            ref_frame, sam_pred, yolox_sess, margin=vis_margin
        )
        del sam_pred
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        if sam_ckpt and not _SAM_AVAILABLE:
            logger.warning("SAM checkpoint given but segment-anything not installed; using keypoint visibility.")
        visible_region = compute_visibility_region_keypoints(ref_cand, margin=vis_margin)
    visible_indices = _visible_keypoint_indices(ref_cand, visible_region)
    logger.info("Visible keypoint indices: %s", visible_indices)
    if len(visible_indices) < 2:
        logger.error("Need at least 2 visible keypoints on reference.")
        return

    # Video input: single file or directory of videos
    if args.source_video_paths.endswith(".mp4") or args.source_video_paths.endswith(".avi"):
        video_paths = [args.source_video_paths]
    else:
        video_paths = [
            os.path.join(args.source_video_paths, f)
            for f in sorted(os.listdir(args.source_video_paths))
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]
    if not video_paths:
        logger.error("No video files found under %s", args.source_video_paths)
        return

    # Collect all frames and fps from first video
    frames_list = []
    input_fps = 30.0
    for vpath in video_paths:
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            logger.warning("Could not open %s", vpath)
            continue
        if frames_list == []:
            input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        cap.release()
    if not frames_list:
        logger.error("No frames read from videos.")
        return

    # Optionally load pose video (same crop applied; must match frame count or we use min)
    pose_frames_list = []
    source_pose = getattr(args, "source_pose_video_paths", "").strip()
    output_pose = getattr(args, "output_pose_video", "").strip()
    if source_pose and output_pose:
        if source_pose.endswith(".mp4") or source_pose.endswith(".avi") or source_pose.endswith(".mov"):
            pose_paths = [source_pose]
        else:
            pose_paths = [
                os.path.join(source_pose, f)
                for f in sorted(os.listdir(source_pose))
                if f.lower().endswith((".mp4", ".avi", ".mov"))
            ]
        for vpath in pose_paths:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                logger.warning("Could not open pose video %s", vpath)
                continue
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pose_frames_list.append(frame)
            cap.release()
        if len(pose_frames_list) != len(frames_list):
            n_use = min(len(pose_frames_list), len(frames_list))
            logger.warning(
                "Pose video has %d frames, driven video has %d; using first %d frames for both.",
                len(pose_frames_list), len(frames_list), n_use,
            )
            frames_list = frames_list[:n_use]
            pose_frames_list = pose_frames_list[:n_use]
        else:
            logger.info("Loaded pose video: %d frames (same as driven video).", len(pose_frames_list))

    # Output resolution: same as driven video by default; override with --output_size if set
    out_w = frames_list[0].shape[1]
    out_h = frames_list[0].shape[0]
    if getattr(args, "output_size", "").strip():
        parts = args.output_size.strip().lower().split("x")
        if len(parts) == 2:
            out_w, out_h = int(parts[0]), int(parts[1])
    # libx264 requires width and height divisible by 2
    out_w = max(2, int(out_w) // 2 * 2)
    out_h = max(2, int(out_h) // 2 * 2)
    logger.info("Output resolution: %dx%d (same as driven video)", out_w, out_h)

    fps = getattr(args, "fps", None) or input_fps
    if getattr(args, "fps", None) is not None:
        logger.info("Output FPS: %.2f (from --fps)", fps)
    else:
        logger.info("Output FPS: %.2f (auto-detected from driven video)", fps)

    # Determine crop area once from first frame + ref (temporal consistency, no flickering)
    first_frame = frames_list[0]
    H_vid, W_vid = first_frame.shape[:2]
    logger.info("Computing crop from first frame and reference ...")
    pose_first = dwpose(first_frame)
    vid_cand_first = pose_first["bodies"]["candidate"]
    common = [j for j in visible_indices if j < vid_cand_first.shape[0] and is_valid_kp(vid_cand_first[j])]
    if len(common) < 2:
        cx = W_vid / 2
        cy = H_vid / 2
        w = min(W_vid, H_vid * W_ref / H_ref)
        h = w * H_ref / W_ref
        logger.warning("Few keypoints on first frame; using center crop.")
    else:
        crop_result = compute_crop_from_keypoints(
            ref_cand, vid_cand_first, common, W_ref, H_ref, W_vid, H_vid
        )
        if crop_result is None:
            cx, cy = W_vid / 2, H_vid / 2
            w = min(W_vid, H_vid * W_ref / H_ref)
            h = w * H_ref / W_ref
            logger.warning("Crop solver failed; using center crop.")
        else:
            cx, cy, w, h = crop_result
    x1, y1, x2, y2 = clamp_crop_to_frame(cx, cy, w, h, W_vid, H_vid)
    logger.info("Fixed crop area (applied to all frames): cx=%.1f cy=%.1f w=%.1f h=%.1f", cx, cy, w, h)

    out_dir = os.path.dirname(args.output_video)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if pose_frames_list:
        pose_out_dir = os.path.dirname(args.output_pose_video)
        if pose_out_dir:
            os.makedirs(pose_out_dir, exist_ok=True)
    temp_frames_dir = tempfile.mkdtemp(prefix="video_crop_align_")
    temp_pose_frames_dir = tempfile.mkdtemp(prefix="video_crop_align_pose_") if pose_frames_list else None
    crop_history = [{"frame": 0, "cx": cx, "cy": cy, "w": w, "h": h}]
    try:
        for fi, frame in enumerate(frames_list):
            H_vid, W_vid = frame.shape[:2]
            x1, y1, x2, y2 = clamp_crop_to_frame(cx, cy, w, h, W_vid, H_vid)
            out_frame = extract_crop(frame, x1, y1, x2, y2, out_w, out_h)
            if out_frame is not None:
                frame_path = os.path.join(temp_frames_dir, "{:04d}.jpg".format(fi))
                cv2.imwrite(frame_path, out_frame)
            if pose_frames_list and fi < len(pose_frames_list):
                pose_frame = pose_frames_list[fi]
                H_p, W_p = pose_frame.shape[:2]
                # Scale crop to pose video if resolution differs (same relative crop area)
                if (W_p, H_p) == (W_vid, H_vid):
                    cx_p, cy_p, w_p, h_p = cx, cy, w, h
                else:
                    cx_p = cx * (W_p / float(W_vid))
                    cy_p = cy * (H_p / float(H_vid))
                    w_p = w * (W_p / float(W_vid))
                    h_p = h * (H_p / float(H_vid))
                x1_p, y1_p, x2_p, y2_p = clamp_crop_to_frame(cx_p, cy_p, w_p, h_p, W_p, H_p)
                out_pose_frame = extract_crop(pose_frame, x1_p, y1_p, x2_p, y2_p, out_w, out_h)
                if out_pose_frame is not None:
                    pose_frame_path = os.path.join(temp_pose_frames_dir, "{:04d}.jpg".format(fi))
                    cv2.imwrite(pose_frame_path, out_pose_frame)

            if (fi + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", fi + 1, len(frames_list))

        # Encode driven video with ffmpeg (libx264)
        ffmpeg_cmd = (
            "ffmpeg -y -framerate {} -i {}/%04d.jpg "
            "-c:v libx264 -pix_fmt yuv420p -crf 18 {}"
        ).format(fps, temp_frames_dir, args.output_video)
        logger.info("Encoding video with ffmpeg ...")
        ret = os.system(ffmpeg_cmd)
        if ret != 0:
            logger.error("ffmpeg encoding failed (exit code %s). Ensure ffmpeg is installed.", ret)
        else:
            logger.info("Wrote %s (%d frames, %.1f fps)", args.output_video, len(frames_list), fps)

        # Encode pose video if provided
        if pose_frames_list and temp_pose_frames_dir:
            ffmpeg_pose_cmd = (
                "ffmpeg -y -framerate {} -i {}/%04d.jpg "
                "-c:v libx264 -pix_fmt yuv420p -crf 18 {}"
            ).format(fps, temp_pose_frames_dir, args.output_pose_video)
            logger.info("Encoding pose video with ffmpeg ...")
            ret_pose = os.system(ffmpeg_pose_cmd)
            if ret_pose != 0:
                logger.error("ffmpeg pose video encoding failed (exit code %s).", ret_pose)
            else:
                logger.info("Wrote %s (%d frames, %.1f fps)", args.output_pose_video, len(pose_frames_list), fps)
    finally:
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        if temp_pose_frames_dir:
            shutil.rmtree(temp_pose_frames_dir, ignore_errors=True)

    if not getattr(args, "video_only", True) and crop_history:
        out_json = args.output_video.rsplit(".", 1)[0] + "_crops.json"
        json_dir = os.path.dirname(out_json)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(crop_history, f, indent=0)
        logger.info("Wrote crop boxes to %s", out_json)


def main():
    parser = argparse.ArgumentParser(
        description="Crop full-body video to match partial-body reference framing."
    )
    parser.add_argument("--ref_name", type=str, required=True, help="Reference image (partial body).")
    parser.add_argument(
        "--source_video_paths",
        type=str,
        required=True,
        help="Input video file (.mp4/.avi) or directory of video files.",
    )
    parser.add_argument("--output_video", type=str, required=True, help="Output cropped video path.")
    parser.add_argument(
        "--source_pose_video_paths",
        type=str,
        default="",
        help="Pose video (same length as driven video); crop is applied to produce a second output.",
    )
    parser.add_argument(
        "--output_pose_video",
        type=str,
        default="",
        help="Output path for cropped pose video (required when --source_pose_video_paths is set).",
    )
    parser.add_argument(
        "--output_size",
        type=str,
        default="",
        help="Output size as WxH (default: same as reference).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS (default: auto-detect from driven video; fallback 30).",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="",
        help="Path to SAM checkpoint for visibility region (optional).",
    )
    parser.add_argument(
        "--visibility_margin",
        type=float,
        default=0.05,
        help="Margin around visible region (default: 0.05).",
    )
    parser.add_argument(
        "--video_only",
        action="store_true",
        default=True,
        help="Do not save crop JSON (default: True).",
    )
    parser.add_argument(
        "--save_crops",
        action="store_true",
        help="Save crop boxes to JSON next to output video.",
    )
    args = parser.parse_args()
    if args.save_crops:
        args.video_only = False
    run(args)


if __name__ == "__main__":
    main()
