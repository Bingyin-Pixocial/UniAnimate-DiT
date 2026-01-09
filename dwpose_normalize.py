# Normalized Pose Extraction
# Given any video, outputs a skeleton video with:
# 1. Fixed body proportions (canonical human ratios)
# 2. Fixed position on canvas (anchored at neck)
# This ensures two different characters performing the same pose produce identical skeleton outputs.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import numpy as np
import copy
import argparse
import logging
import sys

import dwpose.util as util
from dwpose.wholebody import Wholebody


def get_logger(name="pose_normalize"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger


logger = get_logger('pose_normalize')


# =============================================================================
# Canonical Body Proportions (normalized, relative to torso height)
# These define the "standard" human body ratios that all outputs will use.
# Values are in normalized coordinates (0-1 range relative to canvas)
# =============================================================================

class CanonicalBody:
    """
    Defines canonical body proportions for a normalized human skeleton.
    All lengths are relative to a base unit (torso_height).
    
    Keypoint indices:
    0: nose, 1: neck, 2: right shoulder, 3: right elbow, 4: right wrist
    5: left shoulder, 6: left elbow, 7: left wrist
    8: right hip, 9: right knee, 10: right ankle
    11: left hip, 12: left knee, 13: left ankle
    14: right eye, 15: left eye, 16: right ear, 17: left ear
    18: left foot, 19: right foot
    """
    
    def __init__(self, canvas_height=768, canvas_width=512):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        
        # Base unit: torso height (neck to hip center) as fraction of canvas height
        # This controls the overall scale of the skeleton
        self.torso_height = 0.25  # 25% of canvas height
        
        # All proportions relative to torso_height = 1.0
        self.proportions = {
            # Torso
            'neck_length': 0.15,        # nose to neck
            'shoulder_width': 0.5,      # half width each side from neck
            'hip_width': 0.35,          # half width each side from spine
            
            # Arms
            'upper_arm': 0.45,          # shoulder to elbow
            'lower_arm': 0.40,          # elbow to wrist
            
            # Legs
            'upper_leg': 0.65,          # hip to knee
            'lower_leg': 0.60,          # knee to ankle
            'foot_length': 0.15,        # ankle to foot
            
            # Head
            'eye_offset_x': 0.08,       # eye horizontal offset from nose
            'eye_offset_y': 0.05,       # eye vertical offset from nose
            'ear_offset_x': 0.12,       # ear horizontal offset from eye
            'ear_offset_y': 0.02,       # ear vertical offset from eye
        }
        
        # Anchor point: neck position on canvas (normalized 0-1)
        # Positioned in upper-center of canvas
        self.anchor_x = 0.5           # center horizontally
        self.anchor_y = 0.25          # 25% from top
        
    def get_length(self, name):
        """Get absolute length in normalized coordinates (0-1)"""
        return self.proportions[name] * self.torso_height


class DWposeDetector:
    def __init__(self):
        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18].copy()

            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            bodyfoot_score = subset[:, :24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > 0.3:
                        bodyfoot_score[i][j] = int(18 * i + j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:, 18] and -1 not in bodyfoot_score[:, 19]:
                bodyfoot_score[:, 18] = np.array([18.])
            else:
                bodyfoot_score[:, 18] = np.array([-1.])
            if -1 not in bodyfoot_score[:, 21] and -1 not in bodyfoot_score[:, 22]:
                bodyfoot_score[:, 19] = np.array([19.])
            else:
                bodyfoot_score[:, 19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:, :24].copy()

            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18] + bodyfoot[i][19]) / 2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21] + bodyfoot[i][22]) / 2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])

            bodyfoot = bodyfoot[:, :20, :]
            bodyfoot = bodyfoot.reshape(nums * 20, locs)

            faces = candidate[:, 24:92]
            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return pose


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_body_and_foot(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas_without_face = copy.deepcopy(canvas)
    canvas = util.draw_facepose(canvas, faces)

    return canvas_without_face, canvas


def compute_limb_length(p1, p2):
    """Compute Euclidean distance between two points. Returns -1 if either point is invalid."""
    if p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0:
        return -1
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_unit_vector(p1, p2):
    """Get unit vector from p1 to p2. Returns None if points are invalid or identical."""
    if p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0:
        return None
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-6:
        return None
    return diff / length


def normalize_pose(pose, canonical):
    """
    Normalize a detected pose to canonical body proportions and fixed anchor position.
    
    Args:
        pose: Dictionary with 'bodies', 'hands', 'faces' from DWpose
        canonical: CanonicalBody instance defining target proportions
        
    Returns:
        Normalized pose dictionary
    """
    pose = copy.deepcopy(pose)
    
    candidate = pose['bodies']['candidate']
    hands = pose['hands']
    faces = pose['faces']
    
    # Get current keypoints
    nose = candidate[0].copy()
    neck = candidate[1].copy()
    r_shoulder = candidate[2].copy()
    r_elbow = candidate[3].copy()
    r_wrist = candidate[4].copy()
    l_shoulder = candidate[5].copy()
    l_elbow = candidate[6].copy()
    l_wrist = candidate[7].copy()
    r_hip = candidate[8].copy()
    r_knee = candidate[9].copy()
    r_ankle = candidate[10].copy()
    l_hip = candidate[11].copy()
    l_knee = candidate[12].copy()
    l_ankle = candidate[13].copy()
    r_eye = candidate[14].copy()
    l_eye = candidate[15].copy()
    r_ear = candidate[16].copy()
    l_ear = candidate[17].copy()
    l_foot = candidate[18].copy()
    r_foot = candidate[19].copy()
    
    # Check if we have valid neck (anchor point)
    if neck[0] < 0 or neck[1] < 0:
        # Try to estimate neck from shoulders
        if r_shoulder[0] >= 0 and l_shoulder[0] >= 0:
            neck = (r_shoulder + l_shoulder) / 2
            neck[1] -= 0.05  # Slightly above shoulder midpoint
        else:
            logger.warning("Cannot find valid anchor point, skipping normalization")
            return pose
    
    # ==========================================================================
    # Step 1: Build normalized skeleton from neck (anchor) using canonical lengths
    # ==========================================================================
    
    # New keypoints array
    new_candidate = np.full_like(candidate, -1.0)
    
    # Anchor: neck at fixed position
    new_neck = np.array([canonical.anchor_x, canonical.anchor_y])
    new_candidate[1] = new_neck
    
    # --- Nose (above neck) ---
    if nose[0] >= 0:
        dir_neck_nose = get_unit_vector(neck, nose)
        if dir_neck_nose is not None:
            new_candidate[0] = new_neck + dir_neck_nose * canonical.get_length('neck_length')
        else:
            # Default: straight up
            new_candidate[0] = new_neck + np.array([0, -canonical.get_length('neck_length')])
    
    # --- Shoulders ---
    # Right shoulder (candidate[2])
    if r_shoulder[0] >= 0:
        dir_neck_rshoulder = get_unit_vector(neck, r_shoulder)
        if dir_neck_rshoulder is not None:
            new_candidate[2] = new_neck + dir_neck_rshoulder * canonical.get_length('shoulder_width')
        else:
            new_candidate[2] = new_neck + np.array([-canonical.get_length('shoulder_width'), 0])
    
    # Left shoulder (candidate[5])
    if l_shoulder[0] >= 0:
        dir_neck_lshoulder = get_unit_vector(neck, l_shoulder)
        if dir_neck_lshoulder is not None:
            new_candidate[5] = new_neck + dir_neck_lshoulder * canonical.get_length('shoulder_width')
        else:
            new_candidate[5] = new_neck + np.array([canonical.get_length('shoulder_width'), 0])
    
    # --- Hips ---
    # Hip center is below neck by torso_height
    hip_center_y = new_neck[1] + canonical.torso_height
    
    # Right hip (candidate[8])
    if r_hip[0] >= 0:
        # Preserve relative horizontal position but use canonical vertical
        hip_center_current = (r_hip + l_hip) / 2 if l_hip[0] >= 0 else r_hip
        dir_center_rhip = get_unit_vector(hip_center_current, r_hip) if l_hip[0] >= 0 else np.array([-1, 0])
        if dir_center_rhip is not None:
            new_hip_center = np.array([new_neck[0], hip_center_y])
            new_candidate[8] = new_hip_center + np.array([-canonical.get_length('hip_width'), 0])
    
    # Left hip (candidate[11])
    if l_hip[0] >= 0:
        new_hip_center = np.array([new_neck[0], hip_center_y])
        new_candidate[11] = new_hip_center + np.array([canonical.get_length('hip_width'), 0])
    
    # --- Right arm ---
    if new_candidate[2][0] >= 0:  # If right shoulder is valid
        # Right elbow
        if r_elbow[0] >= 0:
            dir_rshoulder_relbow = get_unit_vector(r_shoulder, r_elbow)
            if dir_rshoulder_relbow is not None:
                new_candidate[3] = new_candidate[2] + dir_rshoulder_relbow * canonical.get_length('upper_arm')
            
            # Right wrist
            if r_wrist[0] >= 0 and new_candidate[3][0] >= 0:
                dir_relbow_rwrist = get_unit_vector(r_elbow, r_wrist)
                if dir_relbow_rwrist is not None:
                    new_candidate[4] = new_candidate[3] + dir_relbow_rwrist * canonical.get_length('lower_arm')
    
    # --- Left arm ---
    if new_candidate[5][0] >= 0:  # If left shoulder is valid
        # Left elbow
        if l_elbow[0] >= 0:
            dir_lshoulder_lelbow = get_unit_vector(l_shoulder, l_elbow)
            if dir_lshoulder_lelbow is not None:
                new_candidate[6] = new_candidate[5] + dir_lshoulder_lelbow * canonical.get_length('upper_arm')
            
            # Left wrist
            if l_wrist[0] >= 0 and new_candidate[6][0] >= 0:
                dir_lelbow_lwrist = get_unit_vector(l_elbow, l_wrist)
                if dir_lelbow_lwrist is not None:
                    new_candidate[7] = new_candidate[6] + dir_lelbow_lwrist * canonical.get_length('lower_arm')
    
    # --- Right leg ---
    if new_candidate[8][0] >= 0:  # If right hip is valid
        # Right knee
        if r_knee[0] >= 0:
            dir_rhip_rknee = get_unit_vector(r_hip, r_knee)
            if dir_rhip_rknee is not None:
                new_candidate[9] = new_candidate[8] + dir_rhip_rknee * canonical.get_length('upper_leg')
            
            # Right ankle
            if r_ankle[0] >= 0 and new_candidate[9][0] >= 0:
                dir_rknee_rankle = get_unit_vector(r_knee, r_ankle)
                if dir_rknee_rankle is not None:
                    new_candidate[10] = new_candidate[9] + dir_rknee_rankle * canonical.get_length('lower_leg')
                
                # Right foot
                if r_foot[0] >= 0 and new_candidate[10][0] >= 0:
                    dir_rankle_rfoot = get_unit_vector(r_ankle, r_foot)
                    if dir_rankle_rfoot is not None:
                        new_candidate[19] = new_candidate[10] + dir_rankle_rfoot * canonical.get_length('foot_length')
    
    # --- Left leg ---
    if new_candidate[11][0] >= 0:  # If left hip is valid
        # Left knee
        if l_knee[0] >= 0:
            dir_lhip_lknee = get_unit_vector(l_hip, l_knee)
            if dir_lhip_lknee is not None:
                new_candidate[12] = new_candidate[11] + dir_lhip_lknee * canonical.get_length('upper_leg')
            
            # Left ankle
            if l_ankle[0] >= 0 and new_candidate[12][0] >= 0:
                dir_lknee_lankle = get_unit_vector(l_knee, l_ankle)
                if dir_lknee_lankle is not None:
                    new_candidate[13] = new_candidate[12] + dir_lknee_lankle * canonical.get_length('lower_leg')
                
                # Left foot
                if l_foot[0] >= 0 and new_candidate[13][0] >= 0:
                    dir_lankle_lfoot = get_unit_vector(l_ankle, l_foot)
                    if dir_lankle_lfoot is not None:
                        new_candidate[18] = new_candidate[13] + dir_lankle_lfoot * canonical.get_length('foot_length')
    
    # --- Head features (relative to nose) ---
    if new_candidate[0][0] >= 0:  # If nose is valid
        new_nose = new_candidate[0]
        
        # Right eye
        if r_eye[0] >= 0:
            new_candidate[14] = new_nose + np.array([
                -canonical.get_length('eye_offset_x'),
                -canonical.get_length('eye_offset_y')
            ])
        
        # Left eye
        if l_eye[0] >= 0:
            new_candidate[15] = new_nose + np.array([
                canonical.get_length('eye_offset_x'),
                -canonical.get_length('eye_offset_y')
            ])
        
        # Right ear
        if r_ear[0] >= 0 and new_candidate[14][0] >= 0:
            new_candidate[16] = new_candidate[14] + np.array([
                -canonical.get_length('ear_offset_x'),
                canonical.get_length('ear_offset_y')
            ])
        
        # Left ear
        if l_ear[0] >= 0 and new_candidate[15][0] >= 0:
            new_candidate[17] = new_candidate[15] + np.array([
                canonical.get_length('ear_offset_x'),
                canonical.get_length('ear_offset_y')
            ])
    
    # ==========================================================================
    # Step 2: Normalize hands relative to wrists
    # ==========================================================================
    
    new_hands = np.full_like(hands, -1.0)
    
    # Calculate scale factor for hands based on canonical proportions
    hand_scale = canonical.torso_height * 0.15  # Hand size relative to body
    
    # Left hand (hands[0]) relative to left wrist
    if new_candidate[7][0] >= 0 and l_wrist[0] >= 0:
        wrist_offset = new_candidate[7] - l_wrist
        # Scale hand keypoints
        for j in range(hands.shape[1]):
            if hands[0, j, 0] >= 0:
                # Get relative position to wrist
                rel_pos = hands[0, j] - l_wrist
                # Normalize and scale
                orig_dist = np.linalg.norm(rel_pos)
                if orig_dist > 1e-6:
                    new_hands[0, j] = new_candidate[7] + (rel_pos / orig_dist) * hand_scale * (orig_dist / 0.1)
    
    # Right hand (hands[1]) relative to right wrist
    if new_candidate[4][0] >= 0 and r_wrist[0] >= 0:
        for j in range(hands.shape[1]):
            if hands[1, j, 0] >= 0:
                rel_pos = hands[1, j] - r_wrist
                orig_dist = np.linalg.norm(rel_pos)
                if orig_dist > 1e-6:
                    new_hands[1, j] = new_candidate[4] + (rel_pos / orig_dist) * hand_scale * (orig_dist / 0.1)
    
    # ==========================================================================
    # Step 3: Normalize face relative to nose
    # ==========================================================================
    
    new_faces = np.full_like(faces, -1.0)
    
    if new_candidate[0][0] >= 0 and nose[0] >= 0:
        face_scale = canonical.torso_height * 0.4  # Face size relative to body
        
        for j in range(faces.shape[1]):
            if faces[0, j, 0] >= 0:
                rel_pos = faces[0, j] - nose
                orig_dist = np.linalg.norm(rel_pos)
                if orig_dist > 1e-6:
                    new_faces[0, j] = new_candidate[0] + (rel_pos / orig_dist) * face_scale * (orig_dist / 0.15)
                else:
                    new_faces[0, j] = new_candidate[0]
    
    # Update pose
    pose['bodies']['candidate'] = new_candidate
    pose['hands'] = new_hands
    pose['faces'] = new_faces
    
    return pose


def process_video(video_path, output_path, canvas_height=768, canvas_width=512, save_frames=False):
    """
    Process a video and output normalized skeleton video at the same fps as input.
    
    Args:
        video_path: Path to input video
        output_path: Path to output skeleton video (e.g., 'output/skeleton.mp4')
        canvas_height: Output canvas height
        canvas_width: Output canvas width
        save_frames: If True, also save individual frames to output directory
    """
    logger.info(f'Loading DWpose model...')
    dwpose_model = DWposeDetector()
    
    logger.info(f'Creating canonical body definition...')
    canonical = CanonicalBody(canvas_height, canvas_width)
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read video
    logger.info(f'Processing video: {video_path}')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f'Failed to open video: {video_path}')
        return
    
    # Get input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f'Video info: {total_frames} frames at {fps} fps')
    
    # Initialize video writer with same fps as input
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
    
    if not out.isOpened():
        logger.error(f'Failed to create output video: {output_path}')
        cap.release()
        return
    
    # Create frames directory if saving frames
    if save_frames:
        frames_dir = os.path.splitext(output_path)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pose
        pose = dwpose_model(frame)
        
        # Normalize pose to canonical proportions and fixed position
        normalized_pose = normalize_pose(pose, canonical)
        
        # Draw skeleton
        skeleton_woface, skeleton_wface = draw_pose(normalized_pose, canvas_height, canvas_width)
        
        # Write frame to video
        out.write(skeleton_woface)
        
        # Optionally save individual frame
        if save_frames:
            frame_path = os.path.join(frames_dir, f'{frame_idx:04d}.jpg')
            cv2.imwrite(frame_path, skeleton_woface)
        
        if frame_idx % 30 == 0:
            logger.info(f'Processed frame {frame_idx}/{total_frames}')
        
        frame_idx += 1
    
    cap.release()
    out.release()
    logger.info(f'Finished processing. {frame_idx} frames written to {output_path} at {fps} fps')


def main():
    parser = argparse.ArgumentParser(
        description="Normalize pose skeletons to canonical body proportions and fixed position. "
                    "Any two videos with the same pose sequence will produce identical skeleton outputs."
    )
    parser.add_argument(
        "--source_video", 
        type=str, 
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_video", 
        type=str, 
        required=True,
        help="Path to output skeleton video (e.g., output/skeleton.mp4)"
    )
    parser.add_argument(
        "--canvas_height", 
        type=int, 
        default=768,
        help="Output canvas height (default: 768)"
    )
    parser.add_argument(
        "--canvas_width", 
        type=int, 
        default=512,
        help="Output canvas width (default: 512)"
    )
    parser.add_argument(
        "--torso_height", 
        type=float, 
        default=0.25,
        help="Torso height as fraction of canvas (default: 0.25)"
    )
    parser.add_argument(
        "--anchor_x", 
        type=float, 
        default=0.5,
        help="Anchor X position (0-1, default: 0.5 = center)"
    )
    parser.add_argument(
        "--anchor_y", 
        type=float, 
        default=0.25,
        help="Anchor Y position (0-1, default: 0.25 = upper area)"
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Also save individual frames as images"
    )
    
    args = parser.parse_args()
    
    process_video(
        args.source_video,
        args.output_video,
        args.canvas_height,
        args.canvas_width,
        args.save_frames
    )


if __name__ == '__main__':
    main()
