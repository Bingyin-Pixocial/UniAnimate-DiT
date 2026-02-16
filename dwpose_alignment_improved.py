# Improved DWpose Alignment for UniAnimate-DiT
# =============================================
# Based on dwpose_alignment.py with the following improvements:
#   1. Angle-based (rotation) retargeting instead of translation-based
#   2. Temporal smoothing using One Euro Filter
#   3. Missing/occluded keypoint handling with temporal interpolation
#   4. Hands and face aligned relative to parent joints (wrist / nose)
#   5. Ground-plane / foot contact constraints
#   6. Per-frame adaptive scaling for depth changes
#   7. Two-anchor (hip-center + neck) alignment for natural body sway
#   8. Physical plausibility post-processing (angle limits, boundary clamping)
#
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# DWpose detector unchanged; only retargeting pipeline is improved.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
import copy
import argparse
import shutil
import math
import logging
import sys

import dwpose.util as util
from dwpose.wholebody import Wholebody


# =============================================================================
# Constants: kinematic tree and joint metadata (OpenPose 20-keypoint body+feet)
# =============================================================================
# Index mapping:
#   0=nose, 1=neck, 2=Rshoulder, 3=Relbow, 4=Rwrist,
#   5=Lshoulder, 6=Lelbow, 7=Lwrist, 8=Rhip, 9=Rknee,
#   10=Rankle, 11=Lhip, 12=Lknee, 13=Lankle, 14=Reye,
#   15=Leye, 16=Rear, 17=Lear, 18=Lfoot, 19=Rfoot

# Parent->child pairs ordered root-outward (neck=1 is the root hub)
KINEMATIC_CHAINS = [
    # Head
    (1, 0),    # neck -> nose
    (0, 14),   # nose -> right eye
    (14, 16),  # right eye -> right ear
    (0, 15),   # nose -> left eye
    (15, 17),  # left eye -> left ear
    # Right arm
    (1, 2),    # neck -> right shoulder
    (2, 3),    # right shoulder -> right elbow
    (3, 4),    # right elbow -> right wrist
    # Left arm
    (1, 5),    # neck -> left shoulder
    (5, 6),    # left shoulder -> left elbow
    (6, 7),    # left elbow -> left wrist
    # Right leg
    (1, 8),    # neck -> right hip
    (8, 9),    # right hip -> right knee
    (9, 10),   # right knee -> right ankle
    (10, 19),  # right ankle -> right foot
    # Left leg
    (1, 11),   # neck -> left hip
    (11, 12),  # left hip -> left knee
    (12, 13),  # left knee -> left ankle
    (13, 18),  # left ankle -> left foot
]

# hands[0] = left hand (wrist=7), hands[1] = right hand (wrist=4)
HAND_WRIST_MAP = {0: 7, 1: 4}

# Elbow index for each wrist (used for hand-scale computation)
WRIST_ELBOW_MAP = {4: 3, 7: 6}

# Joint angle limits (parent, joint, child, min_deg, max_deg)
ANGLE_LIMIT_CONFIGS = [
    (2, 3, 4, 5, 175),    # right elbow
    (5, 6, 7, 5, 175),    # left elbow
    (8, 9, 10, 5, 175),   # right knee
    (11, 12, 13, 5, 175), # left knee
]

# Critical joints that must be detected for reliable retargeting
CRITICAL_JOINTS = [0, 1, 2, 5, 8, 11]


# =============================================================================
# One Euro Filter: adaptive low-pass filter for temporal smoothing
# =============================================================================
def _smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """Attempt to reduce jitter while preserving fast motions."""

    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 1e-10:
            return x

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# =============================================================================
# Logging
# =============================================================================
def get_logger(name="essmc2"):
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


# =============================================================================
# Geometry helpers
# =============================================================================
def is_valid_kp(kp):
    """Return True if keypoint is detected (not the -1 sentinel)."""
    return kp[0] > -0.5 and kp[1] > -0.5


def bone_length(p1, p2):
    """Euclidean distance between two 2-D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalize_vec(v):
    """Unit vector; returns zeros if degenerate."""
    ln = math.sqrt(v[0] ** 2 + v[1] ** 2)
    if ln < 1e-8:
        return np.array([0.0, 0.0])
    return v / ln


def angle_between(v1, v2):
    """Angle in degrees between two 2-D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def rotate_2d(v, angle_rad):
    """Rotate vector v by angle_rad (counter-clockwise)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([v[0] * c - v[1] * s,
                     v[0] * s + v[1] * c])


# =============================================================================
# DWpose Detector (unchanged from original)
# =============================================================================
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
            if -1 not in bodyfoot_score[:, 18] and \
               -1 not in bodyfoot_score[:, 19]:
                bodyfoot_score[:, 18] = np.array([18.])
            else:
                bodyfoot_score[:, 18] = np.array([-1.])
            if -1 not in bodyfoot_score[:, 21] and \
               -1 not in bodyfoot_score[:, 22]:
                bodyfoot_score[:, 19] = np.array([19.])
            else:
                bodyfoot_score[:, 19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:, :24].copy()
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and \
                   -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18] +
                                       bodyfoot[i][19]) / 2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and \
                   -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21] +
                                       bodyfoot[i][22]) / 2
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


# =============================================================================
# Drawing (unchanged from original)
# =============================================================================
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


# =============================================================================
# Improvement 2: reference bone lengths + angle-based retargeting
# =============================================================================
def compute_ref_bone_lengths(ref_candidate):
    """Compute bone lengths from the reference character's pose."""
    lengths = {}
    for parent, child in KINEMATIC_CHAINS:
        p, c = ref_candidate[parent], ref_candidate[child]
        if is_valid_kp(p) and is_valid_kp(c):
            lengths[(parent, child)] = bone_length(p, c)
        else:
            lengths[(parent, child)] = None
    return lengths


def retarget_body_angle_based(driving_cand, ref_cand,
                              ref_bone_lengths, root_pos):
    """
    Reconstruct skeleton using driving-pose directions + reference
    bone lengths.

    For each bone (parent -> child) in the kinematic chain:
      direction = normalize(driving_child - driving_parent)
      child_pos = retargeted_parent + direction * ref_bone_length

    This preserves the driving motion's joint angles while applying
    the reference character's body proportions.

    Parameters
    ----------
    driving_cand     : (20, 2) driving frame keypoints (normalised).
    ref_cand         : (20, 2) reference character keypoints.
    ref_bone_lengths : dict (parent, child) -> float.
    root_pos         : (2,) neck position for kinematic root.

    Returns
    -------
    retargeted : (20, 2) array.
    """
    retargeted = driving_cand.copy()
    retargeted[1] = root_pos

    for parent, child in KINEMATIC_CHAINS:
        drv_p = driving_cand[parent]
        drv_c = driving_cand[child]

        if not is_valid_kp(drv_c):
            retargeted[child] = np.array([-1.0, -1.0])
            continue

        if not is_valid_kp(drv_p):
            # Parent invalid, child valid: use reference direction
            ref_p = ref_cand[parent]
            ref_c = ref_cand[child]
            if is_valid_kp(ref_p) and is_valid_kp(ref_c):
                direction = normalize_vec(ref_c - ref_p)
            else:
                direction = np.array([0.0, 0.0])
            ref_len = ref_bone_lengths.get((parent, child))
            if ref_len is None:
                ref_len = 0.0
            retargeted[child] = retargeted[parent] + direction * ref_len
            continue

        # Normal case: both driving parent and child are valid
        direction = normalize_vec(drv_c - drv_p)

        # Degenerate direction: fall back to reference
        if np.linalg.norm(direction) < 1e-8:
            ref_p = ref_cand[parent]
            ref_c = ref_cand[child]
            if is_valid_kp(ref_p) and is_valid_kp(ref_c):
                direction = normalize_vec(ref_c - ref_p)

        ref_len = ref_bone_lengths.get((parent, child))
        if ref_len is None:
            ref_len = bone_length(drv_p, drv_c)

        retargeted[child] = retargeted[parent] + direction * ref_len

    return retargeted


# =============================================================================
# Improvement 3: temporal interpolation for missing keypoints
# =============================================================================
def _interp_1d_sequence(values, valid_mask):
    """Linear-interpolate a 1-D sequence where valid_mask[i] is True."""
    n = len(values)
    if n == 0 or not np.any(valid_mask):
        return values

    valid_indices = np.where(valid_mask)[0]
    for i in range(n):
        if valid_mask[i]:
            continue
        prev_arr = valid_indices[valid_indices <= i]
        next_arr = valid_indices[valid_indices >= i]
        prev_v = int(prev_arr[-1]) if len(prev_arr) > 0 else None
        next_v = int(next_arr[0]) if len(next_arr) > 0 else None

        if (prev_v is not None and next_v is not None
                and prev_v != next_v):
            t = (i - prev_v) / (next_v - prev_v)
            values[i] = (values[prev_v]
                         + t * (values[next_v] - values[prev_v]))
        elif prev_v is not None:
            values[i] = values[prev_v]
        elif next_v is not None:
            values[i] = values[next_v]
    return values


def interpolate_missing_keypoints(poses_seq):
    """Fill in missing (-1) keypoints via linear temporal interpolation."""
    n = len(poses_seq)
    if n < 2:
        return poses_seq

    num_body = poses_seq[0]['bodies']['candidate'].shape[0]

    # Body
    for kp in range(num_body):
        xs = np.array([poses_seq[f]['bodies']['candidate'][kp, 0]
                        for f in range(n)])
        ys = np.array([poses_seq[f]['bodies']['candidate'][kp, 1]
                        for f in range(n)])
        valid = np.array([
            is_valid_kp(poses_seq[f]['bodies']['candidate'][kp])
            for f in range(n)])
        if not np.any(valid):
            continue
        xs = _interp_1d_sequence(xs, valid)
        ys = _interp_1d_sequence(ys, valid)
        for f in range(n):
            kp_val = poses_seq[f]['bodies']['candidate'][kp]
            if not is_valid_kp(kp_val):
                poses_seq[f]['bodies']['candidate'][kp] = np.array(
                    [xs[f], ys[f]])

    # Hands (2, 21, 2)
    num_hands = poses_seq[0]['hands'].shape[0]
    num_hand_kps = poses_seq[0]['hands'].shape[1]
    for hi in range(num_hands):
        for kp in range(num_hand_kps):
            xs = np.array([poses_seq[f]['hands'][hi, kp, 0]
                            for f in range(n)])
            ys = np.array([poses_seq[f]['hands'][hi, kp, 1]
                            for f in range(n)])
            valid = np.array([
                is_valid_kp(poses_seq[f]['hands'][hi, kp])
                for f in range(n)])
            if not np.any(valid):
                continue
            xs = _interp_1d_sequence(xs, valid)
            ys = _interp_1d_sequence(ys, valid)
            for f in range(n):
                if not is_valid_kp(poses_seq[f]['hands'][hi, kp]):
                    poses_seq[f]['hands'][hi, kp] = np.array(
                        [xs[f], ys[f]])

    # Faces (num_persons, 68, 2)
    num_faces = poses_seq[0]['faces'].shape[0]
    num_face_kps = poses_seq[0]['faces'].shape[1]
    for fi in range(num_faces):
        for kp in range(num_face_kps):
            xs = np.array([poses_seq[f]['faces'][fi, kp, 0]
                            for f in range(n)])
            ys = np.array([poses_seq[f]['faces'][fi, kp, 1]
                            for f in range(n)])
            valid = np.array([
                is_valid_kp(poses_seq[f]['faces'][fi, kp])
                for f in range(n)])
            if not np.any(valid):
                continue
            xs = _interp_1d_sequence(xs, valid)
            ys = _interp_1d_sequence(ys, valid)
            for f in range(n):
                if not is_valid_kp(poses_seq[f]['faces'][fi, kp]):
                    poses_seq[f]['faces'][fi, kp] = np.array(
                        [xs[f], ys[f]])

    return poses_seq


# =============================================================================
# Improvement 4: retarget hands / face relative to parent joints
# =============================================================================
def retarget_hands(drv_hands, drv_cand, ret_cand, ref_cand):
    """Re-anchor each hand at its retargeted wrist, scale by forearm."""
    ret_hands = drv_hands.copy()

    for hand_idx, wrist_idx in HAND_WRIST_MAP.items():
        drv_wrist = drv_cand[wrist_idx]
        ret_wrist = ret_cand[wrist_idx]
        if not is_valid_kp(drv_wrist) or not is_valid_kp(ret_wrist):
            continue

        elbow_idx = WRIST_ELBOW_MAP[wrist_idx]
        drv_forearm = bone_length(drv_cand[elbow_idx],
                                  drv_cand[wrist_idx])
        ref_forearm = bone_length(ref_cand[elbow_idx],
                                  ref_cand[wrist_idx])
        if drv_forearm > 1e-6:
            hand_scale = ref_forearm / drv_forearm
        else:
            hand_scale = 1.0

        for kp in range(drv_hands.shape[1]):
            if is_valid_kp(drv_hands[hand_idx, kp]):
                rel = drv_hands[hand_idx, kp] - drv_wrist
                ret_hands[hand_idx, kp] = ret_wrist + rel * hand_scale

    return ret_hands


def retarget_face(drv_faces, drv_cand, ret_cand, ref_cand):
    """Re-anchor face at retargeted nose, scale by head proportion."""
    ret_faces = drv_faces.copy()
    nose_idx = 0
    neck_idx = 1

    drv_nose = drv_cand[nose_idx]
    ret_nose = ret_cand[nose_idx]
    if not is_valid_kp(drv_nose) or not is_valid_kp(ret_nose):
        return ret_faces

    drv_head = bone_length(drv_cand[neck_idx], drv_cand[nose_idx])
    ref_head = bone_length(ref_cand[neck_idx], ref_cand[nose_idx])
    scale = (ref_head / drv_head) if drv_head > 1e-6 else 1.0

    for fi in range(drv_faces.shape[0]):
        for kp in range(drv_faces.shape[1]):
            if is_valid_kp(drv_faces[fi, kp]):
                rel = drv_faces[fi, kp] - drv_nose
                ret_faces[fi, kp] = ret_nose + rel * scale

    return ret_faces


# =============================================================================
# Improvement 5: ground-plane / foot contact constraints
# =============================================================================
def detect_foot_contacts(poses_seq, vel_threshold=0.002):
    """Mark frames where a foot is in ground contact (low velocity
    near a local y-maximum, i.e. the lowest spatial position)."""
    n = len(poses_seq)
    left_contact = np.zeros(n, dtype=bool)
    right_contact = np.zeros(n, dtype=bool)

    for foot_idx, contact_arr in [(18, left_contact),
                                  (19, right_contact)]:
        ys = []
        for f in range(n):
            kp = poses_seq[f]['bodies']['candidate'][foot_idx]
            ys.append(kp[1] if is_valid_kp(kp) else None)

        for f in range(1, n - 1):
            if (ys[f] is None or ys[f - 1] is None
                    or ys[f + 1] is None):
                continue
            vel = abs(ys[f + 1] - ys[f - 1]) / 2.0
            is_low = ys[f] >= ys[f - 1] and ys[f] >= ys[f + 1]
            if vel < vel_threshold and is_low:
                contact_arr[f] = True

    return left_contact, right_contact


def apply_ground_constraints(poses_seq, left_c, right_c):
    """Pin feet to a median ground-plane y during contact frames."""
    n = len(poses_seq)
    ground_ys = []
    for f in range(n):
        if left_c[f]:
            kp = poses_seq[f]['bodies']['candidate'][18]
            if is_valid_kp(kp):
                ground_ys.append(kp[1])
        if right_c[f]:
            kp = poses_seq[f]['bodies']['candidate'][19]
            if is_valid_kp(kp):
                ground_ys.append(kp[1])

    if len(ground_ys) == 0:
        return poses_seq

    ground_y = float(np.median(ground_ys))

    for f in range(n):
        if left_c[f]:
            foot = poses_seq[f]['bodies']['candidate'][18]
            if is_valid_kp(foot):
                dy = ground_y - foot[1]
                poses_seq[f]['bodies']['candidate'][18][1] = ground_y
                ankle = poses_seq[f]['bodies']['candidate'][13]
                if is_valid_kp(ankle):
                    ankle[1] += dy * 0.5
        if right_c[f]:
            foot = poses_seq[f]['bodies']['candidate'][19]
            if is_valid_kp(foot):
                dy = ground_y - foot[1]
                poses_seq[f]['bodies']['candidate'][19][1] = ground_y
                ankle = poses_seq[f]['bodies']['candidate'][10]
                if is_valid_kp(ankle):
                    ankle[1] += dy * 0.5

    return poses_seq


# =============================================================================
# Improvement 6: per-frame adaptive scale (depth compensation)
# =============================================================================
def per_frame_scale(drv_cand, base_cand):
    """Ratio of shoulder width: current frame vs base frame.
    Values > 1 mean the person moved closer to the camera."""
    if not (is_valid_kp(drv_cand[2]) and is_valid_kp(drv_cand[5])
            and is_valid_kp(base_cand[2])
            and is_valid_kp(base_cand[5])):
        return 1.0
    drv_w = bone_length(drv_cand[2], drv_cand[5])
    base_w = bone_length(base_cand[2], base_cand[5])
    if base_w < 1e-6:
        return 1.0
    return float(np.clip(drv_w / base_w, 0.5, 2.0))


# =============================================================================
# Improvement 7: two-anchor root position (hip-center driven)
# =============================================================================
def compute_root_position(drv_cand, ref_cand, base_drv_cand):
    """
    Derive neck position via hip-center displacement.

    1. Hip displacement = current_hip - base_hip.
    2. Scale by (ref_torso / base_torso).
    3. New hip = ref_hip + scaled displacement.
    4. New neck = new_hip + torso_direction * ref_torso_length.

    This allows natural lateral sway and vertical bounce.
    """
    drv_hip = 0.5 * (drv_cand[8] + drv_cand[11])
    base_hip = 0.5 * (base_drv_cand[8] + base_drv_cand[11])
    ref_hip = 0.5 * (ref_cand[8] + ref_cand[11])

    hips_ok = (is_valid_kp(drv_cand[8])
               and is_valid_kp(drv_cand[11])
               and is_valid_kp(base_drv_cand[8])
               and is_valid_kp(base_drv_cand[11])
               and is_valid_kp(ref_cand[8])
               and is_valid_kp(ref_cand[11]))

    if not hips_ok:
        return ref_cand[1].copy()

    ref_torso_len = bone_length(ref_cand[1], ref_hip)
    base_torso_len = bone_length(base_drv_cand[1], base_hip)
    if base_torso_len > 1e-6:
        torso_scale = ref_torso_len / base_torso_len
    else:
        torso_scale = 1.0

    hip_disp = (drv_hip - base_hip) * torso_scale
    new_hip = ref_hip + hip_disp

    drv_torso_dir = normalize_vec(drv_cand[1] - drv_hip)
    if np.linalg.norm(drv_torso_dir) < 1e-8:
        drv_torso_dir = normalize_vec(ref_cand[1] - ref_hip)

    new_neck = new_hip + drv_torso_dir * ref_torso_len
    return new_neck


# =============================================================================
# Improvement 8: physical plausibility checks
# =============================================================================
def _propagate_downstream(candidate, root_joint, offset):
    """Shift all downstream joints in the kinematic tree by offset."""
    for p, ch in KINEMATIC_CHAINS:
        if p == root_joint and is_valid_kp(candidate[ch]):
            candidate[ch] += offset
            _propagate_downstream(candidate, ch, offset)


def validate_pose(candidate):
    """Clamp to canvas [0,1] and enforce joint-angle limits."""
    # Boundary clamping
    for i in range(candidate.shape[0]):
        if is_valid_kp(candidate[i]):
            candidate[i][0] = np.clip(candidate[i][0], 0.0, 1.0)
            candidate[i][1] = np.clip(candidate[i][1], 0.0, 1.0)

    # Joint-angle limits
    for parent, joint, child, min_a, max_a in ANGLE_LIMIT_CONFIGS:
        p = candidate[parent]
        j = candidate[joint]
        c = candidate[child]
        if not (is_valid_kp(p) and is_valid_kp(j) and is_valid_kp(c)):
            continue

        v1 = p - j
        v2 = c - j
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue

        ang = angle_between(v1, v2)
        if min_a <= ang <= max_a:
            continue

        target = min_a if ang < min_a else max_a
        delta = math.radians(target - ang)

        v2_unit = normalize_vec(v2)
        rotated = rotate_2d(v2_unit, delta)
        new_child = j + rotated * n2
        offset = new_child - candidate[child]
        candidate[child] = new_child

        _propagate_downstream(candidate, child, offset)

    return candidate


# =============================================================================
# Temporal smoothing (One Euro Filter on every keypoint trajectory)
# =============================================================================
def apply_temporal_smoothing(poses_seq, fps=30.0, min_cutoff=1.7,
                             beta=0.3):
    """Apply One Euro Filter to every keypoint x/y trajectory."""
    n = len(poses_seq)
    if n < 2:
        return poses_seq
    dt = 1.0 / fps

    def _smooth(get_fn, set_fn, num_kps):
        for kp_i in range(num_kps):
            first = None
            for f in range(n):
                if is_valid_kp(get_fn(f, kp_i)):
                    first = f
                    break
            if first is None:
                continue
            init = get_fn(first, kp_i)
            fx = OneEuroFilter(first * dt, float(init[0]),
                               min_cutoff=min_cutoff, beta=beta)
            fy = OneEuroFilter(first * dt, float(init[1]),
                               min_cutoff=min_cutoff, beta=beta)
            for f in range(first + 1, n):
                val = get_fn(f, kp_i)
                if is_valid_kp(val):
                    t = f * dt
                    smoothed = np.array([fx(t, float(val[0])),
                                         fy(t, float(val[1]))])
                    set_fn(f, kp_i, smoothed)

    # Body
    nb = poses_seq[0]['bodies']['candidate'].shape[0]

    def body_get(f, k):
        return poses_seq[f]['bodies']['candidate'][k]

    def body_set(f, k, v):
        poses_seq[f]['bodies']['candidate'][k] = v

    _smooth(body_get, body_set, nb)

    # Hands (flatten: hand_idx * kps_per_hand)
    nh = poses_seq[0]['hands'].shape[0]
    nk = poses_seq[0]['hands'].shape[1]

    def hand_get(f, k):
        return poses_seq[f]['hands'][k // nk, k % nk]

    def hand_set(f, k, v):
        poses_seq[f]['hands'][k // nk, k % nk] = v

    _smooth(hand_get, hand_set, nh * nk)

    # Faces
    nfp = poses_seq[0]['faces'].shape[0]
    nfk = poses_seq[0]['faces'].shape[1]

    def face_get(f, k):
        return poses_seq[f]['faces'][k // nfk, k % nfk]

    def face_set(f, k, v):
        poses_seq[f]['faces'][k // nfk, k % nfk] = v

    _smooth(face_get, face_set, nfp * nfk)

    return poses_seq


# =============================================================================
# Main pipeline
# =============================================================================
def mp_main(args):
    # Load video(s)
    if args.source_video_paths.endswith('mp4'):
        video_paths = [args.source_video_paths]
    else:
        video_paths = [
            os.path.join(args.source_video_paths, f)
            for f in sorted(os.listdir(args.source_video_paths))]

    logger.info("Videos to process: {}".format(len(video_paths)))
    logger.info('Loading DWpose model ...')
    dwpose_model = DWposeDetector()

    # Step 1: extract poses from driving video
    results_vis = []
    for i, fpath in enumerate(video_paths):
        logger.info("  [{}/{}] {}".format(i + 1, len(video_paths), fpath))
        cap = cv2.VideoCapture(fpath)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results_vis.append(dwpose_model(frame))
        cap.release()

    logger.info("Total driving frames: {}".format(len(results_vis)))
    if len(results_vis) == 0:
        logger.error("No frames extracted. Exiting.")
        return

    # Step 2: detect poses for reference and video-character images
    ref_frame = cv2.imread(args.ref_name, cv2.IMREAD_COLOR)
    assert ref_frame is not None, \
        "Cannot read reference image: {}".format(args.ref_name)
    pose_ref = dwpose_model(ref_frame)
    ref_cand = pose_ref['bodies']['candidate']

    base_char_image = cv2.imread(args.video_char_image, cv2.IMREAD_COLOR)
    assert base_char_image is not None, \
        "Cannot read video character image: {}".format(args.video_char_image)
    base_char_pose = dwpose_model(base_char_image)
    base_cand = base_char_pose['bodies']['candidate']

    for j in CRITICAL_JOINTS:
        if not is_valid_kp(ref_cand[j]):
            logger.warning(
                "Reference: critical joint {} NOT detected!".format(j))
        if not is_valid_kp(base_cand[j]):
            logger.warning(
                "Video character: critical joint {} NOT detected!".format(j))

    # Step 3: compute reference bone lengths
    ref_bone_lengths = compute_ref_bone_lengths(ref_cand)
    logger.info("Reference bone lengths computed.")

    # Step 4: interpolate missing keypoints (Improvement 3)
    logger.info("Interpolating missing keypoints ...")
    results_vis = interpolate_missing_keypoints(results_vis)

    # Step 5: per-frame retargeting (Improvements 2, 4, 6, 7, 8)
    logger.info("Running angle-based pose retargeting ...")
    base_drv_cand = results_vis[0]['bodies']['candidate'].copy()

    retargeted = []
    for f in range(len(results_vis)):
        drv_cand = results_vis[f]['bodies']['candidate']
        drv_hands = results_vis[f]['hands']
        drv_faces = results_vis[f]['faces']

        # Improvement 6: depth-adaptive normalisation
        scale = per_frame_scale(drv_cand, base_drv_cand)
        scaled_cand = drv_cand.copy()
        if abs(scale - 1.0) > 0.01:
            hip_ctr = 0.5 * (drv_cand[8] + drv_cand[11])
            for j_idx in range(scaled_cand.shape[0]):
                if is_valid_kp(scaled_cand[j_idx]):
                    scaled_cand[j_idx] = (
                        hip_ctr + (scaled_cand[j_idx] - hip_ctr) / scale)

        # Improvement 7: two-anchor root position
        root = compute_root_position(scaled_cand, ref_cand, base_drv_cand)

        # Improvement 2: angle-based body retargeting
        ret_body = retarget_body_angle_based(
            scaled_cand, ref_cand, ref_bone_lengths, root)

        # Improvement 8: physical plausibility
        ret_body = validate_pose(ret_body)

        # Improvement 4: hands and face relative to parent joints
        ret_h = retarget_hands(drv_hands, drv_cand, ret_body, ref_cand)
        ret_f = retarget_face(drv_faces, drv_cand, ret_body, ref_cand)

        retargeted.append({
            'bodies': {
                'candidate': ret_body,
                'subset': results_vis[f]['bodies']['subset'].copy(),
            },
            'hands': ret_h,
            'faces': ret_f,
        })

    # Step 6: ground-plane constraints (Improvement 5)
    logger.info("Applying ground-plane constraints ...")
    lc, rc = detect_foot_contacts(retargeted)
    retargeted = apply_ground_constraints(retargeted, lc, rc)
    logger.info("  foot contacts: left={} frames, right={} frames".format(
        int(np.sum(lc)), int(np.sum(rc))))

    # Step 7: temporal smoothing (Improvement 1)
    logger.info("Applying temporal smoothing (One Euro Filter) ...")
    retargeted = apply_temporal_smoothing(
        retargeted,
        fps=args.fps,
        min_cutoff=args.smooth_min_cutoff,
        beta=args.smooth_beta,
    )

    # Step 8: render and save (images + video)
    save_dir = args.saved_pose_dir
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    render_h, render_w = 768, 512
    video_path = os.path.join(save_dir, "pose_sequence.mp4")
    # Use mp4v (MPEG-4 Part 2) to avoid green-frame artefacts that
    # H.264 fourcc ('avc1'/'H264') can produce with OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        video_path, fourcc, args.fps, (render_w, render_h))

    logger.info("Rendering {} frames to {} ...".format(
        len(retargeted), save_dir))
    for i, pose in enumerate(retargeted):
        wo_face, _ = draw_pose(pose, H=render_h, W=render_w)
        img_path = os.path.join(save_dir, "{:04d}.jpg".format(i))
        cv2.imwrite(img_path, wo_face)
        video_writer.write(wo_face)

    video_writer.release()
    logger.info("Saved video: {}".format(video_path))
    logger.info("Done.")


# =============================================================================
logger = get_logger('dwpose-alignment-improved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Improved DWpose alignment with angle-based retargeting, "
                    "temporal smoothing, occlusion handling, and more.")
    parser.add_argument(
        "--ref_name", type=str, required=True,
        help="Path to the reference character image.")
    parser.add_argument(
        "--video_char_image", type=str, required=True,
        help="Path to a character image from the source video.")
    parser.add_argument(
        "--source_video_paths", type=str, required=True,
        help="Path to source driving video (.mp4) or directory.")
    parser.add_argument(
        "--saved_pose_dir", type=str, required=True,
        help="Output directory for aligned pose images.")
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Video FPS for temporal smoothing (default: 30).")
    parser.add_argument(
        "--smooth_min_cutoff", type=float, default=1.7,
        help="One-Euro min cutoff; higher = less smoothing (default: 1.7).")
    parser.add_argument(
        "--smooth_beta", type=float, default=0.3,
        help="One-Euro beta; higher = less lag on fast motion (default: 0.3).")
    args = parser.parse_args()
    mp_main(args)
