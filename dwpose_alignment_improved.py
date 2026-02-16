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
from dwpose.onnxdet import inference_detector

# Optional SAM dependency for precise visibility masking
_SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    pass


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
# SAM-based person mask extraction (optional)
# =============================================================================
def load_sam_predictor(checkpoint_path, device="cuda"):
    """Load a SAM model and return a SamPredictor.

    The model type (vit_b, vit_l, vit_h) is auto-detected from the
    checkpoint filename.
    """
    if not _SAM_AVAILABLE:
        raise RuntimeError(
            "segment-anything is not installed. "
            "Install with: pip install segment-anything")

    fname = os.path.basename(checkpoint_path).lower()
    if "vit_h" in fname:
        model_type = "vit_h"
    elif "vit_l" in fname:
        model_type = "vit_l"
    else:
        model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return SamPredictor(sam)


def get_person_mask_sam(image_bgr, sam_predictor, person_bbox):
    """Run SAM with a YOLOX-detected person box prompt.

    Parameters
    ----------
    image_bgr    : (H, W, 3) BGR image (as read by cv2).
    sam_predictor : SamPredictor instance.
    person_bbox   : (x1, y1, x2, y2) pixel coordinates.

    Returns
    -------
    mask : (H, W) bool array.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    box_np = np.array(person_bbox).reshape(1, 4)
    masks, _, _ = sam_predictor.predict(
        box=box_np,
        multimask_output=False,
    )
    return masks[0]  # (H, W) bool


# =============================================================================
# Partial-body: visibility region computation
# =============================================================================
def compute_visibility_region_sam(image_bgr, sam_predictor, yolox_session,
                                  margin=0.05):
    """Compute the visible person region using SAM.

    Returns (y_min, y_max, x_min, x_max) in normalised [0, 1] coords.
    """
    H, W = image_bgr.shape[:2]

    bboxes = inference_detector(yolox_session, image_bgr)
    if len(bboxes) == 0:
        return (0.0, 1.0, 0.0, 1.0)

    person_box = bboxes[0]  # most confident person
    mask = get_person_mask_sam(image_bgr, sam_predictor, person_box)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 1.0, 0.0, 1.0)

    y_min = max(0.0, float(ys.min()) / H - margin)
    y_max = min(1.0, float(ys.max()) / H + margin)
    x_min = max(0.0, float(xs.min()) / W - margin)
    x_max = min(1.0, float(xs.max()) / W + margin)
    return (y_min, y_max, x_min, x_max)


def compute_visibility_region_keypoints(ref_cand, margin=0.05):
    """Fallback: determine the visible region from detected keypoints.

    Returns (y_min, y_max, x_min, x_max) in normalised [0, 1] coords.
    """
    valid_xs, valid_ys = [], []
    for j in range(ref_cand.shape[0]):
        if is_valid_kp(ref_cand[j]):
            valid_xs.append(ref_cand[j][0])
            valid_ys.append(ref_cand[j][1])

    if len(valid_xs) == 0:
        return (0.0, 1.0, 0.0, 1.0)

    y_min = max(0.0, min(valid_ys) - margin)
    y_max = min(1.0, max(valid_ys) + margin)
    x_min = max(0.0, min(valid_xs) - margin)
    x_max = min(1.0, max(valid_xs) + margin)
    return (y_min, y_max, x_min, x_max)


# =============================================================================
# Partial-body: coordinate transform (edited ref -> original ref)
# =============================================================================
def compute_coord_transform(orig_ref_cand, edit_ref_cand):
    """Fit an affine mapping: orig = scale * edited + offset.

    Uses common valid keypoints in both skeletons.  Separate scale and
    offset for x and y.  Falls back to identity if fewer than 2 common
    keypoints are found.

    Returns (sx, sy, tx, ty).
    """
    common_x_orig, common_y_orig = [], []
    common_x_edit, common_y_edit = [], []

    for j in range(min(orig_ref_cand.shape[0], edit_ref_cand.shape[0])):
        if is_valid_kp(orig_ref_cand[j]) and is_valid_kp(edit_ref_cand[j]):
            common_x_orig.append(orig_ref_cand[j][0])
            common_y_orig.append(orig_ref_cand[j][1])
            common_x_edit.append(edit_ref_cand[j][0])
            common_y_edit.append(edit_ref_cand[j][1])

    if len(common_x_orig) < 2:
        return (1.0, 1.0, 0.0, 0.0)

    common_x_orig = np.array(common_x_orig)
    common_y_orig = np.array(common_y_orig)
    common_x_edit = np.array(common_x_edit)
    common_y_edit = np.array(common_y_edit)

    sx, tx = np.polyfit(common_x_edit, common_x_orig, 1)
    sy, ty = np.polyfit(common_y_edit, common_y_orig, 1)
    return (float(sx), float(sy), float(tx), float(ty))


# =============================================================================
# Partial-body: apply coordinate transform and visibility mask
# =============================================================================
def _transform_kp(kp, sx, sy, tx, ty):
    """Transform a single (x, y) keypoint."""
    return np.array([sx * kp[0] + tx, sy * kp[1] + ty])


def apply_coord_transform_pose(pose, sx, sy, tx, ty):
    """Transform all keypoints in a pose dict from edited-ref space to
    original-ref space."""
    cand = pose['bodies']['candidate']
    for j in range(cand.shape[0]):
        if is_valid_kp(cand[j]):
            cand[j] = _transform_kp(cand[j], sx, sy, tx, ty)

    hands = pose['hands']
    for hi in range(hands.shape[0]):
        for k in range(hands.shape[1]):
            if is_valid_kp(hands[hi, k]):
                hands[hi, k] = _transform_kp(hands[hi, k], sx, sy, tx, ty)

    faces = pose['faces']
    for fi in range(faces.shape[0]):
        for k in range(faces.shape[1]):
            if is_valid_kp(faces[fi, k]):
                faces[fi, k] = _transform_kp(faces[fi, k], sx, sy, tx, ty)

    return pose


def compute_visible_joint_set(orig_ref_cand, propagate=True):
    """Determine which joint types should be visible based on the original
    reference pose.

    Parameters
    ----------
    orig_ref_cand : (20, 2) array of keypoints from the original reference.
    propagate     : bool.  When True (SAM path), propagate through kinematic
                    chains so downstream children of detected joints are also
                    visible.  When False (no-SAM fallback), only directly
                    detected joints are visible — no kinematic expansion.

    Returns a set of visible joint indices.
    """
    visible = set()
    for j in range(orig_ref_cand.shape[0]):
        if is_valid_kp(orig_ref_cand[j]):
            visible.add(j)

    if propagate:
        changed = True
        while changed:
            changed = False
            for parent, child in KINEMATIC_CHAINS:
                if parent in visible and child not in visible:
                    visible.add(child)
                    changed = True

    return visible


def _mask_joint(cand, subset, j):
    """Set joint j to invalid (-1) in both candidate and subset."""
    cand[j] = np.array([-1.0, -1.0])
    for s in range(subset.shape[0]):
        if int(subset[s][j]) == j:
            subset[s][j] = -1


def apply_visibility_mask(pose, visible_region, visible_joints=None):
    """Mask out keypoints that should not be rendered.

    Strategies applied to **body joints** (in order):
    1. **Extreme-outlier guard**: joints farther than 1 canvas-width
       off-screen (outside [−1, 2]) are masked to avoid numerical issues.
       Joints that are merely slightly off-canvas are *kept* so that
       OpenCV can draw the connecting limb up to the canvas edge.
    2. **Joint-type visibility** (``visible_joints``): if provided, any
       joint type NOT in the set is masked.
    3. **Y-axis spatial clipping** (``visible_region``): joints whose
       y-coordinate falls outside [y_min, y_max] are masked.

    **Hand / face keypoints** use strict [0, 1] canvas-bounds clipping
    to avoid partial hand/face artifacts at the edges.

    Parameters
    ----------
    pose           : dict with 'bodies', 'hands', 'faces'.
    visible_region : (y_min, y_max, x_min, x_max) in normalised coords.
                     Only y_min and y_max are used for spatial clipping.
    visible_joints : set of int, optional.  Joint indices that are allowed.
    """
    y_min, y_max = visible_region[0], visible_region[1]
    cand = pose['bodies']['candidate']
    subset = pose['bodies']['subset']

    # Generous body-joint bounds: allow joints to extend up to 1 canvas-
    # width off-screen so limb connections are preserved and OpenCV clips
    # the drawn lines naturally at the canvas edge.
    BODY_LO, BODY_HI = -1.0, 2.0

    for j in range(cand.shape[0]):
        if not is_valid_kp(cand[j]):
            continue

        x, y = cand[j][0], cand[j][1]

        # Strategy 1: extreme-outlier guard for body joints
        if x < BODY_LO or x > BODY_HI or y < BODY_LO or y > BODY_HI:
            _mask_joint(cand, subset, j)
            continue

        # Strategy 2: joint-type visibility
        if visible_joints is not None and j not in visible_joints:
            _mask_joint(cand, subset, j)
            continue

        # Strategy 3: Y-axis spatial clipping (with same generous margin)
        if y < y_min - 1.0 or y > y_max + 1.0:
            _mask_joint(cand, subset, j)

    # Mask hands: strict [0, 1] canvas bounds for individual keypoints
    for hand_idx, wrist_idx in HAND_WRIST_MAP.items():
        if not is_valid_kp(cand[wrist_idx]):
            pose['hands'][hand_idx] = -1.0
        else:
            for k in range(pose['hands'].shape[1]):
                kp = pose['hands'][hand_idx, k]
                if is_valid_kp(kp):
                    if kp[0] < 0 or kp[0] > 1 or kp[1] < 0 or kp[1] > 1:
                        pose['hands'][hand_idx, k] = np.array([-1.0, -1.0])

    # Mask face: strict [0, 1] canvas bounds for individual keypoints
    if not is_valid_kp(cand[0]):
        pose['faces'][:] = -1.0
    else:
        for fi in range(pose['faces'].shape[0]):
            for k in range(pose['faces'].shape[1]):
                kp = pose['faces'][fi, k]
                if is_valid_kp(kp):
                    if kp[0] < 0 or kp[0] > 1 or kp[1] < 0 or kp[1] > 1:
                        pose['faces'][fi, k] = np.array([-1.0, -1.0])

    return pose


# =============================================================================
# Motion attenuation for partial-body / close-up references
# =============================================================================
def _shift_all_keypoints(pose, dx, dy):
    """Shift every valid keypoint in a pose dict by (dx, dy)."""
    cand = pose['bodies']['candidate']
    for j in range(cand.shape[0]):
        if is_valid_kp(cand[j]):
            cand[j][0] += dx
            cand[j][1] += dy
    for hi in range(pose['hands'].shape[0]):
        for k in range(pose['hands'].shape[1]):
            if is_valid_kp(pose['hands'][hi, k]):
                pose['hands'][hi, k][0] += dx
                pose['hands'][hi, k][1] += dy
    for fi in range(pose['faces'].shape[0]):
        for k in range(pose['faces'].shape[1]):
            if is_valid_kp(pose['faces'][fi, k]):
                pose['faces'][fi, k][0] += dx
                pose['faces'][fi, k][1] += dy


def _get_anchor(cand):
    """Return the best anchor position: nose > neck > hip center."""
    if is_valid_kp(cand[0]):
        return cand[0].copy()
    if is_valid_kp(cand[1]):
        return cand[1].copy()
    if is_valid_kp(cand[8]) and is_valid_kp(cand[11]):
        return 0.5 * (cand[8] + cand[11])
    return None


def attenuate_motion(retargeted, sx, sy, ref_anchor):
    """De-amplify global motion caused by coordinate-transform scaling.

    When the coord transform maps from edited-ref space to original-ref
    space, it multiplies all positions by (sx, sy).  This also amplifies
    frame-to-frame motion: a 2% sway at 5x scale becomes 10%.  For
    close-up references this pushes the skeleton off-canvas.

    This function:
    1. Anchors the first frame to ``ref_anchor`` (position correction).
    2. For subsequent frames, keeps only 1/|scale| of the per-frame
       displacement relative to frame 0, preserving the *original*
       motion magnitude while removing the amplification.

    Result: head tilts, expressions, and subtle sway are preserved at
    their natural scale; the skeleton stays centred on the canvas.

    Parameters
    ----------
    retargeted  : list of pose dicts (already coord-transformed).
    sx, sy      : float, coordinate-transform scale factors.
    ref_anchor  : (2,) target anchor position in original-ref space.
    """
    if len(retargeted) == 0:
        return retargeted

    scale_mag = max(abs(sx), abs(sy), 1.0)

    # Motion retention factor: inverse of scale, clamped to [0.1, 1.0].
    # scale=1 → keep 100%; scale=5 → keep 20%; scale=10 → keep 10%.
    motion_retain = np.clip(1.0 / scale_mag, 0.1, 1.0)

    # Step 1: anchor frame 0 to ref_anchor
    f0_cand = retargeted[0]['bodies']['candidate']
    f0_anchor = _get_anchor(f0_cand)
    if f0_anchor is None:
        return retargeted

    initial_offset = ref_anchor - f0_anchor
    _shift_all_keypoints(retargeted[0], initial_offset[0], initial_offset[1])

    # Record the corrected frame-0 anchor
    f0_anchor_corrected = ref_anchor.copy()

    # Step 2: for each subsequent frame, attenuate displacement from frame 0
    for f in range(1, len(retargeted)):
        cand = retargeted[f]['bodies']['candidate']
        anchor = _get_anchor(cand)
        if anchor is None:
            _shift_all_keypoints(retargeted[f],
                                 initial_offset[0], initial_offset[1])
            continue

        # Raw displacement of this frame's anchor from frame 0's anchor
        # (both still in un-corrected transformed space).
        raw_disp = anchor - f0_anchor

        # The transform amplified this displacement by ~scale_mag.
        # Keep only 1/scale_mag of it to restore the original magnitude.
        desired_disp = raw_disp * motion_retain

        # Target position = corrected frame-0 anchor + attenuated motion
        target = f0_anchor_corrected + desired_disp
        shift = target - anchor
        _shift_all_keypoints(retargeted[f], shift[0], shift[1])

    return retargeted


# =============================================================================
# Position correction (Issue 2: character position mismatch)
# =============================================================================
def apply_position_correction(retargeted_seq, target_ref_cand):
    """Shift all frames so the first frame's anchor matches the reference.

    Uses neck (joint 1) as the primary anchor; falls back to hip center
    if neck is not detected in the reference.
    """
    if len(retargeted_seq) == 0:
        return retargeted_seq

    # Determine anchor joint from reference
    if is_valid_kp(target_ref_cand[1]):
        ref_anchor = target_ref_cand[1].copy()
        anchor_idx = 1
    elif (is_valid_kp(target_ref_cand[8])
          and is_valid_kp(target_ref_cand[11])):
        ref_anchor = 0.5 * (target_ref_cand[8] + target_ref_cand[11])
        anchor_idx = None
    else:
        return retargeted_seq

    # Compute anchor from first retargeted frame
    first_cand = retargeted_seq[0]['bodies']['candidate']
    if anchor_idx is not None:
        if not is_valid_kp(first_cand[anchor_idx]):
            return retargeted_seq
        ret_anchor = first_cand[anchor_idx].copy()
    else:
        if (not is_valid_kp(first_cand[8])
                or not is_valid_kp(first_cand[11])):
            return retargeted_seq
        ret_anchor = 0.5 * (first_cand[8] + first_cand[11])

    offset = ref_anchor - ret_anchor
    if np.linalg.norm(offset) < 1e-6:
        return retargeted_seq

    for pose in retargeted_seq:
        _shift_all_keypoints(pose, offset[0], offset[1])

    return retargeted_seq


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
# Auto max_bone_ratio: compute from skeleton scale difference
# =============================================================================
def auto_max_bone_ratio(ref_cand, drv_first_cand, floor=1.3, ceiling=3.0):
    """Estimate a suitable max_bone_ratio from the overall scale difference
    between the reference and the driving character's first frame.

    Compares multiple body metrics (shoulder width, torso length, arm span)
    and uses the median ratio with a safety margin.

    Returns a float clamped to [floor, ceiling].
    """
    ratios = []

    # Shoulder width
    for a, b in [(2, 5)]:
        if (is_valid_kp(ref_cand[a]) and is_valid_kp(ref_cand[b])
                and is_valid_kp(drv_first_cand[a])
                and is_valid_kp(drv_first_cand[b])):
            ref_d = bone_length(ref_cand[a], ref_cand[b])
            drv_d = bone_length(drv_first_cand[a], drv_first_cand[b])
            if drv_d > 1e-6:
                ratios.append(ref_d / drv_d)

    # Torso length (neck to hip-center)
    ref_hip_ok = is_valid_kp(ref_cand[8]) and is_valid_kp(ref_cand[11])
    drv_hip_ok = (is_valid_kp(drv_first_cand[8])
                  and is_valid_kp(drv_first_cand[11]))
    if (is_valid_kp(ref_cand[1]) and ref_hip_ok
            and is_valid_kp(drv_first_cand[1]) and drv_hip_ok):
        ref_hip = 0.5 * (ref_cand[8] + ref_cand[11])
        drv_hip = 0.5 * (drv_first_cand[8] + drv_first_cand[11])
        ref_d = bone_length(ref_cand[1], ref_hip)
        drv_d = bone_length(drv_first_cand[1], drv_hip)
        if drv_d > 1e-6:
            ratios.append(ref_d / drv_d)

    # Individual bone lengths
    for parent, child in KINEMATIC_CHAINS:
        rp, rc = ref_cand[parent], ref_cand[child]
        dp, dc = drv_first_cand[parent], drv_first_cand[child]
        if (is_valid_kp(rp) and is_valid_kp(rc)
                and is_valid_kp(dp) and is_valid_kp(dc)):
            ref_d = bone_length(rp, rc)
            drv_d = bone_length(dp, dc)
            if drv_d > 1e-6 and ref_d > 1e-6:
                ratios.append(ref_d / drv_d)

    if len(ratios) == 0:
        return ceiling

    median_ratio = float(np.median(ratios))
    # Use 1.3x the median as the max ratio (some headroom for variation)
    auto_val = max(median_ratio, 1.0 / median_ratio) * 1.3
    result = float(np.clip(auto_val, floor, ceiling))
    return result


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
                              ref_bone_lengths, root_pos,
                              max_bone_ratio=1.5):
    """
    Reconstruct skeleton using driving-pose directions + **clamped**
    reference bone lengths.

    For each bone (parent -> child) in the kinematic chain:
      direction  = normalize(driving_child - driving_parent)
      ratio      = clamp(ref_len / drv_len, 1/max_bone_ratio, max_bone_ratio)
      used_len   = drv_len * ratio
      child_pos  = retargeted_parent + direction * used_len

    The ratio clamping prevents unrealistically stretched or compressed
    limbs when the reference and driving characters have very different
    body proportions.

    Parameters
    ----------
    driving_cand     : (20, 2) driving frame keypoints (normalised).
    ref_cand         : (20, 2) reference character keypoints.
    ref_bone_lengths : dict (parent, child) -> float.
    root_pos         : (2,) neck position for kinematic root.
    max_bone_ratio   : float, maximum allowed bone-length ratio
                       between reference and driving (default 1.5).

    Returns
    -------
    retargeted : (20, 2) array.
    """
    retargeted = driving_cand.copy()
    retargeted[1] = root_pos

    inv_ratio = 1.0 / max_bone_ratio

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
        drv_len = bone_length(drv_p, drv_c)

        if ref_len is not None and drv_len > 1e-6:
            ratio = np.clip(ref_len / drv_len, inv_ratio, max_bone_ratio)
            used_len = drv_len * ratio
        elif ref_len is not None:
            used_len = ref_len
        else:
            used_len = drv_len

        retargeted[child] = retargeted[parent] + direction * used_len

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
def retarget_hands(drv_hands, drv_cand, ret_cand, ref_cand,
                   max_bone_ratio=1.5):
    """Re-anchor each hand at its retargeted wrist, scale by forearm.

    The scale ratio is clamped to [1/max_bone_ratio, max_bone_ratio] to
    prevent unrealistically large or small hands.
    """
    ret_hands = drv_hands.copy()
    inv_ratio = 1.0 / max_bone_ratio

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
            hand_scale = np.clip(ref_forearm / drv_forearm,
                                 inv_ratio, max_bone_ratio)
        else:
            hand_scale = 1.0

        for kp in range(drv_hands.shape[1]):
            if is_valid_kp(drv_hands[hand_idx, kp]):
                rel = drv_hands[hand_idx, kp] - drv_wrist
                ret_hands[hand_idx, kp] = ret_wrist + rel * hand_scale

    return ret_hands


def retarget_face(drv_faces, drv_cand, ret_cand, ref_cand,
                  max_bone_ratio=1.5):
    """Re-anchor face at retargeted nose, scale by head proportion.

    The scale ratio is clamped to [1/max_bone_ratio, max_bone_ratio] to
    prevent unrealistically large or small faces.
    """
    ret_faces = drv_faces.copy()
    nose_idx = 0
    neck_idx = 1

    drv_nose = drv_cand[nose_idx]
    ret_nose = ret_cand[nose_idx]
    if not is_valid_kp(drv_nose) or not is_valid_kp(ret_nose):
        return ret_faces

    drv_head = bone_length(drv_cand[neck_idx], drv_cand[nose_idx])
    ref_head = bone_length(ref_cand[neck_idx], ref_cand[nose_idx])
    inv_ratio = 1.0 / max_bone_ratio
    scale = (np.clip(ref_head / drv_head, inv_ratio, max_bone_ratio)
             if drv_head > 1e-6 else 1.0)

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
    Values > 1 mean the person moved closer to the camera.

    Clamped to [0.75, 1.35] to prevent over-correction that can
    distort poses when the shoulder detection jitters.
    """
    if not (is_valid_kp(drv_cand[2]) and is_valid_kp(drv_cand[5])
            and is_valid_kp(base_cand[2])
            and is_valid_kp(base_cand[5])):
        return 1.0
    drv_w = bone_length(drv_cand[2], drv_cand[5])
    base_w = bone_length(base_cand[2], base_cand[5])
    if base_w < 1e-6:
        return 1.0
    return float(np.clip(drv_w / base_w, 0.75, 1.35))


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
    partial_body_mode = (hasattr(args, 'edited_ref_name')
                         and args.edited_ref_name)

    ref_frame = cv2.imread(args.ref_name, cv2.IMREAD_COLOR)
    assert ref_frame is not None, \
        "Cannot read reference image: {}".format(args.ref_name)
    pose_orig_ref = dwpose_model(ref_frame)
    orig_ref_cand = pose_orig_ref['bodies']['candidate']

    if partial_body_mode:
        logger.info("Partial-body mode: using edited ref for retargeting.")
        edit_ref_frame = cv2.imread(args.edited_ref_name, cv2.IMREAD_COLOR)
        assert edit_ref_frame is not None, \
            "Cannot read edited reference image: {}".format(
                args.edited_ref_name)
        pose_edit_ref = dwpose_model(edit_ref_frame)
        edit_ref_cand = pose_edit_ref['bodies']['candidate']

        # Use the edited (full-body) ref for retargeting
        ref_cand = edit_ref_cand

        # Coordinate transform: edited ref space -> original ref space
        coord_tf = compute_coord_transform(orig_ref_cand, edit_ref_cand)
        logger.info("Coord transform (edited->orig): "
                     "sx={:.3f} sy={:.3f} tx={:.3f} ty={:.3f}".format(
                         *coord_tf))

        # Visibility region
        vis_margin = getattr(args, 'visibility_margin', 0.05)
        sam_ckpt = getattr(args, 'sam_checkpoint', None)
        using_sam = False
        if sam_ckpt and _SAM_AVAILABLE:
            logger.info("Loading SAM from {} ...".format(sam_ckpt))
            sam_pred = load_sam_predictor(sam_ckpt)
            yolox_sess = dwpose_model.pose_estimation.session_det
            visible_region = compute_visibility_region_sam(
                ref_frame, sam_pred, yolox_sess, margin=vis_margin)
            logger.info("Visible region (SAM): y=[{:.3f},{:.3f}] "
                         "x=[{:.3f},{:.3f}]".format(*visible_region))
            del sam_pred
            torch.cuda.empty_cache()
            using_sam = True
        else:
            if sam_ckpt and not _SAM_AVAILABLE:
                logger.warning("SAM checkpoint provided but "
                               "segment-anything is not installed. "
                               "Falling back to keypoint-based visibility.")
            visible_region = compute_visibility_region_keypoints(
                orig_ref_cand, margin=vis_margin)
            logger.info("Visible region (keypoints): y=[{:.3f},{:.3f}] "
                         "x=[{:.3f},{:.3f}]".format(*visible_region))

        # Kinematic-chain-based joint visibility.
        # SAM path: propagate through kinematic chains (arms extend
        # from shoulders even if not in the original ref).
        # No-SAM path: only directly detected joints are visible.
        visible_joints = compute_visible_joint_set(
            orig_ref_cand, propagate=using_sam)
        logger.info("Visible joints ({}): {}".format(
            "propagated" if using_sam else "detected-only",
            sorted(visible_joints)))
    else:
        ref_cand = orig_ref_cand
        coord_tf = None
        visible_region = None
        visible_joints = None

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

    # Auto max_bone_ratio: when set to 0, compute from skeleton scale diff
    if args.max_bone_ratio <= 0:
        mbr = auto_max_bone_ratio(ref_cand, base_drv_cand)
        logger.info("Auto max_bone_ratio: {:.2f}".format(mbr))
    else:
        mbr = args.max_bone_ratio
        logger.info("Using max_bone_ratio: {:.2f}".format(mbr))

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
        ret_body = retarget_body_angle_based(
            scaled_cand, ref_cand, ref_bone_lengths, root,
            max_bone_ratio=mbr)

        # Improvement 8: physical plausibility
        ret_body = validate_pose(ret_body)

        # Improvement 4: hands and face relative to parent joints
        ret_h = retarget_hands(drv_hands, drv_cand, ret_body, ref_cand,
                               max_bone_ratio=mbr)
        ret_f = retarget_face(drv_faces, drv_cand, ret_body, ref_cand,
                              max_bone_ratio=mbr)

        retargeted.append({
            'bodies': {
                'candidate': ret_body,
                'subset': results_vis[f]['bodies']['subset'].copy(),
            },
            'hands': ret_h,
            'faces': ret_f,
        })

    # Step 5.5: partial-body coord transform + motion attenuation
    if partial_body_mode and coord_tf is not None:
        sx, sy, tx, ty = coord_tf
        logger.info("Applying coordinate transform: sx={:.2f}, sy={:.2f}, "
                     "tx={:.2f}, ty={:.2f}".format(sx, sy, tx, ty))

        # Apply coordinate transform to all frames
        for pose in retargeted:
            apply_coord_transform_pose(pose, sx, sy, tx, ty)

        # Compute reference anchor in original-ref space
        ref_anchor = _get_anchor(orig_ref_cand)
        if ref_anchor is not None:
            scale_mag = max(abs(sx), abs(sy), 1.0)
            if scale_mag > 1.5:
                logger.info("Motion attenuation: scale_mag={:.1f}, "
                            "retaining {:.0f}% of global motion".format(
                                scale_mag, 100.0 / scale_mag))
            retargeted = attenuate_motion(retargeted, sx, sy, ref_anchor)
        else:
            logger.warning("No valid anchor in original ref; "
                           "skipping motion attenuation.")

        # Apply visibility mask AFTER motion attenuation (final positions)
        if visible_region is not None:
            for pose in retargeted:
                apply_visibility_mask(pose, visible_region,
                                      visible_joints=visible_joints)
    else:
        # Standard mode: simple position correction
        logger.info("Applying position correction ...")
        retargeted = apply_position_correction(retargeted, orig_ref_cand)

    # Step 6: ground-plane constraints (Improvement 5)
    # Skip if lower body is not visible (partial-body mode)
    lower_body_visible = (is_valid_kp(orig_ref_cand[10])
                          or is_valid_kp(orig_ref_cand[13])
                          or is_valid_kp(orig_ref_cand[18])
                          or is_valid_kp(orig_ref_cand[19]))
    if lower_body_visible:
        logger.info("Applying ground-plane constraints ...")
        lc, rc = detect_foot_contacts(retargeted)
        retargeted = apply_ground_constraints(retargeted, lc, rc)
        logger.info("  foot contacts: left={} frames, right={} frames".format(
            int(np.sum(lc)), int(np.sum(rc))))
    else:
        logger.info("Skipping ground constraints (lower body not visible).")

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

    logger.info("Rendering {} frames to {} ...".format(
        len(retargeted), save_dir))
    for i, pose in enumerate(retargeted):
        wo_face, _ = draw_pose(pose, H=render_h, W=render_w)
        img_path = os.path.join(save_dir, "{:04d}.jpg".format(i))
        cv2.imwrite(img_path, wo_face)

    video_path = os.path.join(save_dir, "pose_sequence.mp4")
    ffmpeg_cmd = (
        'ffmpeg -y -framerate {} -i {}/%04d.jpg '
        '-c:v libx264 -pix_fmt yuv420p -crf 18 {}'
    ).format(args.fps, save_dir, video_path)
    logger.info("Encoding video with ffmpeg ...")
    os.system(ffmpeg_cmd)
    logger.info("Saved video: {}".format(video_path))

    if getattr(args, 'video_only', False):
        for fn in os.listdir(save_dir):
            if fn.endswith('.jpg'):
                os.remove(os.path.join(save_dir, fn))
        logger.info("--video_only: removed individual frame images.")

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
    parser.add_argument(
        "--max_bone_ratio", type=float, default=0,
        help="Max allowed bone-length ratio between reference and driving "
             "characters. Set to 0 for automatic detection based on the "
             "actual scale difference between skeletons (recommended). "
             "A positive value (e.g., 1.5) is used as-is (default: 0 = auto).")
    parser.add_argument(
        "--video_only", action="store_true",
        help="Only save the video; delete individual frame images after "
             "encoding.")
    parser.add_argument(
        "--edited_ref_name", type=str, default="",
        help="Path to an edited (full-body) version of the reference image. "
             "When provided, enables partial-body mode: retargeting uses "
             "the full skeleton from this image, then maps poses back to "
             "the original reference's visible region.")
    parser.add_argument(
        "--sam_checkpoint", type=str, default="",
        help="Path to a SAM checkpoint (e.g., sam_vit_b_01ec64.pth) for "
             "precise person-mask visibility detection. Requires "
             "segment-anything to be installed. Falls back to keypoint-based "
             "visibility if not provided.")
    parser.add_argument(
        "--visibility_margin", type=float, default=0.05,
        help="Margin (normalised) added around the detected visible region "
             "in partial-body mode (default: 0.05).")
    args = parser.parse_args()
    mp_main(args)
