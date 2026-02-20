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

# Left/right symmetric body pairs for mirror completion when one side is missing
SYMMETRIC_BODY_PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13),
                        (14, 15), (16, 17), (18, 19)]

# (parent, child) -> symmetric (p2, c2) for direction fallback
SYMMETRIC_CHAIN = {
    (2, 3): (5, 6), (3, 4): (6, 7), (5, 6): (2, 3), (6, 7): (3, 4),
    (1, 8): (1, 11), (8, 9): (11, 12), (9, 10): (12, 13), (10, 19): (13, 18),
    (1, 11): (1, 8), (11, 12): (8, 9), (12, 13): (9, 10), (13, 18): (10, 19),
}

# Default direction (dx, dy) from neck for first-level bones when ref has no data
_DEFAULT_NECK_DIRECTIONS = {
    (1, 0): (0.0, -0.08),   # nose
    (1, 2): (0.10, 0.02),   # R shoulder
    (1, 5): (-0.10, 0.02),  # L shoulder
    (1, 8): (0.06, 0.10),   # R hip
    (1, 11): (-0.06, 0.10), # L hip
}

# View-dependent visibility: joints typically occluded in back or side view
FRONT_ONLY_JOINTS = {0, 14, 15, 16, 17}   # face (occluded when facing away)
LEFT_BODY_JOINTS = {5, 6, 7, 11, 12, 13, 18}   # left arm + left leg
RIGHT_BODY_JOINTS = {2, 3, 4, 8, 9, 10, 19}   # right arm + right leg
NECK_JOINT = 1
WRIST_JOINTS = {4, 7}   # right wrist, left wrist (mask when near head to avoid hand–head connections)
# Distance (normalized) below which wrist is considered "near head" and occluded
WRIST_NEAR_HEAD_THRESHOLD = 0.10
# Hand extent: max distance (normalized) from wrist to any hand keypoint; above = unrealistic, don't draw
HAND_MAX_EXTENT = 0.09
# Typical forearm length (normalized) when inferring wrist from arm
DEFAULT_FOREARM_LEN = 0.07


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
def compute_coord_transform(orig_ref_cand, edit_ref_cand, joint_indices=None):
    """Fit an affine mapping: orig = scale * edited + offset.

    Uses common valid keypoints in both skeletons.  Separate scale and
    offset for x and y.  Falls back to identity if fewer than 2 common
    keypoints are found.

    joint_indices : set or sequence of int, optional.  If provided, only
                    these joint indices are used for the fit.

    Returns (sx, sy, tx, ty).
    """
    common_x_orig, common_y_orig = [], []
    common_x_edit, common_y_edit = [], []

    indices = range(min(orig_ref_cand.shape[0], edit_ref_cand.shape[0]))
    if joint_indices is not None:
        indices = [j for j in indices if j in joint_indices]

    for j in indices:
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


# Neck -> hip edges: do not propagate to lower body unless hips are detected
# in the original reference (avoids showing legs for upper-body-only refs).
_UPPER_LOWER_BRIDGE = {(1, 8), (1, 11)}

# Joints used to fit edit->orig coordinate transform in partial-body mode
# when the ref has no visible upper legs (default: upper-body only).
_UPPER_BODY_JOINTS_FOR_COORD_TF = (0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)


def _kp_inside_region(kp, region):
    """Return True if keypoint (x, y) lies inside (y_min, y_max, x_min, x_max)."""
    if not is_valid_kp(kp):
        return False
    y_min, y_max, x_min, x_max = region[0], region[1], region[2], region[3]
    return (x_min <= kp[0] <= x_max and y_min <= kp[1] <= y_max)


def _coord_transform_joint_indices(orig_ref_cand, edit_ref_cand,
                                   visible_region):
    """Return joint indices for edit->orig coord transform in partial-body mode.

    When the partial-body ref shows upper legs (hips or knees visible in the
    visible region), include hip and knee keypoints so the transform aligns
    them instead of excluding and potentially misplacing them. Otherwise use
    upper-body only to avoid pulling retargeted hips up to biased DWPose hips.
    """
    base = set(_UPPER_BODY_JOINTS_FOR_COORD_TF)
    if visible_region is None:
        return tuple(sorted(base))
    # Check if any upper-leg keypoint (hip or knee) is valid and inside region.
    ref_has_upper_legs = False
    for j in (8, 11, 9, 12):
        if (is_valid_kp(orig_ref_cand[j]) and is_valid_kp(edit_ref_cand[j])
                and _kp_inside_region(orig_ref_cand[j], visible_region)):
            ref_has_upper_legs = True
            break
    if ref_has_upper_legs:
        base.add(8)
        base.add(11)
        if (is_valid_kp(orig_ref_cand[9]) and is_valid_kp(edit_ref_cand[9])
                and _kp_inside_region(orig_ref_cand[9], visible_region)):
            base.add(9)
        if (is_valid_kp(orig_ref_cand[12]) and is_valid_kp(edit_ref_cand[12])
                and _kp_inside_region(orig_ref_cand[12], visible_region)):
            base.add(12)
    return tuple(sorted(base))


def compute_visible_joint_set(orig_ref_cand, propagate=True, visible_region=None):
    """Determine which joint types should be visible based on the original
    reference pose.

    Parameters
    ----------
    orig_ref_cand   : (20, 2) array of keypoints from the original reference.
    propagate       : bool.  When True (SAM path), propagate through kinematic
                      chains so downstream children of detected joints are also
                      visible.  When False (no-SAM fallback), only directly
                      detected joints are visible — no kinematic expansion.
    visible_region  : (y_min, y_max, x_min, x_max) optional.  When provided
                      (e.g. from SAM), a joint is only considered "detected" if
                      it lies inside this region.

    When propagating, the neck->hip links (1->8, 1->11) are only followed
    if the hip joint was directly detected in the reference.  This keeps
    upper-body-only references from showing leg connections.
    """
    detected = set()
    for j in range(orig_ref_cand.shape[0]):
        if not is_valid_kp(orig_ref_cand[j]):
            continue
        if visible_region is not None and not _kp_inside_region(
                orig_ref_cand[j], visible_region):
            continue
        detected.add(j)

    visible = set(detected)

    if propagate:
        changed = True
        while changed:
            changed = False
            for parent, child in KINEMATIC_CHAINS:
                if parent in visible and child not in visible:
                    # Do not propagate from neck to hips unless hips detected
                    if (parent, child) in _UPPER_LOWER_BRIDGE and child not in detected:
                        continue
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

    # Mask hands: strict [0, 1] canvas bounds for individual keypoints only.
    # Do not exclude entire hand when body wrist is invalid (wrist may be resolved later or drawn from hand).
    for hand_idx, wrist_idx in HAND_WRIST_MAP.items():
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


def _ref_lower_body_visible(ref_cand):
    """Return True if the reference has at least one valid lower-body keypoint."""
    for j in (10, 13, 18, 19):  # ankles and feet
        if is_valid_kp(ref_cand[j]):
            return True
    return False


def correct_ref_hips_for_partial_ref(ref_cand, torso_to_shoulder_ratio=1.35):
    """When the ref has no visible lower body, re-estimate hip positions lower.

    DWPose often places hips at the visible bottom (crop/shirt hem), which is
    above true anatomical hip height. This overwrites ref_cand[8] and ref_cand[11]
    with an estimate: neck + downward direction * (torso_length), with
    torso_length = shoulder_width * torso_to_shoulder_ratio.

    Modifies ref_cand in place. Returns True if correction was applied.
    """
    if _ref_lower_body_visible(ref_cand):
        return False
    if not (is_valid_kp(ref_cand[1]) and is_valid_kp(ref_cand[2])
            and is_valid_kp(ref_cand[5])):
        return False
    neck = ref_cand[1]
    shoulder_w = bone_length(ref_cand[2], ref_cand[5])
    if shoulder_w < 1e-6:
        return False
    torso_len = shoulder_w * torso_to_shoulder_ratio
    # y increases downward in normalized coords
    down = np.array([0.0, 1.0])
    hip_center = neck + down * torso_len
    half_w = shoulder_w * 0.5
    ref_cand[8] = np.array([hip_center[0] + half_w, hip_center[1]],
                           dtype=ref_cand.dtype)
    ref_cand[11] = np.array([hip_center[0] - half_w, hip_center[1]],
                            dtype=ref_cand.dtype)
    return True


def apply_face_ref_anchor_partial_body(retargeted_seq, orig_ref_cand,
                                       pose_orig_ref, skip_frame_0=True):
    """Stage 2 partial-body: apply ref_name face size to face (face already moved with body).

    After coord transform, face keypoints already follow the body. Use ref_name
    face keypoints only for size: scale each frame's face around its nose so
    the face extent matches ref_name face extent. Modifies retargeted_seq in place.
    """
    if len(retargeted_seq) == 0:
        return
    if not is_valid_kp(orig_ref_cand[0]):
        return
    ref_extent = _face_extent(pose_orig_ref['faces'], orig_ref_cand[0])
    if ref_extent is None:
        return
    start = 1 if skip_frame_0 else 0
    for f in range(start, len(retargeted_seq)):
        pose = retargeted_seq[f]
        cand = pose['bodies']['candidate']
        nose = cand[0]
        if not is_valid_kp(nose):
            continue
        curr_extent = _face_extent(pose['faces'], nose)
        if curr_extent is None or curr_extent < 1e-6:
            continue
        scale = np.clip(ref_extent / curr_extent, 1.0, 2.0)  # only scale up, never shrink
        for fi in range(pose['faces'].shape[0]):
            for k in range(pose['faces'].shape[1]):
                if is_valid_kp(pose['faces'][fi, k]):
                    offset = pose['faces'][fi, k] - nose
                    pose['faces'][fi, k] = nose + scale * offset


def _joints_inside_region(pose, visible_region):
    """Return set of body joint indices whose keypoints lie inside visible_region."""
    if visible_region is None:
        return set()
    cand = pose['bodies']['candidate']
    inside = set()
    for j in range(cand.shape[0]):
        if is_valid_kp(cand[j]) and _kp_inside_region(cand[j], visible_region):
            inside.add(j)
    return inside


def _kinematic_neighbors():
    """Return dict: joint index -> set of adjacent joint indices (along kinematic chains)."""
    neighbors = {}
    for parent, child in KINEMATIC_CHAINS:
        neighbors.setdefault(parent, set()).add(child)
        neighbors.setdefault(child, set()).add(parent)
    return neighbors


def _is_wrist_near_head(cand, wrist_j, threshold=None):
    """True if wrist keypoint is within threshold of head (nose or neck); treat as occluded."""
    if threshold is None:
        threshold = WRIST_NEAR_HEAD_THRESHOLD
    if wrist_j not in WRIST_JOINTS or not is_valid_kp(cand[wrist_j]):
        return False
    w = cand[wrist_j]
    for head_j in (0, 1):  # nose, neck
        if not is_valid_kp(cand[head_j]):
            continue
        d = np.linalg.norm(w - cand[head_j])
        if d <= threshold:
            return True
    return False


# Hand keypoint index that corresponds to wrist (palm base); used to infer wrist from hand
HAND_WRIST_KP_IDX = 0


def _hand_extent_too_large(pose, hand_idx, wrist_idx, cand, max_extent=None):
    """True if the hand keypoints extend unrealistically far from the wrist (e.g. stretched fingers)."""
    if max_extent is None:
        max_extent = HAND_MAX_EXTENT
    if not is_valid_kp(cand[wrist_idx]):
        return True
    wrist = cand[wrist_idx]
    hands = pose['hands']
    if hands is None or hand_idx >= hands.shape[0]:
        return False
    max_d = 0.0
    for k in range(hands.shape[1]):
        kp = hands[hand_idx, k]
        if is_valid_kp(kp):
            d = np.linalg.norm(kp - wrist)
            if d > max_d:
                max_d = d
    return max_d > max_extent


def _infer_wrist_from_hand(pose, hand_idx):
    """Infer wrist position from hand keypoints. Returns (x, y) or None if not enough valid keypoints.

    Prefer hand keypoint at HAND_WRIST_KP_IDX (wrist in hand model); else use centroid of valid keypoints.
    """
    hands = pose['hands']
    if hands is None or hand_idx >= hands.shape[0]:
        return None
    arr = hands[hand_idx]
    if HAND_WRIST_KP_IDX < arr.shape[0] and is_valid_kp(arr[HAND_WRIST_KP_IDX]):
        return arr[HAND_WRIST_KP_IDX].copy()
    xs, ys = [], []
    for k in range(arr.shape[0]):
        if is_valid_kp(arr[k]):
            xs.append(arr[k][0])
            ys.append(arr[k][1])
    if len(xs) < 2:
        return None
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=arr.dtype)


def _hand_anchor(pose, hand_idx):
    """Hand base position for connection to arm: kp0 if valid, else centroid. Returns None if no valid keypoints."""
    wrist_pt = _infer_wrist_from_hand(pose, hand_idx)
    return wrist_pt


def _infer_wrist_from_arm(pose, hand_idx, wrist_idx, cand, forearm_len=None):
    """Infer wrist from arm: elbow + direction toward hand (or default) * length.

    Uses hand keypoints to get direction when available; otherwise extends forearm from shoulder->elbow
    or a default direction. Returns (x, y) or None if elbow invalid.
    """
    if forearm_len is None:
        forearm_len = DEFAULT_FOREARM_LEN
    elbow_idx = WRIST_ELBOW_MAP[wrist_idx]
    if not is_valid_kp(cand[elbow_idx]):
        return None
    elbow = cand[elbow_idx]
    hand_anchor = _hand_anchor(pose, hand_idx)
    if hand_anchor is not None:
        diff = hand_anchor - elbow
        dist = np.linalg.norm(diff)
        if dist > 1e-8:
            direction = diff / dist
            length = min(forearm_len, dist)
            return elbow + direction * length
    # No hand: extend forearm. Use shoulder->elbow direction if available.
    shoulder_idx = 2 if wrist_idx == 4 else 5
    if is_valid_kp(cand[shoulder_idx]):
        diff = elbow - cand[shoulder_idx]
        dist = np.linalg.norm(diff)
        if dist > 1e-8:
            direction = diff / dist
            return elbow + direction * forearm_len
    # Default: right wrist (4) extend right-down, left wrist (7) extend left-down (normalized x right = +)
    sign = 1.0 if wrist_idx == 4 else -1.0
    direction = np.array([sign * 0.5, 0.866], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    return elbow + direction * forearm_len


def _infer_wrist_from_arm_given_cand(cand, wrist_idx, forearm_len=None):
    """Infer wrist from arm using only body candidate (no hand). For use in retarget_hands."""
    if forearm_len is None:
        forearm_len = DEFAULT_FOREARM_LEN
    elbow_idx = WRIST_ELBOW_MAP[wrist_idx]
    if not is_valid_kp(cand[elbow_idx]):
        return None
    elbow = cand[elbow_idx]
    shoulder_idx = 2 if wrist_idx == 4 else 5
    if is_valid_kp(cand[shoulder_idx]):
        diff = elbow - cand[shoulder_idx]
        dist = np.linalg.norm(diff)
        if dist > 1e-8:
            direction = diff / dist
            return elbow + direction * forearm_len
    sign = 1.0 if wrist_idx == 4 else -1.0
    direction = np.array([sign * 0.5, 0.866], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    return elbow + direction * forearm_len


def _infer_wrist_from_hand_given_hands(hands, hand_idx):
    """Infer wrist from hand array only (no full pose). For use in retarget_hands."""
    if hands is None or hand_idx >= hands.shape[0]:
        return None
    arr = hands[hand_idx]
    if HAND_WRIST_KP_IDX < arr.shape[0] and is_valid_kp(arr[HAND_WRIST_KP_IDX]):
        return arr[HAND_WRIST_KP_IDX].copy()
    xs, ys = [], []
    for k in range(arr.shape[0]):
        if is_valid_kp(arr[k]):
            xs.append(arr[k][0])
            ys.append(arr[k][1])
    if len(xs) < 2:
        return None
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=arr.dtype)


def resolve_wrist_and_hand(pose):
    """Resolve wrist position from hand or arm so arm and hand connect plausibly; never exclude valid hand keypoints.

    - If body wrist is valid and hand extent is OK: keep.
    - If extent too large or wrist missing: infer wrist from hand (body wrist = hand base) or from arm
      (wrist = elbow + forearm); when using arm-inferred wrist, shift hand so hand base = wrist so both
      arm-wrist and wrist-hand connections are drawn correctly.
    Updates pose['bodies']['candidate'] and optionally pose['hands'] in place. Ensures subset marks wrist visible when set.
    """
    cand = pose['bodies']['candidate']
    subset = pose['bodies'].get('subset')
    hands = pose.get('hands')

    def mark_wrist_visible(wrist_idx):
        if subset is not None and subset.size > 0:
            if subset.ndim == 2 and wrist_idx < subset.shape[1]:
                subset[0, wrist_idx] = wrist_idx
            elif subset.ndim == 1 and wrist_idx < len(subset):
                subset[wrist_idx] = wrist_idx

    for hand_idx, wrist_idx in HAND_WRIST_MAP.items():
        wrist_from_hand = _infer_wrist_from_hand(pose, hand_idx) if hands is not None else None
        wrist_from_arm = _infer_wrist_from_arm(pose, hand_idx, wrist_idx, cand)
        body_wrist_valid = is_valid_kp(cand[wrist_idx])
        extent_too_large = _hand_extent_too_large(pose, hand_idx, wrist_idx, cand) if body_wrist_valid else True

        if body_wrist_valid and not extent_too_large:
            continue

        # Prefer arm-based wrist when it lands close to hand (then shift hand to attach)
        use_arm = False
        if wrist_from_arm is not None and wrist_from_hand is not None:
            d = np.linalg.norm(wrist_from_arm - wrist_from_hand)
            if d <= HAND_MAX_EXTENT * 1.5:
                use_arm = True
        if use_arm and hands is not None and hand_idx < hands.shape[0]:
            cand[wrist_idx] = wrist_from_arm.copy()
            mark_wrist_visible(wrist_idx)
            # Shift hand so hand base = body wrist (connect wrist to hand)
            base = _hand_anchor(pose, hand_idx)
            if base is not None:
                delta = cand[wrist_idx] - base
                for k in range(hands.shape[1]):
                    if is_valid_kp(hands[hand_idx, k]):
                        hands[hand_idx, k] = hands[hand_idx, k] + delta
            continue

        # Use hand-based wrist: body wrist = hand base; arm connects to hand
        if wrist_from_hand is not None:
            cand[wrist_idx] = wrist_from_hand.copy()
            mark_wrist_visible(wrist_idx)
        elif wrist_from_arm is not None:
            cand[wrist_idx] = wrist_from_arm.copy()
            mark_wrist_visible(wrist_idx)


def apply_fullbody_hand_validity_mask(retargeted_seq, fullbody_hand_validity):
    """Remove hand keypoints that do not exist in the corresponding full-body pose frame.

    After kinematic/visibility inference, some hand keypoints may be present in the
    partial-body frame but were not valid in the full-body retargeted pose for that
    frame. This sets those to invalid so only keypoints that existed in the full-body
    pose are kept. Modifies retargeted_seq in place.
    """
    if not fullbody_hand_validity or len(fullbody_hand_validity) == 0:
        return
    for f, pose in enumerate(retargeted_seq):
        if f >= len(fullbody_hand_validity) or fullbody_hand_validity[f] is None:
            continue
        hands = pose.get('hands')
        if hands is None:
            continue
        valid_mask = fullbody_hand_validity[f]
        for h in range(min(hands.shape[0], valid_mask.shape[0])):
            for k in range(min(hands.shape[1], valid_mask.shape[1])):
                if not valid_mask[h, k]:
                    hands[h, k] = np.array([-1.0, -1.0], dtype=hands.dtype)


def apply_fullbody_body_validity_mask(retargeted_seq, fullbody_body_validity):
    """Set body keypoints to invalid if they were not valid in the corresponding full-body pose frame.

    Ensures we do not show keypoints or connections that were not present in the full-body pose.
    Modifies retargeted_seq in place.
    """
    if not fullbody_body_validity or len(fullbody_body_validity) == 0:
        return
    for f, pose in enumerate(retargeted_seq):
        if f >= len(fullbody_body_validity) or fullbody_body_validity[f] is None:
            continue
        cand = pose['bodies']['candidate']
        subset = pose['bodies'].get('subset')
        valid_mask = fullbody_body_validity[f]
        n_joints = min(cand.shape[0], len(valid_mask))
        for j in range(n_joints):
            if not valid_mask[j]:
                cand[j] = np.array([-1.0, -1.0], dtype=cand.dtype)
                if subset is not None and subset.size > 0:
                    if subset.ndim == 2 and j < subset.shape[1]:
                        subset[0, j] = -1
                    elif subset.ndim == 1 and j < len(subset):
                        subset[j] = -1


def _hand_centroid(pose, hand_idx):
    """Centroid of valid keypoints for hand hand_idx. Returns None only if zero valid keypoints."""
    hands = pose.get('hands')
    if hands is None or hand_idx >= hands.shape[0]:
        return None
    arr = hands[hand_idx]
    xs, ys = [], []
    for k in range(arr.shape[0]):
        if is_valid_kp(arr[k]):
            xs.append(arr[k][0])
            ys.append(arr[k][1])
    if len(xs) == 0:
        return None
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=arr.dtype)


def _hand_wrist_arm_connections_valid(pose, hand_idx):
    """True if hand connects to wrist and wrist connects to arm (valid for drawing).

    Requires: wrist keypoint valid, elbow valid (so arm segment exists), and hand base
    within HAND_MAX_EXTENT of wrist so the hand is not detached.
    """
    cand = pose['bodies']['candidate']
    wrist_idx = HAND_WRIST_MAP.get(hand_idx)
    if wrist_idx is None or wrist_idx >= cand.shape[0] or not is_valid_kp(cand[wrist_idx]):
        return False
    elbow_idx = WRIST_ELBOW_MAP.get(wrist_idx)
    if elbow_idx is None or not is_valid_kp(cand[elbow_idx]):
        return False
    hand_base = _hand_centroid(pose, hand_idx)
    if hand_base is None:
        return False
    if np.linalg.norm(hand_base - cand[wrist_idx]) > HAND_MAX_EXTENT:
        return False
    return True


def _arm_chain_valid(cand, wrist_idx):
    """True if shoulder, elbow, and wrist are all valid for this arm (no inference)."""
    if wrist_idx not in WRIST_ELBOW_MAP or wrist_idx >= cand.shape[0]:
        return False
    elbow_idx = WRIST_ELBOW_MAP[wrist_idx]
    shoulder_idx = 2 if wrist_idx == 4 else 5  # right arm 2,3,4; left arm 5,6,7
    return (is_valid_kp(cand[shoulder_idx])
            and is_valid_kp(cand[elbow_idx])
            and is_valid_kp(cand[wrist_idx]))


def apply_relative_hand_positions(
        retargeted_seq, fullbody_in_refspace, skip_frame_0=True,
        orig_ref_cand=None, edit_ref_cand=None, coord_tf=None):
    """Refine hand only when hand-wrist-arm connections are invalid; use full-body relative position.

    (1) If connections are valid (hand–wrist–arm connected), do not overwrite kinematic result.
    (2) If connections are missing, recompute hand position from full-body relative (hand–neck offset)
        in ref space: target = neck_curr + (hand_ref - neck_ref). No size rescale.
    Modifies retargeted_seq in place.
    """
    if not fullbody_in_refspace or len(fullbody_in_refspace) == 0:
        return
    neck_idx = NECK_JOINT
    start = 1 if skip_frame_0 else 0
    for f in range(start, len(retargeted_seq)):
        if f >= len(fullbody_in_refspace):
            break
        pose = retargeted_seq[f]
        ref_pose = fullbody_in_refspace[f]
        cand = pose['bodies']['candidate']
        ref_cand = ref_pose['bodies']['candidate']
        subset = pose['bodies'].get('subset')
        if cand.shape[0] <= neck_idx or ref_cand.shape[0] <= neck_idx:
            continue
        neck = cand[neck_idx]
        neck_ref = ref_cand[neck_idx]
        if not is_valid_kp(neck) or not is_valid_kp(neck_ref):
            continue
        hands = pose.get('hands')
        ref_hands = ref_pose.get('hands')
        if hands is None or ref_hands is None:
            continue
        for hand_idx in range(hands.shape[0]):
            if hand_idx >= ref_hands.shape[0]:
                continue
            # Only redraw when connections are missing (invalid)
            if _hand_wrist_arm_connections_valid(pose, hand_idx):
                continue
            hand_centroid_ref = _hand_centroid(ref_pose, hand_idx)
            if hand_centroid_ref is None:
                continue
            hand_centroid_curr = _hand_centroid(pose, hand_idx)
            if hand_centroid_curr is None:
                continue
            # Position: full-body relative in ref space (fullbody_in_refspace already in ref space); no size rescale
            target_centroid = neck + (hand_centroid_ref - neck_ref)
            delta = target_centroid - hand_centroid_curr
            for k in range(hands.shape[1]):
                if is_valid_kp(hands[hand_idx, k]):
                    hands[hand_idx, k] = hands[hand_idx, k] + delta
            # Set body wrist to hand base so arm connects
            wrist_idx = HAND_WRIST_MAP.get(hand_idx)
            if wrist_idx is not None and wrist_idx < cand.shape[0]:
                cand[wrist_idx] = target_centroid.copy()
                if subset is not None and subset.size > 0:
                    if subset.ndim == 2 and wrist_idx < subset.shape[1]:
                        subset[0, wrist_idx] = wrist_idx
                    elif subset.ndim == 1 and wrist_idx < len(subset):
                        subset[wrist_idx] = wrist_idx


def _infer_view(pose):
    """Infer view type from relative positions of left vs right keypoints.

    Returns one of: 'front', 'back', 'side_left', 'side_right'.
    - Back: face/nose occluded (invalid or collapsed), so do not infer face keypoints.
    - Side: shoulders (or hips) have similar x (person turned); one side occluded.
    """
    cand = pose['bodies']['candidate']
    # Back view: nose/face not clearly in front (invalid or nose at/below neck level)
    nose_ok = is_valid_kp(cand[0])
    neck_ok = is_valid_kp(cand[1])
    if neck_ok and (not nose_ok or (cand[0][1] >= cand[1][1] - 0.02)):
        return 'back'
    # Side view: shoulder line short in x (person turned ~90°)
    r_shoulder_ok = is_valid_kp(cand[2])
    l_shoulder_ok = is_valid_kp(cand[5])
    shoulder_width_x = 0.0
    if r_shoulder_ok and l_shoulder_ok:
        shoulder_width_x = abs(cand[2][0] - cand[5][0])
    # Normalized coords: typical shoulder width ~0.15–0.25; side view << 0.08
    if shoulder_width_x < 0.08:
        # Which side is toward camera? Use hip x if available, else shoulder x vs neck
        r_hip_ok = is_valid_kp(cand[8])
        l_hip_ok = is_valid_kp(cand[11])
        if neck_ok and (r_shoulder_ok or l_shoulder_ok):
            # Side with shoulder closer to neck in x is often the visible (forward) side
            if r_shoulder_ok and l_shoulder_ok:
                dx_r = abs(cand[2][0] - cand[1][0])
                dx_l = abs(cand[5][0] - cand[1][0])
                return 'side_right' if dx_r < dx_l else 'side_left'
            return 'side_left' if l_shoulder_ok else 'side_right'
        if r_hip_ok and l_hip_ok:
            return 'side_right' if cand[8][0] > cand[11][0] else 'side_left'
        return 'side_left'
    return 'front'


def _visible_joints_with_kinematic_propagation(
        pose, visible_region, ref_visible_joints, kinematic_neighbors):
    """Compute per-frame visible joints: ref + region + kinematic propagation with view awareness.

    Core: ref_visible_joints that are inside visible_region.
    Propagate along kinematic chains, but do NOT infer keypoints/connections that are
    blocked by view (back: face occluded; side: occluded side not propagated).
    """
    cand = pose['bodies']['candidate']
    core = ref_visible_joints & _joints_inside_region(pose, visible_region)
    visible = set(core)
    view = _infer_view(pose)

    def allow_propagate_to(from_j, to_j):
        if to_j not in ref_visible_joints or not is_valid_kp(cand[to_j]):
            return False
        # Do not show wrist (and thus arm–hand connection) when hand is near head (occluded)
        if to_j in WRIST_JOINTS and _is_wrist_near_head(cand, to_j):
            return False
        if view == 'back':
            if to_j in FRONT_ONLY_JOINTS:
                return False
        if view == 'side_left':
            if to_j in RIGHT_BODY_JOINTS:
                return False
        if view == 'side_right':
            if to_j in LEFT_BODY_JOINTS:
                return False
        return True

    changed = True
    while changed:
        changed = False
        for j in list(visible):
            for nb in kinematic_neighbors.get(j, set()):
                if nb in visible:
                    continue
                if not allow_propagate_to(j, nb):
                    continue
                visible.add(nb)
                changed = True

    # Remove wrists that are near head (occluded); avoids drawing hand–head connections
    for j in WRIST_JOINTS:
        if j in visible and _is_wrist_near_head(cand, j):
            visible.discard(j)
    return visible


def apply_visibility_mask_per_frame(retargeted_seq, visible_region,
                                    ref_visible_joints):
    """Apply visibility mask per frame: ref as guidance + region + kinematic propagation.

    For each frame t:
      1. Core: keypoints in ref_visible_joints that lie inside visible_region.
      2. Kinematic propagation: add any joint in ref_visible_joints that is
         adjacent (in the kinematic chain) to an already-visible joint and has
         a valid keypoint. This keeps limbs connected (e.g. full arm when
         shoulder is visible) and avoids mistakenly removing valid keypoints
         or connections.
    Modifies retargeted_seq in place.
    """
    if visible_region is None or ref_visible_joints is None:
        return
    ref_visible_joints = set(ref_visible_joints)
    kinematic_neighbors = _kinematic_neighbors()
    for pose in retargeted_seq:
        visible_this_frame = _visible_joints_with_kinematic_propagation(
            pose, visible_region, ref_visible_joints, kinematic_neighbors)
        apply_visibility_mask(pose, visible_region,
                              visible_joints=visible_this_frame)


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
# Fit pose sequence to [0,1] canvas (remove gap at bottom for non-SAM path)
# =============================================================================
def fit_pose_sequence_to_canvas(poses_seq):
    """Scale and translate all poses so the global keypoint bbox fills [0,1]x[0,1].

    Removes the gap between the bottom keypoints and the bottom edge of the
    rendered video.  Modifies poses_seq in place.
    """
    xs, ys = [], []
    for pose in poses_seq:
        cand = pose['bodies']['candidate']
        for j in range(cand.shape[0]):
            if is_valid_kp(cand[j]):
                xs.append(cand[j][0])
                ys.append(cand[j][1])
        for hi in range(pose['hands'].shape[0]):
            for k in range(pose['hands'].shape[1]):
                if is_valid_kp(pose['hands'][hi, k]):
                    xs.append(pose['hands'][hi, k][0])
                    ys.append(pose['hands'][hi, k][1])
        for fi in range(pose['faces'].shape[0]):
            for k in range(pose['faces'].shape[1]):
                if is_valid_kp(pose['faces'][fi, k]):
                    xs.append(pose['faces'][fi, k][0])
                    ys.append(pose['faces'][fi, k][1])
    if len(xs) < 2:
        return poses_seq
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min:
        x_min, x_max = 0.0, 1.0
    if y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    rx = (x_max - x_min)
    ry = (y_max - y_min)

    def map_kp(kp):
        if not is_valid_kp(kp):
            return
        kp[0] = (kp[0] - x_min) / rx
        kp[1] = (kp[1] - y_min) / ry

    for pose in poses_seq:
        cand = pose['bodies']['candidate']
        for j in range(cand.shape[0]):
            map_kp(cand[j])
        for hi in range(pose['hands'].shape[0]):
            for k in range(pose['hands'].shape[1]):
                map_kp(pose['hands'][hi, k])
        for fi in range(pose['faces'].shape[0]):
            for k in range(pose['faces'].shape[1]):
                map_kp(pose['faces'][fi, k])
    return poses_seq


# =============================================================================
# Drawing
# =============================================================================
def _hand_drawing_plausible(pose, hand_idx, wrist_idx, cand, max_wrist_dist=None, max_internal_extent=None):
    """True if hand keypoints and their distances to the body wrist are plausible for drawing connections.

    (a) Hand keypoints: at least 2 valid; max internal extent (anchor to any keypoint) not too large.
    (b) Wrist distance: body wrist valid and distance from wrist to hand anchor <= max_wrist_dist.
    """
    if max_wrist_dist is None:
        max_wrist_dist = HAND_MAX_EXTENT
    if max_internal_extent is None:
        max_internal_extent = HAND_MAX_EXTENT * 1.5
    if not is_valid_kp(cand[wrist_idx]):
        return False
    wrist = cand[wrist_idx]
    hands = pose.get('hands')
    if hands is None or hand_idx >= hands.shape[0]:
        return False
    arr = hands[hand_idx]
    valid_pts = [arr[k] for k in range(arr.shape[0]) if is_valid_kp(arr[k])]
    if len(valid_pts) < 2:
        return False
    anchor = _hand_anchor(pose, hand_idx)
    if anchor is None:
        return False
    if np.linalg.norm(wrist - anchor) > max_wrist_dist:
        return False
    internal_extent = max(np.linalg.norm(np.array(p) - anchor) for p in valid_pts)
    if internal_extent > max_internal_extent:
        return False
    return True


def draw_pose_partial_body_hands(pose, H, W):
    """Draw pose for partial-body (second phase: edited_ref -> ref_name): body without hands first, then hands conditionally.

    1. Draw body + foot skeleton only (no hand keypoints).
    2. For each hand: if keypoints and wrist distances are plausible, draw connections + keypoints; else draw keypoints only.
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_body_and_foot(canvas, candidate, subset)
    # Draw hands: connections only when plausible; otherwise keypoints only
    if hands is not None:
        for hand_idx, wrist_idx in HAND_WRIST_MAP.items():
            if hand_idx >= hands.shape[0]:
                continue
            hand_row = hands[hand_idx]
            if not is_valid_kp(hand_row[0]) and not any(is_valid_kp(hand_row[k]) for k in range(1, hand_row.shape[0])):
                continue
            if _hand_drawing_plausible(pose, hand_idx, wrist_idx, candidate):
                canvas = util.draw_handpose(canvas, [hand_row])
            else:
                canvas = util.draw_hand_keypoints_only(canvas, [hand_row])
    canvas_without_face = copy.deepcopy(canvas)
    canvas = util.draw_facepose(canvas, faces)
    return canvas_without_face, canvas


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
def auto_max_bone_ratio(ref_cand, drv_first_cand, floor=1.3, ceiling=2.0):
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

        # If the RETARGETED parent is invalid, the chain is broken —
        # propagate invalidity to all downstream children.
        if not is_valid_kp(retargeted[parent]):
            retargeted[child] = np.array([-1.0, -1.0])
            continue

        if not is_valid_kp(drv_c):
            retargeted[child] = np.array([-1.0, -1.0])
            continue

        if not is_valid_kp(drv_p):
            # Do not infer: only retarget valid keypoints from source; leave child invalid.
            retargeted[child] = np.array([-1.0, -1.0])
            continue

        # Normal case: both driving parent and child are valid
        direction = normalize_vec(drv_c - drv_p)

        # Degenerate direction: do not infer from reference; leave child invalid
        if np.linalg.norm(direction) < 1e-8:
            retargeted[child] = np.array([-1.0, -1.0])
            continue

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


def fill_missing_body_connections(poses_seq, ref_cand, ref_bone_lengths,
                                  default_bone_len=0.05):
    """Compensate for DWPose missing detections: fill body keypoints from
    symmetric completion and kinematic propagation.

    When driven video / ref are partial-body, the pose estimator may leave
    some keypoints invalid. This fills them using:
    1. Symmetric completion: mirror valid left/right counterpart (e.g. L shoulder
       from R shoulder) around the midline.
    2. Kinematic propagation: if parent is valid and child invalid, set child =
       parent + direction * length; direction/length from ref or symmetric ref
       or default.

    Modifies poses_seq in place. Use only for full-body pipeline.
    """
    n_joints = ref_cand.shape[0]
    for pose in poses_seq:
        cand = pose['bodies']['candidate']
        subset = pose['bodies']['subset']
        if cand.shape[0] < n_joints:
            continue
        center_x = float(cand[1][0]) if is_valid_kp(cand[1]) else 0.5

        def mark_visible(j):
            if subset.ndim == 2 and subset.shape[0] > 0 and j < subset.shape[1]:
                subset[0, j] = j
            elif subset.ndim == 1 and j < len(subset):
                subset[j] = j

        # 1. Symmetric completion
        for (a, b) in SYMMETRIC_BODY_PAIRS:
            if a >= n_joints or b >= n_joints:
                continue
            if is_valid_kp(cand[a]) and not is_valid_kp(cand[b]):
                cand[b] = np.array([2.0 * center_x - cand[a][0], cand[a][1]])
                mark_visible(b)
            elif not is_valid_kp(cand[a]) and is_valid_kp(cand[b]):
                cand[a] = np.array([2.0 * center_x - cand[b][0], cand[b][1]])
                mark_visible(a)

        # 2. Kinematic propagation (root-outward)
        for parent, child in KINEMATIC_CHAINS:
            if child >= n_joints:
                continue
            if not is_valid_kp(cand[parent]) or is_valid_kp(cand[child]):
                continue

            direction = np.array([0.0, 0.0])
            length = default_bone_len

            if is_valid_kp(ref_cand[parent]) and is_valid_kp(ref_cand[child]):
                diff = ref_cand[child] - ref_cand[parent]
                ln = bone_length(ref_cand[parent], ref_cand[child])
                if ln > 1e-8:
                    direction = normalize_vec(diff)
                    length = ln
            if np.linalg.norm(direction) < 1e-8 and (parent, child) in SYMMETRIC_CHAIN:
                p2, c2 = SYMMETRIC_CHAIN[(parent, child)]
                if (p2 < n_joints and c2 < n_joints
                        and is_valid_kp(ref_cand[p2]) and is_valid_kp(ref_cand[c2])):
                    diff = ref_cand[c2] - ref_cand[p2]
                    diff[0] = -diff[0]
                    ln = bone_length(ref_cand[p2], ref_cand[c2])
                    if ln > 1e-8:
                        direction = normalize_vec(diff)
                        length = ln
            if np.linalg.norm(direction) < 1e-8 and (parent, child) in _DEFAULT_NECK_DIRECTIONS:
                direction = normalize_vec(np.array(_DEFAULT_NECK_DIRECTIONS[(parent, child)]))
                length = ref_bone_lengths.get((parent, child)) or default_bone_len
            if np.linalg.norm(direction) < 1e-8:
                length = ref_bone_lengths.get((parent, child)) or default_bone_len
                if length > 1e-8:
                    direction = normalize_vec(np.array([0.0, 0.1]))

            length = float(length) if (length is not None and length > 1e-8) else default_bone_len
            if np.linalg.norm(direction) >= 1e-8:
                cand[child] = np.array([
                    cand[parent][0] + direction[0] * length,
                    cand[parent][1] + direction[1] * length
                ], dtype=np.float64)
                mark_visible(child)

    return poses_seq


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
# Improvement 4: retarget hands / face (full-body)
# =============================================================================
def retarget_hands(drv_hands, drv_cand, ret_cand, ref_cand,
                  max_bone_ratio=1.5):
    """Re-anchor hand keypoints: wrist-based when arm chain valid and hand attached to wrist, else neck-based.

    Per hand, per-frame (no inference of missing keypoints):
    - If shoulder–elbow–wrist are all valid and the hand is attached to the wrist (hand base within
      HAND_MAX_EXTENT of wrist): use wrist-based re-anchoring when ret wrist is valid.
    - If arm chain is broken or hand is detached from wrist: use neck-based re-anchoring.
    """
    ret_hands = drv_hands.copy()
    drv_pose = {'bodies': {'candidate': drv_cand}, 'hands': drv_hands}

    for hand_idx, wrist_idx in HAND_WRIST_MAP.items():
        if (_hand_wrist_arm_connections_valid(drv_pose, hand_idx)
                and is_valid_kp(ret_cand[wrist_idx])):
            # Wrist-based re-anchoring (no inference)
            drv_wrist = drv_cand[wrist_idx]
            ret_wrist = ret_cand[wrist_idx]
            delta = ret_wrist - drv_wrist
            for kp in range(drv_hands.shape[1]):
                if is_valid_kp(drv_hands[hand_idx, kp]):
                    ret_hands[hand_idx, kp] = drv_hands[hand_idx, kp] + delta
        else:
            # Neck-based re-anchoring when arm chain or wrist missing
            drv_neck = drv_cand[NECK_JOINT]
            ret_neck = ret_cand[NECK_JOINT]
            if not is_valid_kp(drv_neck) or not is_valid_kp(ret_neck):
                ret_hands[hand_idx] = -1.0
                continue
            for kp in range(drv_hands.shape[1]):
                if is_valid_kp(drv_hands[hand_idx, kp]):
                    offset = drv_hands[hand_idx, kp] - drv_neck
                    ret_hands[hand_idx, kp] = ret_neck + offset

    return ret_hands


def _face_extent(faces, nose):
    """Max distance from nose to any valid face keypoint; None if no valid kp."""
    out = 0.0
    for fi in range(faces.shape[0]):
        for k in range(faces.shape[1]):
            if is_valid_kp(faces[fi, k]):
                d = np.linalg.norm(np.asarray(faces[fi, k]) - np.asarray(nose))
                if d > out:
                    out = d
    return out if out > 1e-6 else None


def retarget_face(drv_faces, drv_cand, ret_cand, ref_cand, ref_faces=None,
                  max_bone_ratio=1.5):
    """Nose delta so face follows body; then apply reference face size.

    (1) Apply nose delta: translated_face = drv_face + (ret_nose - drv_nose)
    so face keypoints follow the movement of the body skeleton.
    (2) Use reference image face keypoints for face size: scale the translated
    face around ret_nose so its extent matches the reference face size.
    """
    nose_idx = 0
    drv_nose = drv_cand[nose_idx]
    ret_nose = ret_cand[nose_idx]
    if not is_valid_kp(drv_nose) or not is_valid_kp(ret_nose):
        ret_faces = drv_faces.copy()
        ret_faces[:] = -1.0
        return ret_faces
    delta = ret_nose - drv_nose
    # (1) Nose delta: face follows body
    ret_faces = drv_faces.copy()
    for fi in range(ret_faces.shape[0]):
        for k in range(ret_faces.shape[1]):
            if is_valid_kp(ret_faces[fi, k]):
                ret_faces[fi, k] = ret_faces[fi, k] + delta
    # (2) Reference face size: scale around ret_nose to match ref extent, but never shrink
    # (shrinking causes compacted/shrunken face when ref extent is smaller than current)
    if ref_faces is not None and ref_faces.shape == ret_faces.shape and is_valid_kp(ref_cand[nose_idx]):
        ref_nose = ref_cand[nose_idx]
        ref_extent = _face_extent(ref_faces, ref_nose)
        curr_extent = _face_extent(ret_faces, ret_nose)
        if ref_extent is not None and curr_extent is not None and curr_extent > 1e-6:
            scale = ref_extent / curr_extent
            scale = np.clip(scale, 1.0, 2.0)  # only scale up to match ref, never shrink
            for fi in range(ret_faces.shape[0]):
                for k in range(ret_faces.shape[1]):
                    if is_valid_kp(ret_faces[fi, k]):
                        offset = ret_faces[fi, k] - ret_nose
                        ret_faces[fi, k] = ret_nose + scale * offset
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
    """Enforce joint-angle limits on the retargeted skeleton.

    NOTE: we intentionally do NOT clamp to [0, 1] here.  During dynamic
    motion (e.g., hip-hop dance), limbs can legitimately extend beyond
    the canvas.  Hard-clamping distorts bone lengths and propagates
    errors through the kinematic chain.  The drawing code handles
    out-of-bounds coordinates naturally (OpenCV clips lines at the
    canvas edge).
    """
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
    if args.source_video_paths.endswith('mp4'):
        video_paths = [args.source_video_paths]
    else:
        video_paths = [
            os.path.join(args.source_video_paths, f)
            for f in sorted(os.listdir(args.source_video_paths))]

    logger.info("Videos to process: {}".format(len(video_paths)))
    logger.info('Loading DWpose model ...')
    dwpose_model = DWposeDetector()

    # Step 1: extract poses from driving video; capture FPS from first video for output
    results_vis = []
    source_fps_for_output = None
    for i, fpath in enumerate(video_paths):
        logger.info("  [{}/{}] {}".format(i + 1, len(video_paths), fpath))
        cap = cv2.VideoCapture(fpath)
        if source_fps_for_output is None and getattr(args, 'fps', 30.0) <= 0:
            source_fps_for_output = cap.get(cv2.CAP_PROP_FPS)
            if not (source_fps_for_output and source_fps_for_output > 0):
                source_fps_for_output = 30.0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results_vis.append(dwpose_model(frame))
        cap.release()

    if getattr(args, 'fps', 30.0) <= 0:
        args.fps = float(source_fps_for_output) if source_fps_for_output else 30.0
        logger.info("Output FPS: {:.2f} (auto from source video)".format(args.fps))
    else:
        logger.info("Output FPS: {:.2f} (from --fps)".format(args.fps))

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
    edit_ref_frame = None

    if partial_body_mode:
        # Partial-body algorithm: (1) Full-body retargeting with edited_ref + video_char.
        # (2) Ratio and position (ref_name vs edited_ref_name) → mapping; map all frames to ref.
        # (3) Per-frame visibility: ref_name pose as guidance, infer for sequential frames.
        logger.info("Partial-body mode: using edited ref for retargeting.")
        edit_ref_frame = cv2.imread(args.edited_ref_name, cv2.IMREAD_COLOR)
        assert edit_ref_frame is not None, \
            "Cannot read edited reference image: {}".format(
                args.edited_ref_name)
        pose_edit_ref = dwpose_model(edit_ref_frame)
        edit_ref_cand = pose_edit_ref['bodies']['candidate']

        # Stage 1: full-body retargeting uses edited_ref as reference (video_char → edited_ref_name)
        ref_cand = edit_ref_cand

        # Visibility region (needed before coord transform to decide joint set)
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
        # visible_region filters "detected": only joints inside it count,
        # so SAM upper-body mask excludes extrapolated hips and hides legs.
        visible_joints = compute_visible_joint_set(
            orig_ref_cand, propagate=using_sam, visible_region=visible_region)
        logger.info("Visible joints ({}): {}".format(
            "propagated" if using_sam else "detected-only",
            sorted(visible_joints)))

        # Coordinate transform: edited ref space -> original ref space.
        # When the ref shows upper legs (hips/knees visible), include them in
        # the fit; otherwise upper-body only to avoid biased-high ref hips.
        coord_tf_joints = _coord_transform_joint_indices(
            orig_ref_cand, edit_ref_cand, visible_region)
        coord_tf = compute_coord_transform(
            orig_ref_cand, edit_ref_cand, joint_indices=coord_tf_joints)
        if set(coord_tf_joints) & {8, 11, 9, 12}:
            logger.info("Coord transform (edited->orig, upper-body + hips/knees): "
                         "sx={:.3f} sy={:.3f} tx={:.3f} ty={:.3f}".format(
                             *coord_tf))
        else:
            logger.info("Coord transform (edited->orig, upper-body only): "
                         "sx={:.3f} sy={:.3f} tx={:.3f} ty={:.3f}".format(
                             *coord_tf))
    else:
        ref_cand = orig_ref_cand
        coord_tf = None
        visible_region = None
        visible_joints = None
        using_sam = False
        # Single-ref mode: if ref has no visible lower body, DWPose hips are
        # often too high; re-estimate hip positions for better alignment.
        if correct_ref_hips_for_partial_ref(ref_cand):
            logger.info("Ref has no visible lower body; corrected ref hip "
                        "positions for alignment.")

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

    # Step 3.5 & 4: optional inference on source (default: none for faithful retargeting).
    # If we fill or interpolate the source, missing elbow/wrist/etc. get inferred and
    # then retargeted, so the output shows limbs that were not in the original source.
    # Hand re-anchoring would also use wrist-based when the inferred wrist is valid.
    # By default we do not infer: only keypoints valid in the source are retargeted.
    if getattr(args, 'infer_source', False):
        if not partial_body_mode:
            logger.info("Filling missing body connections (symmetric + kinematic).")
            results_vis = fill_missing_body_connections(
                results_vis, ref_cand, ref_bone_lengths)
        logger.info("Interpolating missing keypoints on source ...")
        results_vis = interpolate_missing_keypoints(results_vis)
    else:
        logger.info("Faithful retargeting: no inference on source (only valid keypoints).")

    # Step 5: per-frame full-body retargeting (driving video → ref_cand).
    # In partial-body mode this is stage 1: video_char (driving video poses) → edited_ref_name.
    # Face: use reference face keypoints (ref_name or edited_ref_name) and re-anchor at retargeted nose (no scaling).
    if partial_body_mode:
        logger.info("Stage 1 (full-body): retargeting video_char → edited_ref ...")
        ref_pose_faces = pose_edit_ref['faces'].copy()
    else:
        ref_pose_faces = pose_orig_ref['faces'].copy()
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
        scaled_hands = drv_hands.copy()
        scaled_faces = drv_faces.copy()
        if abs(scale - 1.0) > 0.01:
            hip_ctr = 0.5 * (drv_cand[8] + drv_cand[11])
            for j_idx in range(scaled_cand.shape[0]):
                if is_valid_kp(scaled_cand[j_idx]):
                    scaled_cand[j_idx] = (
                        hip_ctr + (scaled_cand[j_idx] - hip_ctr) / scale)
            # Keep hands and face in same coordinate system as scaled body
            for h in range(scaled_hands.shape[0]):
                for k in range(scaled_hands.shape[1]):
                    if is_valid_kp(scaled_hands[h, k]):
                        scaled_hands[h, k] = (
                            hip_ctr + (scaled_hands[h, k] - hip_ctr) / scale)
            for fi in range(scaled_faces.shape[0]):
                for k in range(scaled_faces.shape[1]):
                    if is_valid_kp(scaled_faces[fi, k]):
                        scaled_faces[fi, k] = (
                            hip_ctr + (scaled_faces[fi, k] - hip_ctr) / scale)

        # Improvement 7: two-anchor root position
        root = compute_root_position(scaled_cand, ref_cand, base_drv_cand)
        ret_body = retarget_body_angle_based(
            scaled_cand, ref_cand, ref_bone_lengths, root,
            max_bone_ratio=mbr)

        # Improvement 8: physical plausibility
        ret_body = validate_pose(ret_body)

        # Sanity check: if any retargeted bone is way off compared to the
        # scaled driving skeleton, fall back to the scaled driving position
        # for that child joint.  This catches corner-case retargeting errors
        # without breaking the overall skeleton.
        SANITY_RATIO = 3.0
        for parent, child in KINEMATIC_CHAINS:
            if (is_valid_kp(ret_body[parent])
                    and is_valid_kp(ret_body[child])
                    and is_valid_kp(scaled_cand[parent])
                    and is_valid_kp(scaled_cand[child])):
                ret_len = bone_length(ret_body[parent], ret_body[child])
                drv_len = bone_length(scaled_cand[parent], scaled_cand[child])
                if drv_len > 1e-6 and ret_len / drv_len > SANITY_RATIO:
                    # Retargeted bone unreasonably long: use driving direction
                    # with reference length
                    ref_len = ref_bone_lengths.get((parent, child))
                    if ref_len is not None:
                        direction = normalize_vec(
                            scaled_cand[child] - scaled_cand[parent])
                        ret_body[child] = (
                            ret_body[parent] + direction * ref_len)

        # Improvement 4: hands and face relative to parent joints
        # Stage 1 (partial-body): face uses video_char (driving) vs edited_ref (ref_cand) for anchor/scale.
        # Use scaled driving body/hands/face so wrist (and nose) deltas are in one coordinate system
        ret_h = retarget_hands(scaled_hands, scaled_cand, ret_body, ref_cand)
        ret_f = retarget_face(scaled_faces, scaled_cand, ret_body, ref_cand, ref_faces=ref_pose_faces)

        retargeted.append({
            'bodies': {
                'candidate': ret_body,
                'subset': results_vis[f]['bodies']['subset'].copy(),
            },
            'hands': ret_h,
            'faces': ret_f,
        })

    # Step 5.5: partial-body — stage 2: map full-body result (edited_ref space) to ref_name space
    # Stage 1 (Step 5) already produced video_char → edited_ref full-body retargeting.
    # (2) Ratio and position from ref_name vs edited_ref_name → mapping (sx,sy,tx,ty).
    # (3) Map all frames from edit_ref space to ref space. (4) Position correction.
    # (5) Per-frame visibility: ref pose as guidance, infer for sequential frames.
    fullbody_hand_validity = None
    fullbody_body_validity = None
    fullbody_in_refspace = None
    partial_body_edit_ref_cand = None  # edit_ref pose candidate for ref/edit ratio in hand refinement
    retargeted_edited_ref = None  # full-body retargeted sequence (video_char -> edited_ref) for saving edited_ref pose video
    if partial_body_mode and coord_tf is not None:
        retargeted_edited_ref = [copy.deepcopy(pose) for pose in retargeted]
        partial_body_edit_ref_cand = edit_ref_cand.copy()
        # Capture hand and body validity from full-body poses (before coord transform) for later masking
        fullbody_hand_validity = []
        fullbody_body_validity = []
        for pose in retargeted:
            hands = pose.get('hands')
            if hands is None:
                fullbody_hand_validity.append(None)
            else:
                valid = np.array(
                    [[is_valid_kp(hands[h, k]) for k in range(hands.shape[1])]
                     for h in range(hands.shape[0])],
                    dtype=bool)
                fullbody_hand_validity.append(valid)
            cand = pose['bodies']['candidate']
            fullbody_body_validity.append(
                np.array([is_valid_kp(cand[j]) for j in range(cand.shape[0])], dtype=bool))

        sx, sy, tx, ty = coord_tf
        logger.info("Partial-body mapping (edit_ref -> ref_name): "
                     "sx={:.3f}, sy={:.3f}, tx={:.3f}, ty={:.3f}".format(
                         sx, sy, tx, ty))

        # Map all full-body retargeted frames from edited_ref space to ref_name space
        for pose in retargeted:
            apply_coord_transform_pose(pose, sx, sy, tx, ty)

        # Align entire sequence to reference: same offset so all frames match ref
        logger.info("Aligning entire pose sequence to reference ...")
        retargeted = apply_position_correction(retargeted, orig_ref_cand)

        # Stage 2 face: use ref_name face keypoints, re-anchor at each frame's nose (no scaling)
        apply_face_ref_anchor_partial_body(
            retargeted, orig_ref_cand, pose_orig_ref, skip_frame_0=True)
        logger.info("Stage 2 face alignment (ref_name face re-anchored) applied.")

        # Optionally attenuate global motion when transform scale is large
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

        # Save full-body layout in ref space (after attenuation) for relative hand-position correction
        fullbody_in_refspace = [copy.deepcopy(pose) for pose in retargeted]

        # Per-frame visibility: ref_name pose as guidance, infer for sequential frames.
        # Keep keypoint j in frame t only if j in ref_visible_joints and inside visible_region.
        if visible_region is not None and visible_joints is not None:
            logger.info("Applying per-frame visibility (ref guidance + region inference).")
            apply_visibility_mask_per_frame(retargeted, visible_region,
                                            visible_joints)
    else:
        # Standard mode: simple position correction
        logger.info("Applying position correction ...")
        retargeted = apply_position_correction(retargeted, orig_ref_cand)

    # Resolve wrist from hand or arm (kinematic inference) for all modes
    for pose in retargeted:
        resolve_wrist_and_hand(pose)

    # Partial-body: (1) Identify keypoints to draw — mask to full-body validity only
    if fullbody_hand_validity is not None:
        logger.info("Masking hand keypoints to full-body pose validity.")
        apply_fullbody_hand_validity_mask(retargeted, fullbody_hand_validity)
    if fullbody_body_validity is not None:
        logger.info("Masking body keypoints to full-body pose validity.")
        apply_fullbody_body_validity_mask(retargeted, fullbody_body_validity)

    # Partial-body: (2)(3)(4) Refine hand only when connections invalid; use full-body relative + ref/edit ratio
    if fullbody_in_refspace is not None:
        logger.info("Refining hand (only when hand-wrist-arm connections invalid) from full-body relative + ref/edit ratio.")
        apply_relative_hand_positions(
            retargeted, fullbody_in_refspace, skip_frame_0=True,
            orig_ref_cand=orig_ref_cand, edit_ref_cand=partial_body_edit_ref_cand, coord_tf=coord_tf)

    # Skip fit_pose_sequence_to_canvas in partial-body mode: that function
    # rescales all keypoints to fill [0,1]x[0,1], which would destroy
    # alignment with the reference image. Keeping poses in orig-ref space
    # preserves skeleton size and keypoint positions to match ref_pose.jpg.

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

    # Step 7: temporal smoothing (optional)
    if getattr(args, 'temporal_smoothing', False):
        logger.info("Applying temporal smoothing (One Euro Filter) ...")
        retargeted = apply_temporal_smoothing(
            retargeted,
            fps=args.fps,
            min_cutoff=args.smooth_min_cutoff,
            beta=args.smooth_beta,
        )
    else:
        logger.info("Temporal smoothing disabled (default).")

    # Step 8: render and save (images + video)
    save_dir = args.saved_pose_dir
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # In partial-body mode, render at ref image resolution so skeleton size
    # and aspect ratio match ref_pose.jpg; otherwise use fixed size.
    if partial_body_mode and ref_frame is not None:
        render_h, render_w = ref_frame.shape[0], ref_frame.shape[1]
        logger.info("Partial-body: rendering at ref resolution {}x{}".format(
            render_w, render_h))
    else:
        render_h, render_w = 768, 512
    draw_face = getattr(args, 'draw_face', False)

    # Save pose skeletons for ref_name, video_char_image, and edited_ref_name
    # (default: always saved alongside the retargeted pose video).
    def _save_pose_skeleton(pose, image, save_path, draw_face_kp=False):
        H, W = image.shape[0], image.shape[1]
        wo_face, with_face = draw_pose(pose, H=H, W=W)
        out = with_face if draw_face_kp else wo_face
        cv2.imwrite(save_path, out)

    logger.info("Saving pose skeletons for ref, video_char, and edited_ref ...")
    _save_pose_skeleton(
        pose_orig_ref, ref_frame,
        os.path.join(save_dir, "ref_pose.jpg"), draw_face)
    _save_pose_skeleton(
        base_char_pose, base_char_image,
        os.path.join(save_dir, "video_char_pose.jpg"), draw_face)
    if partial_body_mode and edit_ref_frame is not None and pose_edit_ref is not None:
        _save_pose_skeleton(
            pose_edit_ref, edit_ref_frame,
            os.path.join(save_dir, "edited_ref_pose.jpg"), draw_face)

    # Draw pose (body + hand/wrist connections as usual; hand keypoints already masked to full-body validity in partial-body)
    logger.info("Rendering {} frames to {} (draw_face={}) ...".format(
        len(retargeted), save_dir, draw_face))
    for i, pose in enumerate(retargeted):
        wo_face, with_face = draw_pose(pose, H=render_h, W=render_w)
        img_path = os.path.join(save_dir, "{:04d}.jpg".format(i))
        cv2.imwrite(img_path, with_face if draw_face else wo_face)

    video_path = os.path.join(save_dir, "pose_sequence.mp4")
    ffmpeg_cmd = (
        'ffmpeg -y -framerate {} -i {}/%04d.jpg '
        '-c:v libx264 -pix_fmt yuv420p -crf 18 {}'
    ).format(args.fps, save_dir, video_path)
    logger.info("Encoding video with ffmpeg ...")
    os.system(ffmpeg_cmd)
    logger.info("Saved video: {}".format(video_path))

    # In partial-body mode, also save the full-body retargeted pose video (video_char -> edited_ref)
    if retargeted_edited_ref is not None and edit_ref_frame is not None:
        edit_ref_h, edit_ref_w = edit_ref_frame.shape[0], edit_ref_frame.shape[1]
        edited_ref_frames_dir = os.path.join(save_dir, "edited_ref_frames")
        os.makedirs(edited_ref_frames_dir, exist_ok=True)
        logger.info("Rendering {} full-body (edited_ref) pose frames at {}x{} ...".format(
            len(retargeted_edited_ref), edit_ref_w, edit_ref_h))
        for i, pose in enumerate(retargeted_edited_ref):
            wo_face, with_face = draw_pose(pose, H=edit_ref_h, W=edit_ref_w)
            img_path = os.path.join(edited_ref_frames_dir, "{:04d}.jpg".format(i))
            cv2.imwrite(img_path, with_face if draw_face else wo_face)
        video_path_edited_ref = os.path.join(save_dir, "pose_sequence_edited_ref.mp4")
        ffmpeg_cmd_ed = (
            'ffmpeg -y -framerate {} -i {}/%04d.jpg '
            '-c:v libx264 -pix_fmt yuv420p -crf 18 {}'
        ).format(args.fps, edited_ref_frames_dir, video_path_edited_ref)
        os.system(ffmpeg_cmd_ed)
        logger.info("Saved full-body (edited_ref) pose video: {}".format(video_path_edited_ref))
        if getattr(args, 'video_only', False):
            shutil.rmtree(edited_ref_frames_dir, ignore_errors=True)

    if getattr(args, 'video_only', False):
        _keep = {'ref_pose.jpg', 'video_char_pose.jpg', 'edited_ref_pose.jpg'}
        for fn in os.listdir(save_dir):
            if fn.endswith('.jpg') and fn not in _keep:
                os.remove(os.path.join(save_dir, fn))
        logger.info("--video_only: removed individual frame images (kept ref/video_char/edited_ref pose skeletons).")

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
        "--fps", type=float, default=0,
        help="Output video FPS. 0 = auto-detect from first source video (default).")
    parser.add_argument(
        "--temporal_smoothing", action="store_true",
        help="Enable One Euro Filter temporal smoothing on retargeted poses. "
             "Default: disabled.")
    parser.add_argument(
        "--smooth_min_cutoff", type=float, default=1.7,
        help="One-Euro min cutoff when --temporal_smoothing is set; "
             "higher = less smoothing (default: 1.7).")
    parser.add_argument(
        "--smooth_beta", type=float, default=0.3,
        help="One-Euro beta when --temporal_smoothing is set; "
             "higher = less lag on fast motion (default: 0.3).")
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
        "--draw_face", action="store_true",
        help="Draw face keypoints on the output pose video. Face keypoints "
             "are only present when using a wholebody pose model (134 "
             "keypoints); the default dw-ll_ucoco_384.onnx is body-only.")
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
    parser.add_argument(
        "--infer_source", action="store_true",
        help="If set, fill missing body connections (full-body only) and "
             "interpolate missing keypoints on the source pose. Default: off "
             "for faithful retargeting (only keypoints valid in the source "
             "video are retargeted; no inferred elbow/wrist/hand).")
    args = parser.parse_args()
    mp_main(args)
