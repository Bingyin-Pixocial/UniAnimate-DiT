# UniAnimate-DiT 

An expanded version of [UniAnimate](https://arxiv.org/abs/2406.01188) based on [Wan2.1](https://github.com/Wan-Video/Wan2.1)

UniAnimate-DiT is based on a state-of-the-art DiT-based Wan2.1-14B-I2V model for consistent human image animation. This codebase is built upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), thanks for the nice open-sourced project.

<div align="center">

<p align="center">
  <img src='https://github.com/user-attachments/assets/b7290f98-8b33-4626-945c-cf287ba84192' width='784'>

  Overview of the proposed UniAnimate-DiT
</p>

</div>


## 🔥 News 
- **[2025/04/21]** 🔥 We support Unified Sequence Parallel (USP) for multi-GPUs inference.
- **[2025/04/18]** 🔥🔥🔥 **We support teacache for both short video generation and long video generation, which can achieve about 4 times inference acceleration.** Now, it costs ~3 minutes to generate 5s 480p videos and ~13 minutes to generate 5s 720p videos on one A800 GPU. You can use teacache to select seed and disenable teacache for ideal results.
- **[2025/04/18]** 🔥 We support teacache, which can achieve about 4 times inference acceleration. It may have a slight impact on performance, and you can use teacache to select the seed. Long video generation does not currently support teacache acceleration, but we are working hard to overcome this.
- **[2025/04/16]** 🔥 The technical report is avaliable on [ArXiv](https://arxiv.org/pdf/2504.11289).
- **[2025/04/15]** 🔥🔥🔥 We released the training and inference code of UniAnimate-DiT based on [UniAnimate](https://github.com/ali-vilab/UniAnimate) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). The technical report will be avaliable soon.


##  Demo cases
<table>
<center>
<tr>
    <!-- <td width=25% style="border: none"> -->
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/9671e4e1-edf4-4352-af1e-6743aff4e9f0" muted="false"></video>
    </td>
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/c3cf5dc6-19d2-4865-92b8-b687b4e7a901" muted="false"></video>
    </td>
</tr>
</table>



<table>
<center>
<tr>
    <!-- <td width=25% style="border: none"> -->
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/bd8a9dba-33b0-432f-8ae4-911d7044eb28" muted="false"></video>
    </td>
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/79601ec8-ed35-4542-9bb3-777085c6a4a0" muted="false"></video>
    </td>
</tr>
</table>


<table>
<center>
<tr>
    <!-- <td width=25% style="border: none"> -->
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/83ae10c3-9828-4eed-95db-f4e3265924b9" muted="false"></video>
    </td>
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/a6838591-4ed1-436e-b016-0c4d3864d92e" muted="false"></video>
    </td>
</tr>
</table>



<table>
<center>
<tr>
    <!-- <td width=25% style="border: none"> -->
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/9e2d75d3-8b1e-4cbb-91a5-dacf99c18261" muted="false"></video>
    </td>
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/32104e1a-4f20-4070-a458-73d9e9401013" muted="false"></video>
    </td>
</tr>
</table>



<table>
<center>
<tr>
    <!-- <td width=25% style="border: none"> -->
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/e7ae8deb-26e2-4452-844c-a8a043dd9846" muted="false"></video>
    </td>
    <td ><center>
        <video height="260" controls autoplay loop src="https://github.com/user-attachments/assets/7f96e347-617f-4c78-bc59-a2bcef9f8080" muted="false"></video>
    </td>
</tr>
</table>

## Getting Started with UniAnimate-DiT


### (1) Installation

Before using this model, please create the conda environment and install DiffSynth-Studio from **source code**.

```shell
conda create -n UniAnimate-DiT python=3.9.21
# or conda create -n UniAnimate-DiT python=3.10.16 # Python>=3.10 is required for Unified Sequence Parallel (USP)
conda activate UniAnimate-DiT

# CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/ali-vilab/UniAnimate-DiT.git
cd UniAnimate-DiT
pip install -e .
```

UniAnimate-DiT supports multiple Attention implementations. If you have installed any of the following Attention implementations, they will be enabled based on priority.

* [Flash Attention 3](https://github.com/Dao-AILab/flash-attention)
* [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
* [Sage Attention](https://github.com/thu-ml/SageAttention)
* [torch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (default. `torch>=2.5.0` is recommended.)

## Inference


### (2) Download the pretrained checkpoints

(i) Download Wan2.1-14B-I2V-720P models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
```

Or download Wan2.1-14B-I2V-720P models using modelscope-cli:
```
pip install modelscope
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir ./Wan2.1-I2V-14B-720P
```


(ii) Download pretrained UniAnimate-DiT models (only include the weights of lora and additional learnable modules):
```
pip install modelscope
modelscope download xiaolaowx/UniAnimate-DiT --local_dir ./checkpoints
```

Or download UniAnimate-DiT models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download ZheWang123/UniAnimate-DiT --local-dir ./checkpoints
```

(iii) Finally, the model weights will be organized in `./checkpoints/` as follows:
```
./checkpoints/
|---- dw-ll_ucoco_384.onnx
|---- UniAnimate-Wan2.1-14B-Lora-12000.ckpt
└---- yolox_l.onnx
```

### (3) Improved pose alignment developed by Bingyin:
**A. For images that DWPose can extract the skeleton, run:**
```
bash run_dwpose_alignment.sh
```

**B. For images that DWPose cannot extract the skeleton, run:**
```
bash run_xpose_alignment.sh
```

**C. Enhanced pose alignment with angle-based retargeting (recommended):**

`dwpose_alignment_improved.py` is an improved version of `dwpose_alignment.py` with the following enhancements:

| # | Improvement | Description |
|---|------------|-------------|
| 1 | Angle-based retargeting | Preserves driving motion's joint angles while applying reference bone lengths (replaces translation-based approach) |
| 2 | Temporal smoothing | One Euro Filter on all keypoints to reduce jitter while preserving fast motions |
| 3 | Occlusion handling | Missing keypoints are filled via linear temporal interpolation |
| 3.5 | No inference on source (default) | By default the **source (driving) pose is not filled or interpolated**: only keypoints valid in the source video are retargeted. Missing elbow/wrist/arm in the source stay missing in the output; hands use neck-based re-anchoring when arm chain is invalid. Use **`--infer_source`** to enable the previous behavior (fill missing body connections in full-body mode + temporal interpolation on source). |
| 4 | Relative hand/face alignment | **Full-body hands:** Wrist-based only when shoulder–elbow–wrist valid and hand attached to wrist; else neck-based. **Full-body face:** Nose-delta. **Partial-body:** When connections invalid, apply full-body relative hand position. |
| 5 | Ground-plane constraints | Feet pinned to a median ground plane during detected contact frames |
| 6 | Depth-adaptive scaling | Per-frame scale normalisation based on shoulder width to handle camera depth changes; hands and face are scaled with the body so full-body retargeting uses a single coordinate system and hand/wrist positions stay correct |
| 7 | Two-anchor alignment | Hip-center-driven root position for natural lateral sway and vertical bounce |
| 8 | Physical plausibility | Joint angle limits (elbows/knees) and canvas boundary clamping |
| 9 | Partial-body support | When the reference image shows only part of the body (e.g., face/upper body), uses an edited full-body reference for retargeting, then maps poses back to the visible region via coordinate transform + visibility masking |
| 10 | Position correction | Automatic anchor-based global offset so the retargeted skeleton matches the reference character's position regardless of where the driving video character is |
| 11 | Motion attenuation | In partial-body mode, global sway/drift is scaled down inversely to the coordinate-transform zoom so close-up skeletons stay on canvas |
| 12 | Auto max_bone_ratio | Automatic bone-length ratio limit computed from skeleton scale difference; set `--max_bone_ratio 0` (default) |
| 13 | Partial-body coord transform | Edit→orig transform is fitted using **upper-body joints** by default; when the ref shows **upper legs** (hips or knees visible in the visible region), hip and knee keypoints are **included** so they are not excluded or misaligned. Otherwise upper-body only avoids biased-high ref hips |
| 14 | Ref-hip correction (single ref) | When the reference has no visible lower body (no ankles/feet), ref hip positions are re-estimated below the neck using a torso-to-shoulder ratio so the first frame aligns better with the reference image |
| 15 | Full-sequence alignment (partial-body) | After the coord transform, **position correction** is applied so the **entire** retargeted pose sequence (all frames) is aligned to the reference image with the same offset, not only the first frame |
| 16 | First frame = ref pose (partial-body) | Frame 0 is overwritten with the **reference pose** so the first frame exactly matches `ref_pose.jpg` in skeleton size and keypoint positions |
| 17 | No canvas re-fit in partial-body | `fit_pose_sequence_to_canvas` is **not** applied in partial-body mode so that poses stay in reference space and alignment is preserved |
| 18 | Render at ref resolution (partial-body) | The retargeted pose video is rendered at the **reference image resolution** so skeleton size and aspect ratio match `ref_pose.jpg` when compared side by side |

```bash
bash run_dwpose_alignment_improved.sh
```

**Output (default):** In `--saved_pose_dir`, the script saves the retargeted pose video (`pose_sequence.mp4`) and, by default, pose skeleton images for the inputs: `ref_pose.jpg` (skeleton on ref image), `video_char_pose.jpg` (skeleton on video character image), and `edited_ref_pose.jpg` (skeleton on edited ref image, only in partial-body mode). Frame images `0000.jpg`, `0001.jpg`, … are also written unless `--video_only` is set. In **partial-body mode**, the full-body retargeted pose sequence (video_char → edited_ref_name) is also saved as `pose_sequence_edited_ref.mp4` at edited_ref image resolution.

New optional arguments (vs original):
- `--source_video_paths` &mdash; (required) path to source driving video (.mp4) or directory of videos (DWPose runs on each frame).
- `--fps` &mdash; output video FPS; **0** = auto-detect from first source video (default); positive value overrides
- `--temporal_smoothing` &mdash; enable One Euro Filter on retargeted poses (default: off)
- `--smooth_min_cutoff` &mdash; One-Euro min cutoff when `--temporal_smoothing` is set (default: 1.7)
- `--smooth_beta` &mdash; One-Euro beta when `--temporal_smoothing` is set (default: 0.3)
- `--max_bone_ratio` &mdash; maximum allowed bone-length ratio between reference and driving characters; set to **0** for automatic detection based on skeleton scale difference (recommended); a positive value is used as-is (default: 0 = auto)
- `--video_only` &mdash; only keep the output video and the ref/video_char/edited_ref pose skeleton images; individual frame images (0000.jpg, …) are deleted after encoding (available in all three pose alignment scripts)
- `--edited_ref_name` &mdash; path to a full-body edited version of the reference image; enables **partial-body mode** where only the visible region of the original reference is rendered
- `--sam_checkpoint` &mdash; path to a SAM checkpoint (e.g., `sam_vit_b_01ec64.pth`) for precise person-mask visibility detection; requires `pip install segment-anything`; falls back to keypoint-based visibility if not provided. When using SAM, a refinement step aligns each frame’s visible keypoints to the reference image For the SAM path, keypoints visible in ref_name are used as guidance: keypoints not in that set are eliminated from the retargeted poses (before coord transform and again after motion attenuation), so the output only shows what the reference image shows.
- `--visibility_margin` &mdash; margin (normalised) added around the detected visible region in partial-body mode (default: 0.05)
- `--infer_source` &mdash; if set, fill missing body connections (full-body only) and interpolate missing keypoints on the source pose. **Default: off** for faithful retargeting (only keypoints valid in the source video are retargeted; no inferred elbow/wrist/hand)

**Partial-body algorithm** (when `--edited_ref_name` is set):

1. **Full-body retargeting** — Run the full-body retargeting using `edited_ref_name` and `video_char_image`. The result is a pose sequence in edited-ref (full-body) space.
2. **Mapping to ref_name** — Use SAM or keypoint-based visibility to get the visible region. Compute the **ratio and position difference** between `ref_name` and `edited_ref_name` (linear fit on common keypoints → `sx`, `sy`, `tx`, `ty`). Map **all** full-body retargeted frames from edited-ref space to ref_name space; then apply position correction so the whole sequence aligns with the reference image.
3. **Per-frame visibility (ref + kinematic + view-aware)** — Use the ref_name pose as **guidance**. For each frame: **(a)** *Core:* keypoints in the ref’s visible set that lie inside the visible region. **(b)** *Kinematic propagation:* add adjacent joints in the ref’s set to keep limbs connected. **(c)** *View-aware:* infer front/back/side from relative positions of left vs right keypoints; do **not** infer keypoints or connections that are blocked by the view. **(d)** *Wrist near head:* if a wrist is very close to the head (nose/neck), treat it as occluded and do not show it (or the hand), so no hand–head connection is drawn. **(e)** *Hand–wrist connectivity:* wrist is resolved from hand or arm so connections are plausible. If the body wrist is missing or the wrist-to-hand extent is too long, the wrist is inferred either from hand keypoints (hand base or centroid) or from the arm (elbow + forearm direction); when arm-inferred wrist is used and is close to the hand, the hand is shifted so it attaches to the wrist. The body wrist is always updated so both elbow–wrist and wrist–hand are drawn; valid hand keypoints are never excluded.
4. **Keypoints to draw** — Body and hand keypoints are masked to the corresponding full-body pose frame so only keypoints that existed in the full-body pose are kept (no inferred keypoints/connections that are not in the full-body).
5. **Kinematic inference** — Wrist/hand connectivity is inferred (resolve_wrist_and_hand) so arms and hands connect plausibly.
6. **Refine hand only when connections invalid** — For each hand we evaluate whether hand–wrist–arm connections are valid (wrist valid, elbow valid, hand base within range of wrist). **If valid**: do not redraw; keep kinematic result. **If invalid** (hand detached): recompute hand position from full-body relative (hand–neck offset in ref space), set wrist to hand base and mark visible so the arm connects (no size rescale).
7. **Rendering** — All keypoints are drawn together (body, hand–wrist, wrist–arm, and hand internal connections as usual).

**Partial-body mode** is useful when the reference image shows only part of the character (e.g., a close-up face shot) while the driving video shows the full body. Provide an edited full-body version of the reference image via `--edited_ref_name`:

```bash
python dwpose_alignment_improved.py \
  --ref_name face_ref.jpg \
  --edited_ref_name face_ref_fullbody.jpg \
  --video_char_image video_char.png \
  --source_video_paths dance.mp4 \
  --saved_pose_dir output/ \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --visibility_margin 0.05
```

#### Video crop (reference-aligned)

`video_crop_align.py` crops a full-body video to match the partial-body framing of a reference image: given a reference image (e.g. face/upper body) and a full-body video of the same person, it computes the crop area once from the first frame and the reference, then applies that same crop to all frames for temporal consistency and to avoid flickering. It reuses DWPose and optional SAM from the improved alignment pipeline.

```bash
bash run_video_crop_align.sh
```

Arguments:
- `--ref_name` &mdash; reference image path (partial body)
- `--source_video_paths` &mdash; input video file (`.mp4`/`.avi`) or directory of videos
- `--output_video` &mdash; output cropped video path
- `--source_pose_video_paths` &mdash; optional pose video (same length as driven video); the same crop is applied to produce a second output
- `--output_pose_video` &mdash; output path for cropped pose video (required when `--source_pose_video_paths` is set)
- `--output_size` &mdash; optional output resolution as `WxH` (default: same as driven video)
- `--fps` &mdash; output FPS (default: auto-detect from driven video; fallback 30)
- `--sam_checkpoint` &mdash; path to SAM checkpoint for visibility region (optional)
- `--visibility_margin` &mdash; margin around visible region (default: 0.05)
- `--save_crops` &mdash; save the crop box to a JSON file next to the output video

### (4) Pose alignment 

Rescale the target pose sequence to match the pose of the reference image (you can also install `pip install onnxruntime-gpu==1.18.1` for faster extraction on GPU.):
```
# reference image 1
python run_align_pose.py  --ref_name data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg --source_video_paths data/videos/source_video.mp4 --saved_pose_dir data/saved_pose/WOMEN-Blouses_Shirts-id_00004955-01_4_full 

# reference image 2
python run_align_pose.py  --ref_name data/images/musk.jpg --source_video_paths data/videos/source_video.mp4 --saved_pose_dir data/saved_pose/musk 

# reference image 3
python run_align_pose.py  --ref_name data/images/WOMEN-Blouses_Shirts-id_00005125-03_4_full.jpg --source_video_paths data/videos/source_video.mp4 --saved_pose_dir data/saved_pose/WOMEN-Blouses_Shirts-id_00005125-03_4_full

# reference image 4
python run_align_pose.py  --ref_name data/images/IMG_20240514_104337.jpg --source_video_paths data/videos/source_video.mp4 --saved_pose_dir data/saved_pose/IMG_20240514_104337

# reference image 5
python run_align_pose.py  --ref_name data/images/10.jpg --source_video_paths data/videos/source_video.mp4 --saved_pose_dir data/saved_pose/10

# reference image 6
python run_align_pose.py  --ref_name data/images/taiyi2.jpg --source_video_paths data/videos/source_video.mp4 --saved_pose_dir data/saved_pose/taiyi2
```
The processed target pose for demo videos will be in ```data/saved_pose```. `--ref_name` denotes the path of reference image, `--source_video_paths` provides the source poses, `--saved_pose_dir` means the path of processed target poses.


### (5) Run UniAnimate-DiT-14B to generate 480P videos

```
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_480p.py
```
About 23G GPU memory is needed. After this, 81-frame video clips with 832x480 (hight x width) resolution will be generated under the `./outputs` folder.

- **Tips**: you can also set `cfg_scale=1.0` to save inference time, which disables classifier-free guidance and can double the speed with minimal performance impact. https://github.com/ali-vilab/UniAnimate-DiT/blob/c2c7019dbb081464271d470d750b7693ade10dd8/examples/unianimate_wan/inference_unianimate_wan_480p.py#L223-L224

- **Tips**: you can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

|`torch_dtype`|`num_persistent_param_in_dit`|Speed|Required VRAM|Default Setting|
|-|-|-|-|-|
|torch.bfloat16|7*10**9 (7B)|20.5s/it|23G|yes|
|torch.bfloat16|0|23.0s/it|14G||

- **Tips**: you can set `use_teacache=True` to enable teacache, which can achieve about 4 times inference acceleration. It may have a slight impact on performance, and you can also use teacache to select the seed. 

If you have many GPUs for inference, we also support Unified Sequence Parallel (USP), note that python>=3.10 is required for Unified Sequence Parallel (USP):

```
pip install xfuser
torchrun --standalone --nproc_per_node=4 examples/unianimate_wan/inference_unianimate_wan_480p_usp.py
```

For long video generation, run the following comment, the tips above can also be used by yourself:

```
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_long_video_480p.py
```

### (6) Run UniAnimate-DiT-14B to generate 720P videos

```
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_720p.py
```
About 36G GPU memory is needed. After this, 81-frame video clips with 1280x720 resolution will be generated.

- **Tips**: you can also set `cfg_scale=1.0` to save inference time, which disables classifier-free guidance and can double the speed with minimal performance impact. https://github.com/ali-vilab/UniAnimate-DiT/blob/c37c996740cb9584edbdf3b4db2fa9eb47526e30/examples/unianimate_wan/inference_unianimate_wan_720p.py#L224-L225

- **Tips**: you can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

|`torch_dtype`|`num_persistent_param_in_dit`|Speed|Required VRAM|Default Setting|
|-|-|-|-|-|
|torch.bfloat16|7*10**9 (7B)|20.5s/it|36G|yes|
|torch.bfloat16|0|23.0s/it|26G||

- **Tips**: you can set `use_teacache=True` to enable teacache, which can achieve about 4 times inference acceleration. It may have a slight impact on performance, and you can also use teacache to select the seed. 


**Note**: Even though our model was trained on 832x480 resolution, we observed that direct inference on 1280x720 is usually allowed and produces satisfactory results. 


For long video generation, run the following comment, the tips above can also be used by yourself:

```
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_long_video_720p.py
```

**Note**: We find use teacache for 720P long video generation may lead to inconsistent background. We still work on it. You can use teacache to select random seed and disenable teacache for ideal results.

## Train

We support UniAnimate-DiT training on our own dataset. 

### Step 1: Install additional packages

```
pip install peft lightning pandas
# deepspeed for multiple GPUs
pip install -U deepspeed
```

### Step 2: Prepare your dataset

In order to speed up the training, we preprocessed the videos, extracted video frames and corresponding Dwpose in advance, and packaged them with pickle package. You need to manage the training data as follows:

```
data/example_dataset/
└── TikTok
    └── 00001_mp4
      ├── dw_pose_with_foot_wo_face.pkl # packaged Dwpose
      └── frame_data.pkl # packaged frames
```

We encourage adding large amounts of data to finetune models to get better results. The experimental results show that about 1000 training videos can finetune a good human image animation model. Please refer to `prepare_training_data.py` file for more details about packaged Dwpose/frames.

### Step 3: Train

For convenience, we do not pre-process VAE features, but put VAE pre-processing and DiT model training in a training script, and also facilitate data augmentation to improve performance. You can also choose to extract VAE features first and then conduct subsequent DiT model training. 


LoRA training (One A100 GPU):

```shell
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/train_unianimate_wan.py \
   --task train  \
   --train_architecture lora \
   --lora_rank 64 --lora_alpha 64  \
   --dataset_path data/example_dataset   \
   --output_path ./models_out_one_GPU   \
   --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"    \
   --max_epochs 10   --learning_rate 1e-4   \
   --accumulate_grad_batches 1   \
   --use_gradient_checkpointing --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload 
```


LoRA training (Multi-GPUs, based on `Deepseed`):

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" python examples/unianimate_wan/train_unianimate_wan.py  \
   --task train   --train_architecture lora \
   --lora_rank 128 --lora_alpha 128  \
   --dataset_path data/example_dataset   \
   --output_path ./models_out   --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     \
   --max_epochs 10   --learning_rate 1e-4   \
   --accumulate_grad_batches 1   \
   --use_gradient_checkpointing \
   --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
   --use_gradient_checkpointing_offload \
   --training_strategy "deepspeed_stage_2" 
```


You can also finetune our trained model by set `--pretrained_lora_path="./checkpoints/UniAnimate-Wan2.1-14B-Lora-12000.ckpt"`.

### Step 4: Test

Test the LoRA finetuned model trained on one GPU:

```python
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData, WanUniAnimateVideoPipeline


# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",

        ],
        "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, 
)

model_manager.load_lora_v2("models/lightning_logs/version_1/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)

...
...
```

Test the LoRA finetuned model trained on multi-GPUs based on Deepspeed, first you need `python zero_to_fp32.py . output_dir/ --safe_serialization` to change the .pt files to .safetensors files. Note that `zero_to_fp32.py` is an automatically generated file that can be found in the checkpoint folder after training with DeepSpeed on ​​Multi-GPUs. And then run:

```python
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData, WanUniAnimateVideoPipeline


# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",

        ],
        "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, 
)

model_manager.load_lora_v2([
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00001-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00002-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00003-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00004-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00005-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00006-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00007-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00008-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00009-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00010-of-00011.safetensors",
            "./models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt/output_dir/model-00011-of-00011.safetensors",
            ], lora_alpha=1.0)

...
...
```


## Citation

If you find this codebase useful for your research, please cite the following paper:

```
@article{wang2025unianimate,
      title={UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation},
      author={Wang, Xiang and Zhang, Shiwei and Gao, Changxin and Wang, Jiayu and Zhou, Xiaoqiang and Zhang, Yingya and Yan, Luxin and Sang, Nong},
      journal={Science China Information Sciences},
      year={2025}
}
```


## Disclaimer

This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.
