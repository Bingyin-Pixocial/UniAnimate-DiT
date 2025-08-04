import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import cv2
import numpy as np

# Option 1: Use Wan2.2-I2V-A14B (does NOT support control_video)
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)

# Option 2: Use Wan2.1-Fun-Control model (supports control_video)
# Uncomment the following lines to use a control-compatible model:
# pipe = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
#         ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
#         ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
#         ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
#     ],
# )

pipe.enable_vram_management()


# input_image = Image.open("data/pix/start_img.png").resize((832, 480))
input_image = Image.open("data/pix/4.jpg").resize((832, 480))

# Configuration for video length
num_frames = 81  # Change this to generate longer videos (e.g., 120, 150, 200, etc.)

# Load control video frames (only needed if using control-compatible model)
# Uncomment the following code if using Option 2 (control-compatible model):
# control_video_path = "data/pix/pix_dance_7.mp4"
# cap = cv2.VideoCapture(control_video_path)
# control_video = []
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Convert BGR to RGB and resize to match input dimensions
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_resized = cv2.resize(frame_rgb, (832, 480))
#     # Convert numpy array to PIL Image
#     frame_pil = Image.fromarray(frame_resized)
#     control_video.append(frame_pil)
# cap.release()

# # Ensure control video has the correct number of frames
# if len(control_video) != num_frames:
#     if len(control_video) > num_frames:
#         # Sample frames evenly if we have too many
#         indices = np.linspace(0, len(control_video) - 1, num_frames, dtype=int)
#         control_video = [control_video[i] for i in indices]
#     else:
#         # Repeat the last frame if we have too few
#         last_frame = control_video[-1] if control_video else control_video[0]
#         while len(control_video) < num_frames:
#             control_video.append(last_frame)

# print(f"Control video: {type(control_video)}, length: {len(control_video)}, frame size: {control_video[0].size}")

video = pipe(
    prompt="A person dancing energetically in a modern dance style with fluid movements and dynamic poses.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    input_image=input_image,
    # control_video=control_video,  # Uncomment this line if using control-compatible model
    num_frames=num_frames,
)
save_video(video, "./outputs/4_pix_dance_7_wan2.2.mp4", fps=15, quality=5)