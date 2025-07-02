# UniAnimate-DiT Docker Setup

This document provides instructions for running UniAnimate-DiT using Docker, which ensures consistent environment setup across different systems.

## Prerequisites

1. **Docker**: Install Docker Engine (version 20.10 or later)
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   ```

2. **NVIDIA Docker**: For GPU support, install NVIDIA Docker
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **NVIDIA Drivers**: Ensure you have NVIDIA drivers installed on your host system

## Quick Start

### Option 1: Using the provided script (Recommended)

1. **Build the Docker image**:
   ```bash
   ./docker-run.sh build
   ```

2. **Start the container**:
   ```bash
   ./docker-run.sh run
   ```

3. **Access the container shell**:
   ```bash
   ./docker-run.sh shell
   ```

### Option 2: Using Docker Compose directly

1. **Build and start the container**:
   ```bash
   docker-compose up -d --build
   ```

2. **Access the container shell**:
   ```bash
   docker exec -it unianimate-dit bash
   ```

## Container Management

### Using the script:
```bash
# Show help
./docker-run.sh help

# Build image
./docker-run.sh build

# Start container
./docker-run.sh run

# Stop container
./docker-run.sh stop

# Restart container
./docker-run.sh restart

# Show logs
./docker-run.sh logs

# Enter shell
./docker-run.sh shell

# Execute a command
./docker-run.sh exec "python --version"
```

### Using Docker Compose:
```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# Stop container
docker-compose down

# Show logs
docker-compose logs -f

# Execute command
docker exec -it unianimate-dit bash -c "your_command"
```

## Directory Structure

The container mounts the following directories:

- **Current directory** → `/workspace` (project files)
- **./outputs** → `/workspace/outputs` (generated videos)
- **./data** → `/workspace/data` (input data)
- **Docker volume** → `/workspace/checkpoints` (model checkpoints)
- **Docker volume** → `/workspace/Wan2.1-I2V-14B-720P` (base models)

## Usage Examples

### 1. Download Models

Once inside the container, download the required models:

```bash
# Download Wan2.1-14B-I2V-720P models
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P

# Download UniAnimate-DiT checkpoints
pip install modelscope
modelscope download xiaolaowx/UniAnimate-DiT --local_dir ./checkpoints
```

### 2. Run Pose Alignment

```bash
# Example pose alignment
python run_align_pose.py \
  --ref_name data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg \
  --source_video_paths data/videos/source_video.mp4 \
  --saved_pose_dir data/saved_pose/WOMEN-Blouses_Shirts-id_00004955-01_4_full
```

### 3. Generate Videos

```bash
# Generate 480P videos
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_480p.py

# Generate 720P videos
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_720p.py
```

### 4. Training

```bash
# Install additional training dependencies (if not already installed)
pip install peft lightning pandas deepspeed

# Run training
CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/train_unianimate_wan.py \
  --task train \
  --train_architecture lora \
  --lora_rank 64 --lora_alpha 64 \
  --dataset_path data/example_dataset \
  --output_path ./models_out_one_GPU \
  --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors" \
  --max_epochs 10 --learning_rate 1e-4 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --use_gradient_checkpointing_offload
```

## Web Interfaces

The container exposes ports for web interfaces:

- **Gradio**: http://localhost:7860
- **Streamlit**: http://localhost:8501

To run the web interfaces:

```bash
# Gradio interface
cd apps/gradio
python DiffSynth_Studio.py

# Streamlit interface
cd apps/streamlit
streamlit run app.py
```

## Troubleshooting

### GPU Issues

1. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```

2. **Verify NVIDIA Docker**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
   ```

3. **Check container GPU access**:
   ```bash
   docker exec -it unianimate-dit nvidia-smi
   ```

### Memory Issues

1. **Reduce VRAM usage**:
   - Set `cfg_scale=1.0` in inference scripts
   - Set `num_persistent_param_in_dit=0`
   - Use `torch.bfloat16` dtype

2. **Enable gradient checkpointing** for training:
   ```bash
   --use_gradient_checkpointing --use_gradient_checkpointing_offload
   ```

### Build Issues

1. **Clear Docker cache**:
   ```bash
   docker system prune -a
   ```

2. **Rebuild without cache**:
   ```bash
   docker-compose build --no-cache
   ```

### Permission Issues

1. **Fix file permissions**:
   ```bash
   sudo chown -R $USER:$USER .
   ```

2. **Add user to docker group**:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

## Performance Tips

1. **Use Flash Attention**: The container includes Flash Attention for better performance
2. **Enable teacache**: Set `use_teacache=True` for ~4x inference acceleration
3. **Multi-GPU**: Use Unified Sequence Parallel (USP) for multi-GPU inference
4. **Optimize VRAM**: Adjust `num_persistent_param_in_dit` based on your GPU memory

## Environment Variables

You can customize the container behavior by setting environment variables in `docker-compose.yml`:

- `NVIDIA_VISIBLE_DEVICES`: Specify which GPUs to use
- `CUDA_VISIBLE_DEVICES`: Control CUDA device visibility
- `PYTHONPATH`: Add custom Python paths

## Support

For issues related to:
- **Docker setup**: Check this document and Docker logs
- **UniAnimate-DiT**: Refer to the main README.md
- **Model issues**: Check the original UniAnimate-DiT repository

## License

This Docker setup follows the same license as the UniAnimate-DiT project. 