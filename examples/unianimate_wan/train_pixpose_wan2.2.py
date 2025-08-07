import os, sys
# insert project root (two levels up from this file) onto Python’s module search path
ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, ROOT)
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import random
import pickle
from io import BytesIO
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import  torch.nn  as nn
from diffsynth.pipelines.wan_video_new import PixPosePipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
from diffsynth.models.utils import load_state_dict, load_state_dict_from_folder
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, max_num_frames=81, frame_interval=2, num_frames=81, height=480, width=832, is_i2v=False):
        self.data_path = data_path
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v

        # data_list = ['UBC_Fashion', 'self_collected_videos_pose', 'TikTok']
        # data_list = ['TikTok']
        data_list = ['UBC_Fashion']
        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.misc_size = [height, width]
        self.video_list = []

        
        if 'TikTok' in data_list:
            
            self.pose_dir = "./data/example_dataset/TikTok/"
            file_list = os.listdir(self.pose_dir)
            print("!!! all dataset length: ", len(file_list))
            # 
            for iii_index in file_list:
                    self.video_list.append(self.pose_dir+iii_index)

            self.use_pose = True
            print("!!! dataset length: ", len(self.video_list))
        
        if 'UBC_Fashion' in data_list:
            self.pose_dir = "/home/bingyin/code/projects/PixPose/V1/datasets/ubc_fashion/training_video_pose/"
            file_list = os.listdir(self.pose_dir)
            print("!!! all dataset length (UBC_Fashion): ", len(file_list))
            
            for iii_index in file_list:
            #     
                    self.video_list.append(self.pose_dir + iii_index)

            self.use_pose = True
            print("!!! dataset length: ", len(self.video_list))
        if 'self_collected_videos_pose' in data_list:
            
            self.pose_dir = "path_of_your_self_data"
            file_list = os.listdir(self.pose_dir)
            print("!!! all dataset length (self_collected_videos_pose): ", len(file_list))
            # 
            for iii_index in file_list:
                
                self.video_list.append(self.pose_dir+iii_index)

            self.use_pose = True
            print("!!! dataset length: ", len(self.video_list))

        random.shuffle(self.video_list)
            
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        # 
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        # return torch.from_numpy(np.array(image))
        return image
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        index = index % len(self.video_list)
        success=False
        for _try in range(5):
            try:
                if _try >0:
                    
                    index = random.randint(1,len(self.video_list))
                    index = index % len(self.video_list)
                
                clean = True
                path_dir = self.video_list[index]

                frames_all = pickle.load(open(path_dir+'/frame_data.pkl','rb'))
                
                dwpose_all = pickle.load(open(path_dir+'/dw_pose_with_foot_wo_face.pkl','rb'))
                # 
                # random sample fps
                stride = random.randint(1, self.sample_fps)
                print("!!! stride in load_video: ", stride)
                
                _total_frame_num = len(frames_all)
                cover_frame_num = (stride * self.max_frames)
                max_frames = self.max_frames
                if _total_frame_num < cover_frame_num + 1:
                    start_frame = 0
                    end_frame = _total_frame_num-1
                    stride = max((_total_frame_num//max_frames),1)
                    end_frame = min(stride*max_frames, _total_frame_num-1)
                else:
                    start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                    end_frame = start_frame + cover_frame_num
                frame_list = []
                dwpose_list = []
                print("!!! start_frame: ", start_frame)
                print("!!! end_frame: ", end_frame)

                random_ref = random.randint(0,_total_frame_num-1)
                i_key = list(frames_all.keys())[random_ref]
                random_ref_frame = Image.open(BytesIO(frames_all[i_key]))
                if random_ref_frame.mode != 'RGB':
                    random_ref_frame = random_ref_frame.convert('RGB')
                random_ref_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                
                first_frame = None
                for i_index in range(start_frame, end_frame, stride):
                    i_key = list(frames_all.keys())[i_index]
                    i_frame = Image.open(BytesIO(frames_all[i_key]))
                    if i_frame.mode != 'RGB':
                        i_frame = i_frame.convert('RGB')
                    i_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                    
                    if first_frame is None:
                        first_frame=i_frame

                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                    else:
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                if (end_frame-start_frame) < max_frames:
                    for _ in range(max_frames-(end_frame-start_frame)):
                        i_key = list(frames_all.keys())[end_frame-1]
                        
                        i_frame = Image.open(BytesIO(frames_all[i_key]))
                        if i_frame.mode != 'RGB':
                            i_frame = i_frame.convert('RGB')
                        i_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                        
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                have_frames = len(frame_list)>0
                middle_indix = 0

                if have_frames:

                    l_hight = random_ref_frame.size[1]
                    l_width = random_ref_frame.size[0]

                    # random crop
                    x1 = random.randint(0, l_width//14)
                    x2 = random.randint(0, l_width//14)
                    y1 = random.randint(0, l_hight//14)
                    y2 = random.randint(0, l_hight//14)
                    
                    
                    random_ref_frame = random_ref_frame.crop((x1, y1,l_width-x2, l_hight-y2))
                    ref_frame = random_ref_frame 
                    # 
                    
                    random_ref_frame_tmp = torch.from_numpy(np.array(self.resize(random_ref_frame)))
                    random_ref_dwpose_tmp = torch.from_numpy(np.array(self.resize(random_ref_dwpose.crop((x1, y1,l_width-x2, l_hight-y2))))) # [3, 512, 320]
                    
                    video_data_tmp = torch.stack([self.frame_process(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2)))) for ss in frame_list], dim=0) # self.transforms(frames)
                    dwpose_data_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in dwpose_list], dim=0)

                video_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                dwpose_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                
                if have_frames:
                    video_data[:len(frame_list), ...] = video_data_tmp      
                    
                    dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
                    
                video_data = video_data.permute(1,0,2,3)
                dwpose_data = dwpose_data.permute(1,0,2,3)
                
                caption = "a person is dancing"
                break
            except Exception as e:
                # 
                caption = "a person is dancing"
                # 
                video_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                random_ref_frame_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
                vit_image = torch.zeros(3,self.misc_size[0], self.misc_size[1])
                
                dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                # 
                random_ref_dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                print('{} read video frame failed with error: {}'.format(path_dir, e))
                continue


        text = caption 
        path = path_dir 
        
        if self.is_i2v:
            video, first_frame = video_data, random_ref_frame_tmp
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame, "dwpose_data": dwpose_data, "random_ref_dwpose_data": random_ref_dwpose_tmp}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        
        return len(self.video_list)



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        tiled=False, tile_size=(34, 34), tile_stride=(18, 16),
        pretrained_lora_path=None,
    ):
        super().__init__()
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = PixPosePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)


        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Get tiler kwargs
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}


        # Define custom embedding models
        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
                    nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                    
                    )


        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        # print("!!! trainable_models: ", trainable_models)
        # Unfreeze the dwpose_embedding and randomref_embedding_pose
        self.dwpose_embedding.train()
        self.randomref_embedding_pose.train()

        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)

        # Lora pretrained lora weights, only used for resume training
        if pretrained_lora_path is not None:
            # 
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            # 
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():
                
                if 'pipe.dit.' in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                if "dwpose_embedding" in key or "randomref_embedding_pose" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new
            state_dict_new = {}

            for key in state_dict_new_module:
                if "dwpose_embedding" in key:
                    state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict_new_module[key]
            self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

            state_dict_new = {}
            for key in state_dict_new_module:
                if "randomref_embedding_pose" in key:
                    state_dict_new[key.split("randomref_embedding_pose.")[1]] = state_dict_new_module[key]
            self.randomref_embedding_pose.load_state_dict(state_dict_new,strict=True)

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["text"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "input_image": data["first_frame"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        pose_condition = self.dwpose_embedding((torch.cat([data["dwpose_data"][:,:,:1].repeat(1,1,3,1,1), data["dwpose_data"]], dim=2)/255.).to(self.device))
        ref_pose = self.randomref_embedding_pose((data["random_ref_dwpose_data"]/255.).to(torch.bfloat16).to(self.device).permute(0,3,1,2)).unsqueeze(2) # [1, 20, 104, 60]
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        print("!!! models: ", models)
        # Add custom embedding models to the models dictionary
        models.update({
            "dwpose_embedding": self.dwpose_embedding,
            "randomref_embedding_pose": self.randomref_embedding_pose
        })
        loss = self.pipe.training_loss(pose_condition, ref_pose, enable_cfg=True, **models, **inputs)
        return loss

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = LoadDataset(args.custom_dataset_path,
        max_num_frames=args.num_frames, 
        frame_interval=1, 
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width, 
        is_i2v=True,
        )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW([model.trainable_modules(), model.dwpose_embedding.parameters(), model.randomref_embedding_pose.parameters()], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

        