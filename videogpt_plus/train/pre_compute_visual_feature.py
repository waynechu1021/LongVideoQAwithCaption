# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
from videogpt_plus.constants import *
from torch.utils.data import Dataset, DataLoader
from videogpt_plus.train.trainer import VideoGPTPlusTrainer
from videogpt_plus import conversation as conversation_lib
from videogpt_plus.model import *
from videogpt_plus.mm_utils import tokenizer_image_token
from videogpt_plus.config import DataConfig
from PIL import Image
import random
import numpy as np
from videogpt_plus.model.dataloader import _get_rawvideo_dec
from transformers.trainer import TrainerCallback
from videogpt_plus.model.multimodal_encoder.builder import build_vision_tower
from tqdm import tqdm
from einops import rearrange

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct")
    version: Optional[str] = field(default="phi3_instruct")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_image_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    image_mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_box_start_end: bool = field(default=False)
    visual_token_compression_rate: Optional[int] = field(default=2)


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)

    dataset_use: str = field(default="FINETUNING")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    seed = 42

class LogCallback(TrainerCallback):
    def __init__(self, logging):
        self.logging = logging

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logging.info(logs)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        dataset_list = DataConfig[str(data_args.dataset_use)]
        print(f"Loading datasets: {dataset_list}")

        # Read all the datasets and populate list_data_dict
        list_data_dict = []
        save_path_list = []
        for data in dataset_list:
            annotations = json.load(open(data["annotation_path"], "r"))
            for ann in annotations:
                ann['data_path'] = data['data_path']
                ann['save_path'] = data['feature_path']+'_'+str(NUM_FRAMES)
                save_path_list.append(ann['save_path'])
            list_data_dict += annotations
        print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        # Populate the class variables
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        save_path_list = list(set(save_path_list))
        for save_path in save_path_list:
            os.makedirs(save_path, exist_ok=True)
            if 'k710' in save_path:
                os.makedirs(os.path.join(save_path,'k400'), exist_ok=True)
                os.makedirs(os.path.join(save_path,'k600'), exist_ok=True)
                os.makedirs(os.path.join(save_path,'k700'), exist_ok=True)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        if "video" in sources:
            video_folder = sources['data_path']
            video_file = sources['video']
            video_file_path = os.path.join(video_folder, video_file)
            try:
                # Note that currently only DUAL Encoder configuration is supported,
                # so both image and video processor should not be None
                image_processor = self.data_args.image_processor
                video_processor = self.data_args.video_processor
                frames, context_images = _get_rawvideo_dec(video_file_path, image_processor, video_processor,
                                                           frame_resolution=224, max_frames=NUM_FRAMES,
                                                           num_video_frames=NUM_FRAMES,
                                                           num_context_images=NUM_CONTEXT_IMAGES)
                data_dict = {'video_name':video_file,'save_path':sources['save_path']}
            except Exception as e:
                print(f"Caught exception {e} when loading video {video_file_path}. Sampling a random video.")
                index = np.random.randint(0, len(self))
                return self.__getitem__(index)

        else:
            raise NotImplementedError

        # Image exists in the data

        if 'video' in self.list_data_dict[i]:
            data_dict['image'] = frames
            data_dict['context_image'] = context_images
        else:
            # image does not exist in the data, but the model is multimodal
            raise NotImplementedError

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        video_name = [instance['video_name'] for instance in instances]
        save_path = [instance['save_path'] for instance in instances]
        batch = dict(
            video_name = video_name,
            save_path = save_path
        )

        if 'image' in instances[0]:  # Alternatively if 'context_image' in instances[0]
            images = [instance['image'] for instance in instances]
            context_images = [instance['context_image'] for instance in instances]

            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            new_context_images = []
            for image in context_images:
                if type(image) is list:
                    for i in image:
                        new_context_images.append(i)
                else:
                    new_context_images.append(image)
            context_images = new_context_images

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                batch['context_images'] = torch.stack(context_images)
            else:
                batch['images'] = images
                batch['context_images'] = context_images

        return batch


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.bfloat16
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if model_args.vision_tower is not None:
        model_args.vision_tower = f"{model_args.vision_tower}/InternVideo2-stage2_1b-224p-f4.pt"
    vision_tower = build_vision_tower(model_args, image_vision_tower=False).cuda().to(compute_dtype)
    image_vision_tower = build_vision_tower(model_args, image_vision_tower=True).cuda().to(compute_dtype)

    data_args.video_processor = vision_tower.image_processor
    data_args.image_processor = image_vision_tower.image_processor

    bs = 32
    video_token_num = NUM_FRAMES // CHUNK_SIZE
    train_dataset = LazySupervisedDataset(data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset()
    data_loader = DataLoader(train_dataset,batch_size=bs,shuffle=False,collate_fn=data_collator,num_workers=4)
    for data in tqdm(data_loader):
        video_name = data['video_name']
        save_path = data['save_path']

        
        frames = data['images'].cuda().to(compute_dtype)
        context_images = data['context_images'].cuda().to(compute_dtype)
        batch_size = len(video_name)

        frames = rearrange(frames, '(b t) c h w -> b t c h w', b=batch_size)
        num_chunks = frames.shape[1] // CHUNK_SIZE
        L = 256  # Number of features per frame from InternVideo2-Stage2_1B-224p-f4
        D = 1408  # Feature dimension of InternVideo2-Stage2_1B-224p-f4

        video_features = torch.zeros(batch_size, num_chunks, 4 * L, D, device=frames.device, dtype=frames.dtype)
        for i in range(batch_size):
            cur_video = frames[i]  # Current video of shape (t, c, h, w)
            chunks = cur_video.chunk(num_chunks, dim=0)
            # New batch dimension for processing all chunks at once
            chunk_batch = torch.stack(chunks, dim=0)  # (num_chunks, 4, c, h, w)
            chunk_features = vision_tower(chunk_batch)  # (num_chunks, 4*L, D)
            # Store the features in the output tensor - Only storing feature - remove cls
            video_features[i] = chunk_features[:, 1:]

        video_features = rearrange(video_features, 'b p (c l) d -> (b p) (c l) d', c=CHUNK_SIZE)
        context_image_features = image_vision_tower(context_images, select_feature="patch")
        context_image_features = rearrange(context_image_features, '(b t) l d -> b t l d', b=batch_size)

        for idx,(name,path) in enumerate(zip(video_name,save_path)):
            cur_video_features = video_features[idx*video_token_num:(idx+1)*video_token_num]
            torch.save(cur_video_features.to('cpu'),os.path.join(path,name+'_video_feature.pt'))
            cur_context_image_features = context_image_features[idx]
            torch.save(cur_context_image_features.to('cpu'),os.path.join(path,name+'_context_feature.pt'))
            

if __name__ == "__main__":
    train()