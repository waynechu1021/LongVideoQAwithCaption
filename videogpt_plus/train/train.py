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
from torch.utils.data import Dataset
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

    pretrained_tor_and_projector_module: Optional[str] = field(default=None)
    pretrain_mamba_module: Optional[str] = field(default=None)
    mamba_name_or_path: Optional[str] = field(default=None)
    max_num_of_tor: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)

    dataset_use: str = field(default="FINETUNING")
    use_caption: bool = False


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
    stage:int = 1

class LogCallback(TrainerCallback):
    def __init__(self, logging):
        self.logging = logging

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logging.info(logs)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_mm_mlp_adapter", False):
    #     # Only save Adapter
    #     keys_to_match = ['mm_projector', "ctm", "block"]
    #     if getattr(trainer.args, "use_im_start_end", False):
    #         keys_to_match.extend(['embed_tokens', 'embed_in'])

    #     weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    #     trainer.model.config.save_pretrained(output_dir)

    #     current_folder = output_dir.split('/')[-1]
    #     parent_folder = os.path.dirname(output_dir)
    #     if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    #         if current_folder.startswith('checkpoint-'):
    #             mm_projector_folder = os.path.join(parent_folder, "mm_projector")
    #             os.makedirs(mm_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

    if trainer.args.stage == 1:
        keys_to_match = ['mamba']
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.get_model().mamba.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mamba_module")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mamba_module.bin'))
        
        keys_to_match = ["tor_embedding", "tor_projector", "vision_projector", "image_vision_projector"]
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "tor_and_projector_module")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'tor_and_projector_module.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for i, source in enumerate(sources):
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] or DEFAULT_VIDEO_TOKEN in sentence['value']:
                if sentence['value'].endswith(DEFAULT_IMAGE_TOKEN):
                    IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                    sentence['value'] = sentence['value'].replace('\n' + DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM,
                                                                  '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                if sentence['value'].endswith(DEFAULT_VIDEO_TOKEN):
                    VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                    sentence['value'] = sentence['value'].replace('\n' + DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM,
                                                                  '').strip()
                    sentence['value'] = DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > NUM_FRAMES:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM,
                                                                  DEFAULT_IMAGE_TOKEN * NUM_FRAMES).strip()

            replace_token, vid_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * NUM_FRAMES
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN
            if data_args.use_caption and sentence['from'] == 'gpt':
                dense_caption = ''
                cap = sentence['value']
                for item in cap:
                    dense_caption += DEFAULT_TOR_TOKRN + '\n'
                    dense_caption += str(round(item['moment'][0],2)) + '-' + str(round(item['moment'][1],2))
                    dense_caption += ' '
                    dense_caption += item['content']
                sentence['value'] = dense_caption
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)

    return sources


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi3(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len -= 2
                instruction_len -= 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mamba(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.conv_mamba.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        # target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1


            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_meteor(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        mamba_tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        stage = 1
) -> Dict:
    phi3_dict = preprocess_phi3(sources,tokenizer,has_image)
    mamba_dict = preprocess_mamba(sources, mamba_tokenizer, has_image)
    # return dict(
    #     input_ids=phi3_dict['input_ids'],
    #     input_ids_llm = phi3_dict['labels'][phi3_dict['labels']>=0].unsqueeze(0) if stage == 1 else phi3_dict['input_ids'],
    #     labels = phi3_dict['labels'][phi3_dict['labels']>=0].unsqueeze(0) if stage == 1 else phi3_dict['labels'],
    # )
    tmp = phi3_dict['labels'][phi3_dict['labels']!=TOR_TOKEN_INDEX]
    return dict(
        input_ids=mamba_dict['input_ids'],
        input_ids_llm = phi3_dict['labels'][phi3_dict['labels']>=0].unsqueeze(0) if stage == 1 else phi3_dict['input_ids'],
        labels = tmp[tmp>=0].unsqueeze(0) if stage == 1 else phi3_dict['labels'],
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN * source[0]['value'].count(DEFAULT_IMAGE_TOKEN)
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        mamba_tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        stage = 1,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "phi3":
        return preprocess_meteor(sources, tokenizer, mamba_tokenizer, has_image=has_image, stage = stage)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 mamba_tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        dataset_list = DataConfig[str(data_args.dataset_use)]
        rank0_print(f"Loading datasets: {dataset_list}")

        # # Read all the datasets and populate list_data_dict
        list_data_dict = []
        for data in dataset_list:
            annotations = json.load(open(data["annotation_path"], "r"))
            for ann in annotations:
                ann['data_path'] = data['data_path']
            list_data_dict += annotations
        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        # Populate the class variables
        self.tokenizer = tokenizer
        self.mamba_tokenizer = mamba_tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def modality_lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
    #         cur_len = cur_len if 'image' in sample else -cur_len
    #         length_list.append(cur_len)
    #     return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if 'image' in sources:
            image_folder = sources['data_path']
            image_file = sources['image']

            image = Image.open(os.path.join(image_folder, image_file.replace("\\", "/"))).convert('RGB')

            image_processor = self.data_args.image_processor
            video_processor = self.data_args.video_processor
            if image_processor:
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif video_processor:
                image = video_processor.preprocess([np.array(image)], use_image=True)['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            data_dict = preprocess(
                sources,
                self.tokenizer,
                self.mamba_tokenizer, 
                has_image=True, 
                stage = self.data_args.stage
                )

        elif "video" in sources:
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
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in [sources]]),
                    self.data_args)
                data_dict = preprocess(
                    sources,
                    self.tokenizer,
                    self.mamba_tokenizer, 
                    has_image=True,
                    stage = self.data_args.stage)
            except Exception as e:
                print(f"Caught exception {e} when loading video {video_file_path}. Sampling a random video.")
                index = np.random.randint(0, len(self))
                return self.__getitem__(index)

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess(
                sources,
                self.tokenizer,
                self.mamba_tokenizer, 
                has_image=False,
                stage = self.data_args.stage)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], input_ids_llm = data_dict["input_ids_llm"][0])

        # Image exists in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['context_image'] = None
        elif 'video' in self.list_data_dict[i]:
            data_dict['image'] = frames
            data_dict['context_image'] = context_images
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    mamba_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids_llm = [instance['input_ids_llm'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.mamba_tokenizer.pad_token_id)
        input_ids_llm = torch.nn.utils.rnn.pad_sequence(
            input_ids_llm,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        input_ids_llm = input_ids_llm[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            input_ids_llm=input_ids_llm,
            labels=labels,
            attention_mask=input_ids.ne(self.mamba_tokenizer.pad_token_id),
            attention_mask_llm=input_ids_llm.ne(self.tokenizer.pad_token_id),
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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                mamba_tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, mamba_tokenizer=mamba_tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, mamba_tokenizer = mamba_tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                                       datefmt="%m/%d/%Y %H:%M:%S", )
        os.makedirs(training_args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(training_args.output_dir, f"train.log"))
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging.INFO)

        logging.info("Training/evaluation parameters %s", training_args)
        logging.info("Model parameters %s", model_args)
        logging.info("Data parameters %s", data_args)
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    random.seed(training_args.seed)
    os.environ['PYTHONHASHSEED'] = str(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)  # If you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    # At least one of image or videos towers should not be None
    # Note that vision_tower only or image_vision_tower only configuration is only supported in pretraining
    # For fine-tuning both vision_tower and image_vision_tower should not be None
    assert model_args.vision_tower is not None or model_args.image_vision_tower is not None

    # NOTE: Be mindful of this if using a custom video encoder
    if model_args.vision_tower is not None:
        model_args.vision_tower = f"{model_args.vision_tower}/InternVideo2-stage2_1b-224p-f4.pt"

    model = VideoGPTPlusPhi3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["phi3_instruct"]

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    if model_args.vision_tower is not None:
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.float16, device=training_args.device)
        data_args.video_processor = vision_tower.image_processor
    else:
        data_args.video_processor = None

    if model_args.image_vision_tower is not None:
        image_vision_tower = model.get_image_vision_tower()
        image_vision_tower.to(dtype=torch.float16, device=training_args.device)
        data_args.image_processor = image_vision_tower.image_processor
    else:
        data_args.image_processor = None

    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.requires_grad_(False)
    if model_args.tune_mm_mlp_adapter:
        # model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().image_mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        for p in model.get_model().image_mm_projector.parameters():
            p.requires_grad = False
    print('projector grad enable = ',all(p.requires_grad for p in model.get_model().image_mm_projector.parameters()))
    print('mm_projector grad enable = ',all(p.requires_grad for p in model.get_model().mm_projector.parameters()))
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        model.get_model().image_mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_projector_lr = training_args.mm_projector_lr

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_im_start_end = model_args.mm_use_im_start_end

    model.config.mm_use_box_start_end = data_args.mm_use_box_start_end = model_args.mm_use_box_start_end
    training_args.use_im_start_end = model_args.mm_use_box_start_end

    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model_args.use_caption = data_args.use_caption
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Freeze the vision towers (I know they are already frozen, but let's just do it anyway, maybe helpful in case someone wants to finetune it!)
    if model_args.vision_tower is not None:
        for p in model.get_vision_tower().parameters():
            p.requires_grad = False
    if model_args.image_vision_tower is not None:
        for p in model.get_image_vision_tower().parameters():
            p.requires_grad = False
    mamba_tokenizer = None
    if model_args.mamba_name_or_path is not None:
        mamba_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.mamba_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right")
        mamba_tokenizer.add_tokens([DEFAULT_TOR_TOKRN],special_tokens=True)
        model.config.stage = data_args.stage = training_args.stage
        
        model.get_model().initialize_mamba_and_tor_modules(model_args)
        for p in model.get_model().mamba.parameters():
            p.requires_grad = True
        for p in model.get_model().tor_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().vision_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().image_vision_projector.parameters():
            p.requires_grad = True
        model.get_model().tor_embedding.requires_grad = True
    trainable_params = []
    for k,v in model.named_parameters():
        if v.requires_grad:
            trainable_params.append(k)
    logging.info(f'trainable_params = {trainable_params}')
    print(f'trainable_params = {trainable_params}')
    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")  # FIXME: Sometimes it shows wrong total params, not sure why
    print(f"Trainable parameters: {trainable_params}")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, mamba_tokenizer=mamba_tokenizer, data_args=data_args)
    print(f"Total training samples :{len(data_module['train_dataset'])}")

    log_callback = LogCallback(logging)
    trainer = VideoGPTPlusTrainer(model=model, 
                                tokenizer=tokenizer,
                                args=training_args,
                                callbacks=[log_callback],
                                **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
