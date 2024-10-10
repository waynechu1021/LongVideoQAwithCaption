#!/bin/sh

export PYTHONPATH=`pwd`:$PYTHONPATH
export DATASET_DIR=playground/data

BASE_LLM_PATH=.cache/Phi-3-mini-128k-instruct
VISION_TOWER=.cache/InternVideo2-Stage2_1B-224p-f4
IMAGE_VISION_TOWER=.cache/clip-vit-large-patch14-336
PROJECTOR_TYPE=mlp2x_gelu
PRETRAIN_VIDEO_MLP_PATH=.cache/phi3_mini_4k_128k_pretrain/mlp2x_gelu_internvideo2/mm_projector.bin
PRETRAIN_IMAGE_MLP_PATH=.cache/phi3_mini_4k_128k_pretrain/mlp2x_gelu_clip_l14_336px/mm_projector.bin
OUTPUT_DIR_PATH=results/videogpt_plus_finetune_with_caption_stage1_test

# deepspeed --include localhost:6 videogpt_plus/train/train.py \
CUDA_VISIBLE_DEVICES=7 deepspeed --master_port 25400 videogpt_plus/train/train.py \
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
--deepspeed scripts/zero3.json \
--model_name_or_path "$BASE_LLM_PATH" \
--version phi3_instruct \
--dataset_use STAGE2 \
--use_caption False \
--stage 2 \
--max_num_of_tor 20 \
--vision_tower "$VISION_TOWER" \
--image_vision_tower "$IMAGE_VISION_TOWER" \
--mamba_name_or_path '.cache/mamba-130m-hf' \
--mm_projector_type "$PROJECTOR_TYPE" \
--image_mm_projector_type "$PROJECTOR_TYPE" \
--pretrain_mm_mlp_adapter "$PRETRAIN_VIDEO_MLP_PATH" \
--pretrain_image_mm_mlp_adapter "$PRETRAIN_IMAGE_MLP_PATH" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length False \
--bf16 True \
--output_dir $OUTPUT_DIR_PATH \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "no" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 131072 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to tensorboard
