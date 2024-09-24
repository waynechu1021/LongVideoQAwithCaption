#!/bin/sh

export PYTHONPATH=`pwd`:$PYTHONPATH
export DATASET_DIR=playground/data

BASE_LLM_PATH=.cache/Phi-3-mini-4k-instruct-previous-version
MODEL_NAME=videogpt_plus_finetune_wo_caption
MODEL_PATH=results/${MODEL_NAME}
ANSWERS_FILE=${MODEL_PATH}/answer.jsonl


CUDA_VISIBLE_DEVICES=6 python -m eval.model_vqa_loader \
    --model-base ${BASE_LLM_PATH} \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/Moment-10M-eval-QA.json \
    --video-folder ./playground/eval \
    --output-dir ${ANSWERS_FILE} \
    --temperature 0 \
