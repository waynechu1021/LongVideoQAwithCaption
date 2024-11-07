import os

# DATASET_DIR = os.environ.get("DATASET_DIR", "playground/data")
DATASET_DIR = os.environ.get("DATASET_DIR", "/ssd1/zwy/VideoMeteor/playground/VideoGPT-plus_Training_Dataset")

MOMENTOR = {
    "annotation_path": f"/ssd1/zwy/VideoMeteor/playground/Moment-10M_for_stage1_43k.json",
    "data_path": f"/ssd1/zwy/VideoMeteor/playground/Momentor_video",
}

MOMENTOR_CAP_QA = {
    "annotation_path": f"/ssd1/zwy/VideoMeteor/playground/Moment-10M_for_stage1_43k_for_baseline_ablation.json",
    "data_path": f"/ssd1/zwy/VideoMeteor/playground/Momentor_video",
}

MOMENTOR_QA = {
    "annotation_path": f"/ssd1/zwy/VideoMeteor/playground/Moment-10M_1_QA.json",
    "data_path": f"/ssd1/zwy/VideoMeteor/playground/Momentor_video",
}
MOMENTOR_cross_segment_QA = {
    "annotation_path": f"/ssd1/zwy/VideoMeteor/playground/Moment-10M_cross_segment_QA.json",
    "data_path": f"/ssd1/zwy/VideoMeteor/playground/Momentor_video",
}

CC3M_595K = {
    "annotation_path": f"{DATASET_DIR}/pretraining/CC3M-595K/chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/CC3M-595K",
}

COCO_CAP = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_cap_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REG = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_reg_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REC = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_rec_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

CONV_VideoChatGPT = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_HUMAN = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg_human_annotated.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_PLUS_112K = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg-plus_112K.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

CAPTION_VIDEOCHAT = {
    "annotation_path": f"{DATASET_DIR}/annotations/caption_videochat.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}

CLASSIFICATION_K710 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
    "feature_path":f"{DATASET_DIR}/instruction_tuning/k710_feature",
}

CLASSIFICATION_SSV2 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
    "feature_path":f"{DATASET_DIR}/instruction_tuning/ssv2_feature",
}

CONV_VideoChat1 = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochat1.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/videochat_it",
    "feature_path":f"{DATASET_DIR}/instruction_tuning/videochat_it_feature",
}

REASONING_NExTQA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}

REASONING_CLEVRER_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
    "feature_path":f"{DATASET_DIR}/instruction_tuning/clevrer_feature",
}

REASONING_CLEVRER_MC = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
    "feature_path": f"{DATASET_DIR}/instruction_tuning/clevrer_feature",
}

VQA_WEBVID_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
