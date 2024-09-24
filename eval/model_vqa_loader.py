import sys
sys.path.append('/ssd1/zhuweiye/VideoMeteor')
print(sys.path)
import json
from torch.utils.data import Dataset
import torch
import subprocess
from videogpt_plus.constants import *
from eval.video_encoding import _get_rawvideo_dec
from videogpt_plus.mm_utils import tokenizer_image_token, get_model_name_from_path
import shortuuid
from videogpt_plus.conversation import conv_templates
from videogpt_plus.model.builder import load_pretrained_model
import argparse
import traceback
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        print('Using distributed mode: 1')
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '3460')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
        print('Using distributed mode: slurm')
        print(f"world: {os.environ['WORLD_SIZE']}, rank:{os.environ['RANK']},"
              f" local_rank{os.environ['LOCAL_RANK']}, local_size{os.environ['LOCAL_SIZE']}")
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class EvalDatasetGeneric(Dataset):
    def __init__(self, qa_path, video_dir, image_processor, video_processor):
        with open(qa_path) as file:
            self.gt_contents = json.load(file)
        self.video_dir = video_dir
        self.image_processor = image_processor
        self.video_processor = video_processor

        # self.video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __len__(self):
        return len(self.gt_contents)

    def __getitem__(self, idx):
        sample = self.gt_contents[idx]
        video_name = sample['video']
        sample_set = sample

        # Load the video file
        # for fmt in self.video_formats:  # Added this line
        #     temp_path = os.path.join(self.video_dir, f"{video_name}{fmt}")
        #     if os.path.exists(temp_path):
        #         video_path = temp_path
        #         break
        video_path = os.path.join(self.video_dir, f"{video_name}")

        # Check if the video exists
        if os.path.exists(video_path):  # Modified this line
            video_frames, context_frames, slice_len = _get_rawvideo_dec(video_path, self.image_processor,
                                                                        self.video_processor,
                                                                        max_frames=NUM_FRAMES,
                                                                        image_resolution=224,
                                                                        num_video_frames=NUM_FRAMES,
                                                                        num_context_images=NUM_CONTEXT_IMAGES)
        else:
            print(f'Video {video_path} not found')
            video_frames, context_frames, slice_len = "None", "None", 0

        return idx, [sample_set], video_frames, context_frames, slice_len


def eval_model(args):
    # Model
    device = torch.device('cuda:6')
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    vision_tower.load_model(model.config.mm_vision_tower)
    video_processor = vision_tower.image_processor

    image_vision_tower = model.get_image_vision_tower()
    image_vision_tower.load_model()
    image_processor = image_vision_tower.image_processor

    # image_mm_projector_weights = torch.load('.cache/VideoGPT-plus_Phi3-mini-4k_Pretrain/mlp2x_gelu_clip_l14_336px/mm_projector.bin')
    # msg = model.model.image_mm_projector.load_state_dict(get_w(image_mm_projector_weights, 'mm_projector'))

    model = model.to(device)
    model = model.to(torch.bfloat16)

    dataset = EvalDatasetGeneric(args.question_file, args.video_folder, image_processor, video_processor)
    # distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=4, sampler=distributed_sampler)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=0, shuffle=True)

    for (idx, sample_set, video_frames, context_frames, slice_len) in tqdm(dataloader):
        idx, sample_set, video_frames, context_frames, slice_len = int(idx[0]), sample_set[
            0], video_frames, context_frames, int(slice_len[0])

        sample = sample_set
        qs = sample['Q'][0]

        try:
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).to(device)

            stop_str = "<|end|>"

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=torch.cat(video_frames, dim=0).to(device).to(torch.bfloat16),
                    context_images=torch.cat(context_frames, dim=0).to(device).to(torch.bfloat16),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs.replace("<|end|>", '')
            outputs = outputs.strip()
            print(outputs)

            ans_id = shortuuid.uuid()
            results = {'video_name': sample['video'][0],
                       "prompt": cur_prompt,
                       "text": outputs,
                       "answer_id": ans_id,
                       "model_id": model_name,
                       "answer": sample['A'][0],
                       "metadata": {}}
            with open(f"{args.output_dir}/{sample['video'][0]}_{idx}.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            trace = traceback.format_exc()
            print(f"Error processing video file '{sample['video'][0]}': {e}")
            print("Detailed traceback:")
            print(trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="results/videogpt_plus_finetune_wo_caption")
    parser.add_argument("--model-base", type=str, default=".cache/Phi-3-mini-4k-instruct-previous-version")
    parser.add_argument("--video-folder", type=str, default="playground/eval")
    parser.add_argument("--question-file", type=str, default="./playground/Moment-10M-eval-QA.json")
    parser.add_argument("--output-dir", type=str, default="results/videogpt_plus_finetune_wo_caption/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="phi3_instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    init_distributed_mode(args)

    os.makedirs(args.output_dir, exist_ok=True)
    
    eval_model(args)