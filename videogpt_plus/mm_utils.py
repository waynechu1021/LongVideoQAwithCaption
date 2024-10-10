from PIL import Image
from io import BytesIO
import base64
import torch
from transformers import StoppingCriteria
from videogpt_plus.constants import IMAGE_TOKEN_INDEX
import deepspeed, logging

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

def load_zero_partitions(model, state_dict, is_deepspeed_zero3_enabled, pretrained_model_path, ignore_mismatched_sizes=False):
    """
    adept from pytorch lightning and transformers
    with deepspeed.zero.Init():
        model = MyModel()
    state_dict = torch.load(model_path, map_location="cpu")
    load_zero_partitions(model, prefix="")
    """
    
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    loaded_keys = list(state_dict.keys())
    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))

    # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
    # matching the weights in the model.
    mismatched_keys = []
    if ignore_mismatched_sizes:
        for checkpoint_key in loaded_keys:
            model_key = checkpoint_key

            if (
                model_key in model_state_dict
                and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
            ):
                mismatched_keys.append(
                    (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                )
                del state_dict[checkpoint_key]
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        if is_deepspeed_zero3_enabled:
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    module._load_from_state_dict(*args)
        else:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = model
    load(model_to_load, prefix=start_prefix)
    del state_dict
    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
    if len(unexpected_keys) > 0:
        logging.warning(
            f"Some weights of the model checkpoint at {pretrained_model_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
            " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
            f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
            " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
    else:
        logging.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logging.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    elif len(mismatched_keys) == 0:
        logging.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_path}.\nIf your task is similar to the task the model of the checkpoint"
            f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
            " training."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logging.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_path} and are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )