# Transformers
import re
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List
from transformers import MambaForCausalLM, MambaConfig, MambaModel
from transformers.models.mamba.modeling_mamba import MambaCausalLMOutput
from videogpt_plus.model.arch import MetaModel, VideoGPTPlusMetaForCausalLM


class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }

class VideoGPTPlusMambaConfig(MambaConfig):
    model_type = "VideoGPT+4mamba"


class VideoGPTPlusMambaModel(MetaModel, MambaModel):
    config_class = VideoGPTPlusMambaConfig

    def __init__(self, config: MambaConfig):
        super(VideoGPTPlusMambaModel, self).__init__(config)

class VideoGPTPlusMambaForCausalLM(MambaForCausalLM,VideoGPTPlusMetaForCausalLM):
    config_class = VideoGPTPlusMambaConfig
    def __init__(self, config):
        super(MambaForCausalLM, self).__init__(config)
        self.backbone = VideoGPTPlusMambaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.backbone
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_params: Optional[MambaCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        context_images: Optional[torch.FloatTensor] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, context_images)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert inputs_embeds is not None, "inputs_embeds should not be None for mamba"

        mamba_outputs = self.backbone(
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable videogpt_plus/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return (loss,) + output if loss is not None else output
        return MambaCausalLMOutput(
            loss = loss,
            cache_params=mamba_outputs.cache_params,
            logits=logits,
            hidden_states=mamba_outputs.hidden_states
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds,}
        else:
            model_inputs = {"input_ids": input_ids,}

        model_inputs["cache_params"] = cache_params
        return model_inputs