# Transformers
import re
import torch
from torch import nn
from typing import Optional, Tuple, Union
from transformers import MambaForCausalLM, MambaConfig
from transformers.models.mamba.modeling_mamba import MambaOutput
from transformers import LlavaNextForConditionalGeneration, LlavaForConditionalGeneration


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

# Dataclass & ModelOutput
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
@dataclass
class MambaCausalLMOutput(ModelOutput):
    cache_params: Optional[MambaCache] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None

class MeteorMambaForCausalLM(MambaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
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

        # logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        return MambaOutput(
            cache_params=mamba_outputs.cache_params,
            last_hidden_state=hidden_states,
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