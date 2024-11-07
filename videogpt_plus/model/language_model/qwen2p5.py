from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Model, Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from videogpt_plus.model.arch import MetaModel, VideoGPTPlusMetaForCausalLM
from videogpt_plus.constants import *


class VideoGPTPlusQwen2Config(Qwen2Config):
    model_type = "VideoGPT+_Qwen2"
    tor_token_index = TOR_TOKEN_INDEX
    tor_token_index_mamba = TOR_TOKEN_INDEX_MAMBA
    image_token_index = IMAGE_TOKEN_INDEX


class VideoGPTPlusQwen2Model(MetaModel, Qwen2Model):
    config_class = VideoGPTPlusQwen2Config

    def __init__(self, config: Qwen2Config):
        super(VideoGPTPlusQwen2Model, self).__init__(config)


class VideoGPTPlusQwen2ForCausalLM(Qwen2ForCausalLM, VideoGPTPlusMetaForCausalLM):
    config_class = VideoGPTPlusQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VideoGPTPlusQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids_mamba: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            attention_mask_mamba: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            context_images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if getattr(self.get_model(),'mamba',None) is not None:
            if getattr(self.config,'stage',None) == 1:
                input_ids_mamba, input_ids, attention_mask_mamba, attention_mask, past_key_values, inputs_embeds, inputs_embeds_llm, labels = self.prepare_inputs_labels_for_meteor(
                    input_ids_mamba, input_ids, attention_mask_mamba, attention_mask, past_key_values, labels, images, context_images, self.config.stage)
                '''examine the input_embed'''
                # for i in range(inputs_embeds.shape[0]):
                #     tor_token_index = torch.where(input_ids[i]==self.config.tor_token_index)
                #     input_ids_llm_tor = input_ids[i][tor_token_index]
                #     inputs_embeds_llm_tor = inputs_embeds_llm[i][tor_token_index]
                #     tor_token_index = torch.where(input_ids_mamba[i]==self.config.tor_token_index_mamba)
                #     input_ids_tor = input_ids_mamba[i][tor_token_index]
                #     inputs_embeds_tor = inputs_embeds[i][inputs_embeds.shape[1] - input_ids_mamba.shape[1]:][tor_token_index]
                #     tor_embed = self.get_model().tor_embedding[:inputs_embeds_tor.shape[0]]
                #     iftrue = torch.all(inputs_embeds_tor == tor_embed)
                #     print(iftrue)
                mamba_outputs = self.get_model().mamba(
                    inputs_embeds = inputs_embeds,
                    return_dict = return_dict,
                )
                last_hidden_states = mamba_outputs.last_hidden_state.to(inputs_embeds_llm.dtype)
                last_hidden_states = last_hidden_states[:,inputs_embeds.shape[1] - input_ids_mamba.shape[1]:]
                # the number of event is not always the same in a batch
                inputs_embeds_llm = self.merge_input_embeds_with_tor_features(last_hidden_states,input_ids_mamba,input_ids,inputs_embeds_llm)
                outputs = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds_llm,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
                #FIXME there is a BUG of device assertion error when using debugpy mode
                hidden_states = outputs[0]
                hidden_states = self.remove_tor_features(hidden_states,input_ids,labels)
                logits = self.lm_head(hidden_states)
            elif getattr(self.config,'stage',None) == 2:
                input_ids_mamba, input_ids, attention_mask_mamba, attention_mask, past_key_values, inputs_embeds, inputs_embeds_llm, labels = self.prepare_inputs_labels_for_meteor(
                    input_ids_mamba, input_ids, attention_mask_mamba, attention_mask, past_key_values, labels, images, context_images, self.config.stage)
                if input_ids.shape[1] > 1:
                    '''examine the input_embed only suitable for stage2 and without tune_mm_mlp_adapter and mm_use_im_start_end'''
                    # for i in range(inputs_embeds.shape[0]):
                    #     image_token_index = torch.where(input_ids[i]==self.config.image_token_index)
                    #     input_ids_llm_image = input_ids[i][image_token_index]
                    #     visual_token_length = NUM_FRAMES*int(16/self.config.visual_token_compression_rate)**2 + NUM_CONTEXT_IMAGES*int(24/self.config.visual_token_compression_rate)**2
                    #     # inputs_embeds_llm_image = inputs_embeds_llm[i][image_token_index[0][-1]-NUM_FRAMES+visual_token_length+1:image_token_index[0][-1]-NUM_FRAMES+visual_token_length+1+self.get_model().max_num_of_tor]
                    #     # preds_index = torch.where(labels[i] != IGNORE_INDEX)[0]
                    #     # inputs_embeds_llm_image = inputs_embeds_llm[i][preds_index[0]-self.get_model().max_num_of_tor:preds_index[0]]
                    #     inputs_embeds_llm_image = inputs_embeds_llm[i][-self.get_model().max_num_of_tor-1:-1]
                    #     tor_embed = self.get_model().tor_projector(self.get_model().tor_embedding[:self.get_model().max_num_of_tor])
                    #     iftrue = torch.all(inputs_embeds_llm_image == tor_embed)
                    #     print(iftrue)
                    #     image_token_index = torch.where(input_ids_mamba[i]==self.config.image_token_index)
                    #     input_ids_image = input_ids_mamba[i][image_token_index]
                    #     inputs_embeds_image = inputs_embeds[i][image_token_index[0][-1]-NUM_FRAMES+visual_token_length+1:image_token_index[0][-1]-NUM_FRAMES+visual_token_length+1+self.get_model().max_num_of_tor]
                    #     tor_embed = self.get_model().tor_embedding[:self.get_model().max_num_of_tor]
                    #     iftrue = torch.all(inputs_embeds_image == tor_embed)
                    #     print(iftrue)
                    mamba_outputs = self.get_model().mamba(
                        inputs_embeds = inputs_embeds,
                        return_dict = return_dict,
                    )
                    last_hidden_states = mamba_outputs.last_hidden_state.to(inputs_embeds_llm.dtype)
                    # the number of event is not always the same in a batch
                    inputs_embeds_llm = self.merge_input_embeds_with_tor_features(last_hidden_states,input_ids_mamba,input_ids,inputs_embeds_llm, stage=self.config.stage,labels=labels)
                    # new_inputs_embed_llm = []
                    # for i in range(inputs_embeds.shape[0]):
                    #     image_token_index = torch.where(input_ids[i]==self.config.image_token_index)
                    #     inputs_embeds_llm_before_image = inputs_embeds_llm[i][:image_token_index[0][0]]
                    #     inputs_embeds_llm_after_image = inputs_embeds_llm[i][image_token_index[0][-1]-16+3329:]
                    #     new_inputs_embed_llm.append(torch.cat([inputs_embeds_llm_before_image,inputs_embeds_llm_after_image],dim=0))
                    # new_inputs_embed_llm = torch.stack(new_inputs_embed_llm,dim=0)
                    # assert inputs_embeds_llm.shape[1] - new_inputs_embed_llm.shape[1] == 3328
                    # inputs_embeds_llm = new_inputs_embed_llm
                outputs = self.model(
                    input_ids=None if input_ids.shape[1] > 1 else input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds_llm,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states)
            else:
                raise NotImplementedError
        else:
            #just for normal training without meteor mamba
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, context_images)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if getattr(self.config,'stage',None) == 1:
                shift_logits = logits
                shift_labels = labels
            else:
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
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "context_images": kwargs.get("context_images", None),
                "input_ids_mamba": kwargs.get("input_ids_mamba", None),
            }
        )
        return model_inputs


AutoConfig.register("VideoGPT+_Qwen2", VideoGPTPlusQwen2Config)
AutoModelForCausalLM.register(VideoGPTPlusQwen2Config, VideoGPTPlusQwen2ForCausalLM)
