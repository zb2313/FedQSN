from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.activations import ACT2FN
from torch.cuda.amp import autocast
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
import math




import fed_utils

class Fed_Embedding(torch.nn.Module):
  
    def __init__(self, num_embeddings: int, embedding_dim: int, fed_mask, logger) -> None:
        factory_kwargs = {'device': None, 'dtype': None}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._fed_mask = fed_mask
        self._logger = logger
        
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2
        self.scale_grad_by_freq = False

        self.weight = torch.nn.parameter.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                                requires_grad=True)
        self.reset_parameters()


        self.sparse = False

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state =  torch.nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        
        #进行联邦学习的mask
        fed_mask = torch.tensor(self._fed_mask, dtype = hidden_state.dtype, device = hidden_state.device)

        # self._logger.info(f'embeddings shape: {str(hidden_state.shape)}')
        # self._logger.info(f'embeddings : {fed_utils.show_3D_tensor(hidden_state, 4, 2, 2, 10)}')
        # self._logger.info(f'fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'fed_mask : {fed_utils.show_1D_tensor(fed_mask, 2, 10)}')

        fed_mask = fed_mask.expand_as(hidden_state)

        # self._logger.info(f'expanded fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'expanded fed_mask : {fed_utils.show_3D_tensor(fed_mask, 2, 2, 2, 10)}')

        hidden_state = hidden_state * fed_mask

        # self._logger.info(f'maksed embeddings shape: {str(hidden_state.shape)}')
        # self._logger.info(f'maksed embeddings : {fed_utils.show_3D_tensor(hidden_state, 4, 2, 2, 10)}')


        return hidden_state

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask



class Fed_Attention(nn.Module):
    def __init__(self, config, fed_mask, logger):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = False

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = False
        self.layer_idx = None
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn


        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self._fed_mask = fed_mask
        self._logger = logger

        self.pruned_heads = set()

    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        fed_mask = torch.tensor(self._fed_mask, dtype = attn_output.dtype, device = attn_output.device)

        # self._logger.info(f'attn_output shape: {str(attn_output.shape)}')
        # self._logger.info(f'attn_output : {fed_utils.show_3D_tensor(attn_output, 4, 2, 2, 10)}')
        # self._logger.info(f'fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'fed_mask : {fed_utils.show_1D_tensor(fed_mask, 2, 10)}')

        fed_mask = fed_mask.expand_as(attn_output)

        # self._logger.info(f'expanded fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'expanded fed_mask : {fed_utils.show_3D_tensor(fed_mask, 2, 2, 2, 10)}')

        attn_output = attn_output * fed_mask

        # self._logger.info(f'maksed attn_output shape: {str(attn_output.shape)}')
        # self._logger.info(f'maksed attn_output : {fed_utils.show_3D_tensor(attn_output, 4, 2, 2, 10)}')

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class Fed_MLP(nn.Module):
    def __init__(self, intermediate_size, config, fed_mask, logger):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
        self._fed_mask = fed_mask
        self._logger = logger

    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        fed_mask = torch.tensor(self._fed_mask, dtype = hidden_states.dtype, device = hidden_states.device)

        # self._logger.info(f'hidden_states shape: {str(hidden_states.shape)}')
        # self._logger.info(f'hidden_states : {fed_utils.show_3D_tensor(hidden_states, 4, 2, 2, 10)}')
        # self._logger.info(f'fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'fed_mask : {fed_utils.show_1D_tensor(fed_mask, 2, 10)}')

        fed_mask = fed_mask.expand_as(hidden_states)

        # self._logger.info(f'expanded fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'expanded fed_mask : {fed_utils.show_3D_tensor(fed_mask, 2, 2, 2, 10)}')

        hidden_states = hidden_states * fed_mask

        # self._logger.info(f'maksed hidden_states shape: {str(hidden_states.shape)}')
        # self._logger.info(f'maksed hidden_states : {fed_utils.show_3D_tensor(hidden_states, 4, 2, 2, 10)}')

        return hidden_states


class FedSN_GPTLM_Model(nn.Module):
    def __init__(self, model_name, config, fed_mask, logger):
        super().__init__()

        self._fed_mask = fed_mask
        self._config = config
        self._model_name = model_name
        self._logger = logger

        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size

        self._model = GPT2LMHeadModel(config)
        self._model.transformer.wte = Fed_Embedding(config.vocab_size, config.n_embd, fed_mask['emb'], logger)
        for l in range(self._config.n_layer):
            self._model.transformer.h[l].attn = Fed_Attention(config, fed_mask['attn'][l], logger)
            self._model.transformer.h[l].mlp = Fed_MLP(inner_dim, config, fed_mask['mlp'][l], logger)


    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

        self._model.transformer.wte._set_fed_mask(fed_mask['emb'])
        for l in range(self._config.n_layer):
            self._model.transformer.h[l].attn._set_fed_mask(fed_mask['attn'][l])
            self._model.transformer.h[l].mlp._set_fed_mask(fed_mask['mlp'][l])



"""
GPT2Config {
  "_name_or_path": "/disks/sda/zjh/20231026_VBGMMPFL/model/gpt2_medium_lm/model",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 1024,
  "n_head": 16,
  "n_inner": null,
  "n_layer": 24,
  "n_positions": 1024,
  "n_special": 0,
  "predict_special_tokens": true,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "torch_dtype": "float32",
  "transformers_version": "4.35.0",
  "use_cache": true,
  "vocab_size": 50257
}

"""



