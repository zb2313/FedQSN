from transformers import RobertaForSequenceClassification
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
import math

# RobertaForCausalLM(
#   (roberta): RobertaModel(
#     (embeddings): RobertaEmbeddings(
#       (word_embeddings): Embedding(50265, 768, padding_idx=1)
#       (position_embeddings): Embedding(514, 768, padding_idx=1)
#       (token_type_embeddings): Embedding(1, 768)
#       (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#       (dropout): Dropout(p=0.1, inplace=False)
#     )
#     (encoder): RobertaEncoder(
#       (layer): ModuleList(
#         (0-11): 12 x RobertaLayer(
#           (attention): RobertaAttention(
#             (self): RobertaSelfAttention(
#               (query): Linear(in_features=768, out_features=768, bias=True)
#               (key): Linear(in_features=768, out_features=768, bias=True)
#               (value): Linear(in_features=768, out_features=768, bias=True)
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#             (output): RobertaSelfOutput(
#               (dense): Linear(in_features=768, out_features=768, bias=True)
#               (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#           )
# ...
#     (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#     (decoder): Linear(in_features=768, out_features=50265, bias=True)
#   )
# )



import utils


class Fed_RobertaEmbeddings(nn.Module): 
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config, fed_mask, logger):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self._fed_mask = fed_mask
        self._logger = logger
    
    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = utils.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings) #(64,160,768)

        #进行联邦学习的mask
        fed_mask = torch.tensor(self._fed_mask, dtype = embeddings.dtype, device = embeddings.device)
        # self._logger.info(f'embeddings shape: {str(embeddings.shape)}')
        # self._logger.info(f'embeddings : {utils.show_3D_tensor(embeddings, 4, 2, 2, 10)}')
        # self._logger.info(f'fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'fed_mask : {utils.show_1D_tensor(fed_mask, 2, 10)}')
        fed_mask = fed_mask.expand_as(embeddings)
        # self._logger.info(f'expanded fed_mask shape: {str(fed_mask.shape)}')
        # self._logger.info(f'expanded fed_mask : {utils.show_3D_tensor(fed_mask, 2, 2, 2, 10)}')
        embeddings = embeddings * fed_mask
        # self._logger.info(f'maksed embeddings shape: {str(embeddings.shape)}')
        # self._logger.info(f'maksed embeddings : {utils.show_3D_tensor(embeddings, 4, 2, 2, 10)}')

        embeddings = self.dropout(embeddings)




        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

class Fed_RobertaSelfAttention(nn.Module):
    def __init__(self, config, fed_mask, logger, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        self._fed_mask = fed_mask
        self._logger = logger

    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)



        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class Fed_RobertaSelfOutput(nn.Module):
    def __init__(self, config, fed_mask, logger):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._fed_mask = fed_mask
        self._logger = logger

    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        fed_mask = torch.tensor(self._fed_mask, dtype = hidden_states.dtype, device = hidden_states.device)
        fed_mask = fed_mask.expand_as(hidden_states)
        hidden_states = hidden_states * fed_mask
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states




# Copied from transformers.models.bert.modeling_bert.BertOutput
class Fed_RobertaOutput(nn.Module):
    def __init__(self, config, fed_mask, logger):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._fed_mask = fed_mask
        self._logger = logger

    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        fed_mask = torch.tensor(self._fed_mask, dtype = hidden_states.dtype, device = hidden_states.device)
        fed_mask = fed_mask.expand_as(hidden_states)
        hidden_states = hidden_states * fed_mask
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states




class FedSN_RoBERTa_Model(nn.Module):
    def __init__(self, model_name, config, fed_mask, logger):
        super().__init__()

        self._model = RobertaForSequenceClassification(config)
        self._model.roberta.embeddings = Fed_RobertaEmbeddings(config, fed_mask['emb'], logger)
        for l in range(config.num_hidden_layers):
            self._model.roberta.encoder.layer[l].attention.self = Fed_RobertaSelfAttention(config, fed_mask['selfattn'][l], logger)
            self._model.roberta.encoder.layer[l].attention.output = Fed_RobertaSelfOutput(config, fed_mask['selfout'][l], logger)
            self._model.roberta.encoder.layer[l].output = Fed_RobertaOutput(config, fed_mask['out'][l], logger)

        self._fed_mask = fed_mask
        self._config = config
        self._model_name = model_name
        self._logger = logger

    def _set_fed_mask(self, fed_mask):
        self._fed_mask = fed_mask

        self._model.roberta.embeddings._set_fed_mask(fed_mask['emb'])
        for l in range(self._config.num_hidden_layers):
            self._model.roberta.encoder.layer[l].attention.self._set_fed_mask(fed_mask['selfattn'][l])
            self._model.roberta.encoder.layer[l].attention.output._set_fed_mask(fed_mask['selfout'][l])
            self._model.roberta.encoder.layer[l].output._set_fed_mask(fed_mask['out'][l])





