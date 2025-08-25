import numpy as np
import torch
from scipy.stats import norm
import math
import copy

class Quant_Manager():
    def __init__(self, logger, alg, model_name, config, bit_width = None, block_size = None):
        self._logger = logger
        self._alg = alg
        self._model_name = model_name
        self._config = config
        
        self._model_para_meta_dict = {}
        self._mask_meta_dict = {}
        # self._update_method = update_method

        if "roberta" in self._model_name:
            self._num_layers = self._config.num_hidden_layers
            hidden_size = self._config.hidden_size
            num_attention_heads = self._config.num_attention_heads
            attention_head_size = int(self._config.hidden_size / self._config.num_attention_heads)
            all_head_size = num_attention_heads * attention_head_size
            intermediate_size = self._config.intermediate_size
            for layer_id in range(self._num_layers):
                self._model_para_meta_dict[f"roberta.encoder.layer.{str(layer_id)}.attention.self.query.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": all_head_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_query_mask"
                }
                self._model_para_meta_dict[f"roberta.encoder.layer.{str(layer_id)}.attention.self.key.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": all_head_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_key_mask"
                }
                self._model_para_meta_dict[f"roberta.encoder.layer.{str(layer_id)}.attention.self.value.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": all_head_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_value_mask"
                }
                self._model_para_meta_dict[f"roberta.encoder.layer.{str(layer_id)}.attention.output.dense.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": hidden_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_output_mask"
                }
                self._model_para_meta_dict[f"roberta.encoder.layer.{str(layer_id)}.intermediate.dense.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": intermediate_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "intermediate_mask"
                }
                self._model_para_meta_dict[f"roberta.encoder.layer.{str(layer_id)}.output.dense.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": intermediate_size,
                    "output_size": hidden_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "output_mask"
                }
            self._mask_meta_dict["attention_query_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": all_head_size
            }
            self._mask_meta_dict["attention_key_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": all_head_size
            }
            self._mask_meta_dict["attention_value_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": all_head_size
            }
            self._mask_meta_dict["attention_output_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": hidden_size
            }
            self._mask_meta_dict["dense_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": intermediate_size
            }
            self._mask_meta_dict["output_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": intermediate_size,
                "output_size": hidden_size
            }
        elif "gpt" in self._model_name:
            self._num_layers = self._config.n_layer
            hidden_size = self._config.hidden_size
            inner_dim = self._config.n_inner if self._config.n_inner is not None else 4 * hidden_size
            for layer_id in range(self._num_layers):
                self._model_para_meta_dict[f"transformer.h.{str(layer_id)}.attn.c_attn.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": 3 * hidden_size,
                    "para_type": "Fed_Conv1D",
                    "mask_key_name": "attention_attn_mask"
                }
                self._model_para_meta_dict[f"transformer.h.{str(layer_id)}.attn.c_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": hidden_size,
                    "para_type": "Fed_Conv1D",
                    "mask_key_name": "attention_proj_mask"
                }
                self._model_para_meta_dict[f"transformer.h.{str(layer_id)}.mlp.c_fc.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": inner_dim,
                    "para_type": "Fed_Conv1D",
                    "mask_key_name": "mlp_fc_mask"
                }
                self._model_para_meta_dict[f"transformer.h.{str(layer_id)}.mlp.c_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": inner_dim,
                    "output_size": hidden_size,
                    "para_type": "Fed_Conv1D",
                    "mask_key_name": "mlp_proj_mask"
                }
            self._mask_meta_dict["attention_attn_mask"] = {
                "para_type": "Fed_Conv1D",
                "input_size": hidden_size,
                "output_size": 3 * hidden_size
            }
            self._mask_meta_dict["attention_proj_mask"] = {
                "para_type": "Fed_Conv1D",
                "input_size": hidden_size,
                "output_size": hidden_size
            }
            self._mask_meta_dict["mlp_fc_mask"] = {
                "para_type": "Fed_Conv1D",
                "input_size": hidden_size,
                "output_size": inner_dim
            }
            self._mask_meta_dict["mlp_proj_mask"] = {
                "para_type": "Fed_Conv1D",
                "input_size": inner_dim,
                "output_size": hidden_size
            }
        elif "llama" in self._model_name:
            self._num_layers = self._config.num_hidden_layers
            hidden_size = self._config.hidden_size
            num_heads = self._config.num_attention_heads
            head_dim = hidden_size // num_heads
            num_key_value_heads = self._config.num_key_value_heads
            query_hidden_size = num_heads * head_dim
            key_value_hidden_size = num_key_value_heads * head_dim
            intermediate_size = self._config.intermediate_size

            for layer_id in range(self._num_layers):
                self._model_para_meta_dict[f"model.layers.{str(layer_id)}.self_attn.q_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": query_hidden_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_query_mask"
                }
                self._model_para_meta_dict[f"model.layers.{str(layer_id)}.self_attn.k_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": key_value_hidden_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_key_mask"
                }
                self._model_para_meta_dict[f"model.layers.{str(layer_id)}.self_attn.v_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": key_value_hidden_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_value_mask"
                }
                self._model_para_meta_dict[f"model.layers.{str(layer_id)}.self_attn.o_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": hidden_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "attention_output_mask"
                }
                self._model_para_meta_dict[f"model.layers.{str(layer_id)}.mlp.gate_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": intermediate_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "mlp_gate_mask"
                }
                self._model_para_meta_dict[f"model.layers.{str(layer_id)}.mlp.up_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": hidden_size,
                    "output_size": intermediate_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "mlp_up_mask"
                }
                self._model_para_meta_dict[f"model.layers.{str(layer_id)}.mlp.down_proj.weight"] = {
                    "layer_id" : layer_id,
                    "input_size": intermediate_size,
                    "output_size": hidden_size,
                    "para_type": "Fed_Linear",
                    "mask_key_name": "mlp_down_mask"
                }
            self._mask_meta_dict["attention_query_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": query_hidden_size
            }
            self._mask_meta_dict["attention_key_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": key_value_hidden_size
            }
            self._mask_meta_dict["attention_value_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": key_value_hidden_size
            }
            self._mask_meta_dict["attention_output_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": hidden_size
            }
            self._mask_meta_dict["mlp_gate_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": intermediate_size
            }
            self._mask_meta_dict["mlp_up_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": hidden_size,
                "output_size": intermediate_size
            }
            self._mask_meta_dict["mlp_down_mask"] = {
                "para_type": "Fed_Linear",
                "input_size": intermediate_size,
                "output_size": hidden_size
            }


        if alg in ["FedQSN","FedSN"]:
            assert bit_width is not None
            assert block_size is not None
            self._bit_width = bit_width
            self._block_size = block_size
            # self._server_preserve_mask = self._init_mask(p1)
            # self._client_access_masks = []
            # for client_id in range(N):
            #     mask = self._init_mask(1 - p2)
            #     client_access_mask = {}
            #     for mask_key_name in mask:
            #         client_access_mask[mask_key_name] = (1 - self._server_preserve_mask[mask_key_name]) * mask[mask_key_name]
            #     self._client_access_masks.append(client_access_mask)
            if alg in ["FedQSN"]:
                self._create_normal_map()
            
            # if self._update_method ==2:
            #     self._init_delta = {}
            #     for key in model_state_dict_1:
            #         self._init_delta[key] = model_state_dict_1[key] - model_state_dict_2[key]
            
        # elif alg in ["pFedSN"]:
        #     self._server_preserve_mask = self._init_mask(p1)
        #     self._client_access_mask = {}
        #     for mask_key_name in self._server_preserve_mask:
        #         self._client_access_mask[mask_key_name] = 1 - self._server_preserve_mask[mask_key_name]
        #     self._client_open_masks = []
        #     self._client_preserve_masks = []
        #     for client_id in range(N):
        #         mask = self._init_mask(1 - p2)
        #         client_open_mask = {}
        #         client_preserve_mask = {}
        #         for mask_key_name in mask:
        #             client_open_mask[mask_key_name] = self._client_access_mask[mask_key_name] * mask[mask_key_name]
        #             client_preserve_mask[mask_key_name] = self._client_access_mask[mask_key_name] * (1 - mask[mask_key_name])
        #         self._client_open_masks.append(client_open_mask)
        #         self._client_preserve_masks.append(client_preserve_mask)


    def _init_mask(self, p):
        mask = {}
        for mask_key_name in self._mask_meta_dict:
            if self._mask_meta_dict[mask_key_name]["para_type"] == "Fed_Linear":
                r = self._mask_meta_dict[mask_key_name]["output_size"]
                l = self._mask_meta_dict[mask_key_name]["input_size"]
            elif self._mask_meta_dict[mask_key_name]["para_type"] == "Fed_Conv1D":
                r = self._mask_meta_dict[mask_key_name]["input_size"]
                l = self._mask_meta_dict[mask_key_name]["output_size"]
            # 先生成 float 类型的 Bernoulli 张量，然后转换为 int8
            mask[mask_key_name] = torch.bernoulli(torch.full((self._num_layers, r, l), p)).to(torch.int8)
            
        return mask
    def _get_server_preserve_mask(self):
        return self._server_preserve_mask
    def _get_client_access_mask(self, client_id):
        if self._alg in ["FedQSN","FedSN"]:
            return self._client_access_masks[client_id]
        elif self._alg in ["pFedSN"]:
            return self._client_access_mask
    def _get_client_open_mask(self, client_id):
        assert self._alg in ["pFedSN"]
        return self._client_open_masks[client_id]
    def _get_client_preserve_mask(self, client_id):
        assert self._alg in ["pFedSN"]
        return self._client_preserve_masks[client_id]
    

    def _merge_mask_into_model(self, model_state_dict, mask, a):
        for model_para_key_name in self._model_para_meta_dict:
            mask_key_name = self._model_para_meta_dict[model_para_key_name]["mask_key_name"]
            layer_id = self._model_para_meta_dict[model_para_key_name]["layer_id"]
            model_state_dict[model_para_key_name] = model_state_dict[model_para_key_name] * mask[mask_key_name][layer_id].to("cuda:0") * a



    def _reference_para_with_mask(self, para1, para2, mask):
        reference_pata = para1 * mask + para2 + (1 - mask)
        return reference_pata
    def _reference_model_with_mask(self, model_state_dict_1, model_state_dict_2, mask):
        for model_para_key_name in self._model_para_meta_dict:
            mask_key_name = self._model_para_meta_dict[model_para_key_name]["mask_key_name"]
            layer_id = self._model_para_meta_dict[model_para_key_name]["layer_id"]
            model_state_dict_1[model_para_key_name] = self._reference_para_with_mask(model_state_dict_1[model_para_key_name], model_state_dict_2[model_para_key_name], mask[mask_key_name][layer_id])



    def _compute_delta(self, model_state_dict_1, model_state_dict_2):
        delta = {}
        for key in model_state_dict_1:
            delta[key] = model_state_dict_1[key] - model_state_dict_2[key]
        return delta
    
    def _add_delta(self, model_state_dict_1, model_state_dict_2):
        delta = {}
        for key in model_state_dict_1:
            delta[key] = model_state_dict_1[key] + model_state_dict_2[key]
        return delta
    
    def _maintain_delta(self, delta, new_delta, i):
        print(i)
        if i == 0:
            maintained_delta = copy.deepcopy(new_delta)
        else:
            maintained_delta = {}
            for key in delta:
                maintained_delta[key] = (i * delta[key] + new_delta[key]) / (i + 1)
        return maintained_delta
    def _load_delta(self, model_state_dict, delta, delta_lr):
        
        for key in model_state_dict:
            model_state_dict[key] = model_state_dict[key] + delta[key] * delta_lr
        

    def _create_normal_map(self):

        self._offset = 0.9677083
        self._positive_num = 2**(self._bit_width-1)
        self._negetive_num = 2**(self._bit_width-1) 
        v1 = norm.ppf(torch.linspace(self._offset, 0.5, self._positive_num + 1)[:-1]).tolist() 
        v2 = [0] 
        v3 = (-norm.ppf(torch.linspace(self._offset, 0.5, self._negetive_num + 1)[:-1])).tolist() 
        v = v1 + v2 + v3
        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        self._Q = values.tolist()
        self._T = []
        for i in range(len(self._Q) - 1):
            self._T.append((self._Q[i] + self._Q[i + 1]) / 2)


    def _quantize_tensor(self, tensor):
        shape = tensor.shape  
        block_tensor = tensor.view(-1, self._block_size)
        max_c = torch.max(abs(block_tensor), dim=1).values.view(-1, 1)
        norm_block_tensor = block_tensor / max_c

        quan_norm_block_tensor = torch.zeros_like(block_tensor)
        quan_norm_block_tensor = quan_norm_block_tensor + self._Q[0] * (norm_block_tensor <= self._T[0])
        for i in range(1, len(self._Q) - 1):
            quan_norm_block_tensor = quan_norm_block_tensor + self._Q[i] * (norm_block_tensor <= self._T[i]) * (norm_block_tensor > self._T[i-1])
        quan_norm_block_tensor = quan_norm_block_tensor + self._Q[-1] * (norm_block_tensor > self._T[-1])
        quan_block_tensor = quan_norm_block_tensor * max_c
        quan_tensor = quan_block_tensor.view(-1, shape[-1])
        return quan_tensor

    
    def _quantize_model(self, model_state_dict):
        for key in model_state_dict:
            if key in self._model_para_meta_dict:
                model_state_dict[key]  = self._quantize_tensor(model_state_dict[key])
        return model_state_dict