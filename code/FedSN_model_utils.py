import numpy as np

def create_server_mask(config, model_name, p1):
    if "roberta" in model_name:
        n_layer = config.num_hidden_layers
        n_embd = config.hidden_size
        
        # 不放回抽样
        return {
            'emb': np.random.choice([0, 1/(1-p1)], size=n_embd, p=[p1, 1-p1]),
            'selfattn': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1]),
            'selfout': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1]),
            'out': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1])
        }
        

    elif "gpt2" in model_name :
        n_layer = config.n_layer
        n_embd = config.n_embd
        return {
            'emb': np.random.choice([0, 1/(1-p1)], size=n_embd, p=[p1, 1-p1]),
            'attn': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1]),
            'mlp': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1]),
        }
        # return {
        #     'emb': np.random.choice([0, 1/1], size=n_embd, p=[p1, 1-p1]),
        #     'attn': np.random.choice([0, 1/1], size=(n_layer, n_embd), p=[p1, 1-p1]),
        #     'mlp': np.random.choice([0, 1/1], size=(n_layer, n_embd), p=[p1, 1-p1]),
        # }
    
    elif "llama" in model_name:
        n_layer = config.num_hidden_layers
        n_embd = config.max_position_embeddings
        return {
            'emb': np.random.choice([0, 1/(1-p1)], size=n_embd, p=[p1, 1-p1]),
            'self_attn': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1]),
            'mlp': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1]),
            'layer': np.random.choice([0, 1/(1-p1)], size=(n_layer, n_embd), p=[p1, 1-p1]),
        }


def client_mask(arr, p=0.1):
    total_elements = arr.size  # 数组的总元素个数
    num_to_zero = int(p * total_elements)  # 需要置为0的元素总数
    
    # 找到数组中非0元素的索引
    non_zero_indices = np.argwhere(arr != 0)  # 返回的是多维索引的坐标
    
    # 如果非0元素不足以达到 num_to_zero 数量，取非0元素的数量
    num_to_zero = min(num_to_zero, len(non_zero_indices))
    
    # 从非零元素中随机选取 num_to_zero 个进行置零
    selected_indices = np.random.choice(len(non_zero_indices), num_to_zero, replace=False)
    
    # 将选中的元素置为0
    for idx in selected_indices:
        arr[tuple(non_zero_indices[idx])] = 0  # 多维索引需要用 tuple
    
    # 对数组中所有的非零元素进行放大操作
    arr[arr != 0] *= 1 / (1 - p)
    
    return arr



def create_client_mask(config, model_name, server_mask, p2, mask_type):
    if "roberta" in model_name:
        n_layer = config.num_hidden_layers
        n_embd = config.hidden_size
    
    
        if mask_type =="xzb":
            
            return {
                'emb':client_mask(np.array(server_mask['emb']),p2), 
                'selfattn': client_mask(np.array(server_mask['selfattn']),p2),
                'selfout': client_mask(np.array(server_mask['selfout']),p2),
                'out': client_mask(np.array(server_mask['out']),p2),
            }
            
        else:
            return {
                'emb': np.array(server_mask['emb'])*np.random.choice([0, 1/(1-p2)], size=n_embd, p=[p2, 1-p2]),
                'selfattn': np.array(server_mask['selfattn'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
                'selfout': np.array(server_mask['selfout'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
                'out': np.array(server_mask['out'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
            }
    
        
        # 规范采样版本
        
        
    elif "gpt2" in model_name :
        n_layer = config.n_layer
        n_embd = config.n_embd
        
        if mask_type=="xzb":
            return {
                    'emb':client_mask(np.array(server_mask['emb']),p2), 
                    'attn': client_mask(np.array(server_mask['attn']),p2),
                    'mlp': client_mask(np.array(server_mask['mlp']),p2),
                }
            
        else:
            return {
                'emb': np.array(server_mask['emb'])*np.random.choice([0, 1/(1-p2)], size=n_embd, p=[p2, 1-p2]),
                'attn': np.array(server_mask['attn'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
                'mlp': np.array(server_mask['mlp'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
            }
            # return {
            #     'emb': np.array(server_mask['emb'])*np.random.choice([0, 1/1], size=n_embd, p=[p2, 1-p2]),
            #     'attn': np.array(server_mask['attn'])*np.random.choice([0, 1/1], size=(n_layer, n_embd), p=[p2, 1-p2]),
            #     'mlp': np.array(server_mask['mlp'])*np.random.choice([0, 1/1], size=(n_layer, n_embd), p=[p2, 1-p2]),
            # }
    
    elif "llama" in model_name:
        n_layer = config.num_hidden_layers
        n_embd = config.max_position_embeddings
        if mask_type=="xzb":
            return {
                    'emb': client_mask(np.array(server_mask['emb']),p2),    
                    'self_attn': client_mask(np.array(server_mask['self_attn']),p2),
                    'mlp': client_mask(np.array(server_mask['mlp']),p2),
                    'layer': client_mask(np.array(server_mask['layer']),p2),
                    
                }
            
        else:
            return {
                'emb': np.array(server_mask['emb'])*np.random.choice([0, 1/(1-p2)], size=n_embd, p=[p2, 1-p2]),
                'self_attn': np.array(server_mask['self_attn'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
                'mlp': np.array(server_mask['mlp'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
                'layer': np.array(server_mask['layer'])*np.random.choice([0, 1/(1-p2)], size=(n_layer, n_embd), p=[p2, 1-p2]),
            
            }


def update_averaged_state_dict(ava_state_dict, new_state_dict, k):
    for key in ava_state_dict:
        ava_state_dict[key] = (k * ava_state_dict[key] + new_state_dict[key]) / (k + 1)
    return ava_state_dict