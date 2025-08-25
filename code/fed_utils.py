import os
import logging
from tqdm import tqdm
import torch
import json
import random

def set_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    final_path = f"{log_dir}/log"
    logger = logging.getLogger()
    logger.setLevel('INFO')
    control = logging.StreamHandler() 
    control.setLevel('INFO')
    fhlr = logging.FileHandler(final_path)
    logger.addHandler(fhlr)
    logger.addHandler(control)
    return logger






def generate_prepare(test_dataLoader, dataset_path, generation_dir):

    #重新整理测试集，也就是test.jsonl，测试集里一条query可能对应多条ref，这里做一个去重，减少测试量
    test_data_list = []
    query_set = set()
    for id, data in enumerate(tqdm(test_dataLoader)):
        query = str(data["query"].tolist())
        if not query in query_set:
            query_set.add(query)
            cuda_data = {k:v.to(torch.device("cuda")) for k,v in data.items()}
            cuda_data["id"] = id
            test_data_list.append(cuda_data)

    context_list = []
    context_pred_refs_dict = {}
    with open(dataset_path + "/test_formatted.jsonl", 'r', encoding='utf8') as reader:
        for line in reader:
            items = json.loads(line.strip())
            context = items['context']
            completion = items['completion']
            context_list.append(context)

            if not context in context_pred_refs_dict:
                context_pred_refs_dict[context] = {}
                context_pred_refs_dict[context]["refs"] = []
                context_pred_refs_dict[context]["pred"] = "[No answer]"
            new_ref = completion.split('<|endoftext|>')[0].split('\n\n')[0].strip()
            context_pred_refs_dict[context]["refs"].append(new_ref)
    os.makedirs(generation_dir, exist_ok=True)
    
    return test_data_list, context_pred_refs_dict, context_list




def client_choose(R, N, C):
    choosen_client_ids = []
    for r in range(R):
        client_ids = random.sample(range(N), C)
        choosen_client_ids.append(client_ids)
    return choosen_client_ids

def update_save_models(better, save_dir, name, state_dict, mask, best_mask):
    if better:
        torch.save(state_dict, save_dir + f"/best_{name}.pth")
        torch.save(mask, save_dir + f"/best_mask_{name}.pth")
        return mask
    else:
        return best_mask

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):

    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

def show_3D_tensor(t, dp, max_batch_size, max_sen_len, max_hidden):
    l = t[:max_batch_size, :max_sen_len, :max_hidden].tolist()
    return '\n'.join(['[\n\t' + '\n\t'.join(['[' + (', '.join([f'{element:.{dp}f}' for element in l1D])) + ']' for l1D in l2D]) + '\n]' for l2D in l])

def show_1D_tensor(t, dp, max_hidden):
    l = t[:max_hidden].tolist()
    return '[' + (', '.join([f'{element:.{dp}f}' for element in l])) + ']'



