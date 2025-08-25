import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type = str)
parser.add_argument("--dataset_name", type = str)
parser.add_argument("--device", type = int)
args = parser.parse_args()



run_name = str(args.run_name)
dataset_name = str(args.dataset_name)
device = int(args.device)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

import datasets
from transformers import AutoTokenizer, BertModel
import torch
import jsonlines
from tqdm import tqdm
from datetime import datetime

from utils import set_log

time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":", "").replace(" ", "").replace("-", "")
meta_dir = os.path.join('/data/xzb/FedQSN', '[meta]', run_name, time)
log_dir = meta_dir.replace("[meta]", "logs")
logger = set_log(log_dir)

dataset_path = f'/data/xzb/datasets/{dataset_name}'
dataset_emb_path = f'/data/xzb/datasets/llama3.1/{dataset_name}.jsonl'

# gpt2_tokenizer_path = '/data/xzb/models/gpt2-large'
# llama_tokenizer_path = "/data/xzb/models/llama2-7B-Hf"
# bert_model_path = '/data/xzb/models/bert'
# bert_tokenizer_path = '/data/xzb/models/bert'
tokenizer_path = '/data/xzb/models/llama3-8B-Instruct'


data = datasets.load_from_disk(dataset_path)

# gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_tokenizer_path)
# llama_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_path)
# bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)
# bert_model = BertModel.from_pretrained(bert_model_path).to(torch.device('cuda:0'))
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
eos_token_id = tokenizer.eos_token_id

def sample_fix(sample, dataset_name):
    temp = '\nOutput: '
    if "e2e" in dataset_name:
        relation_list = sample['meaning_representation'].split(',')
        key_value_list = []
        for relation in relation_list:
            relation = relation.strip()
            index = relation.find('[')
            key = relation[:index]
            if key == "eatType":
                key = "Type"
            if key == "priceRange":
                key = "price"
            value = relation[index+1:-1]
            key_value_list.append(f'{key}: {value}')
        context = ' | '.join(key_value_list) + temp
        completion = sample['human_reference']
    elif "dart" in dataset_name:
        context = ', '.join([f"({' | '.join(triple)})" for triple in sample['tripleset']]) + temp
        completion = sample['target']
    elif "dialogsum" in dataset_name:
        context = sample['topic'] + " | " + sample['dialogue'] + temp
        completion = sample['summary']        
    elif "viggo" in dataset_name:
        context = sample['meaning_representation'] + temp
        completion = sample['target']   
    elif "greek" in dataset_name:
        context = sample['Message'] + temp
        completion = sample['gtrans_el']     
    elif "tibetan" in dataset_name:
        context = sample['tibetan']+"|" +sample['phonetic']+ temp
        completion = sample['english']

    return {
        "context": context ,
        "completion": completion
    }



# bert_model.to(torch.device(f'cuda:0'))
with jsonlines.open(dataset_emb_path, 'w') as w:
    samples = []
    for split in ['train', 'validation', 'test']:
        for sample in data[split]:
            samples.append(sample)
    for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
        fixed_sample = sample_fix(sample, dataset_name)
        # fixed_sample[f'context_id_gpt2'] = gpt2_tokenizer.encode(fixed_sample['context'], add_special_tokens = False)
        # fixed_sample[f'completion_id_gpt2'] = gpt2_tokenizer.encode(fixed_sample['completion'], add_special_tokens = False)
        # fixed_sample[f'context_id_llama'] = llama_tokenizer.encode(fixed_sample['context'], add_special_tokens = False)
        # fixed_sample[f'completion_id_llama'] = llama_tokenizer.encode(fixed_sample['completion'], add_special_tokens = False)
        fixed_sample[f'context_id'] = tokenizer.encode(fixed_sample['context'], add_special_tokens = False)
        fixed_sample[f'completion_id'] = tokenizer.encode(fixed_sample['completion'], add_special_tokens = False)
        # context_id_bert = bert_tokenizer.encode(fixed_sample['context'], return_tensors = 'pt').to(torch.device(f'cuda:0'))[:, :512]
        # bert_output = bert_model(context_id_bert)
        # fixed_sample[f'context_emb_bert'] = bert_output[0].squeeze(0).mean(dim=0).tolist()
        w.write(fixed_sample)

