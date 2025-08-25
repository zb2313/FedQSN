import datasets
from transformers import GPT2Tokenizer,AutoTokenizer
import jsonlines
from tqdm import tqdm
import os

raw_data = datasets.load_from_disk("/data/xzb/datasets/medqa")
tokenizer = AutoTokenizer.from_pretrained("/data/xzb/models/gpt2-medium")
eos_token_id = tokenizer.eos_token_id
NLG_path = "/data/xzb/feddtaset/medqa/gpt2/medqa"

def e2e_sample_to_sampleNLG(sample):
    temp = '\nOutput: '
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
    context_id = tokenizer.encode(context) + [eos_token_id] 
    completion_id = tokenizer.encode(completion) + [eos_token_id]
    sample_NLG = {
        "context":context_id ,
        "completion":completion_id
    }
    sample_NLG_formatted = {
        "context":context ,
        "completion":completion
    }
    return sample_NLG, sample_NLG_formatted



def dart_sample_to_sampleNLG(sample):
    context = ''
    for triple in sample['tripleset']:
        for element in triple:
            context = context + element + ' | '
        context  = context[:-1] + '| '
    context = context[:-4]
    completion = sample['target']
    context_id = tokenizer.encode(context) + [eos_token_id] 
    completion_id = tokenizer.encode(completion) + [eos_token_id]
    
    # print(context)
    # print(completion)
    # print(context_id)
    # print(completion_id)

    sample_NLG = {
        "completion":completion_id,
        "context":context_id
    }
    sample_NLG_formatted = {
        "context":context,
        "completion":completion
    }
    return sample_NLG, sample_NLG_formatted

def dia_sample_to_sampleNLG(sample):
    context = sample['topic'] + " | " + sample['dialogue']
    completion = sample['summary']
    context_id = tokenizer.encode(context) + [eos_token_id] 
    completion_id = tokenizer.encode(completion) + [eos_token_id]

    sample_NLG = {
        "completion":completion_id,
        "context":context_id
    }
    sample_NLG_formatted = {
        "context":context,
        "completion":completion
    }
    return sample_NLG, sample_NLG_formatted

def viggo_sample_to_sampleNLG(sample):
    context = sample['meaning_representation']
    completion = sample['target']
    context_id = tokenizer.encode(context) + [eos_token_id] 
    completion_id = tokenizer.encode(completion) + [eos_token_id]

    sample_NLG = {
        "completion":completion_id,
        "context":context_id
    }
    sample_NLG_formatted = {
        "context":context,
        "completion":completion
    }
    return sample_NLG, sample_NLG_formatted

def medqa_sample_to_sampleNLG(sample):
    context = sample['meaning_representation']
    completion = sample['target']
    context_id = tokenizer.encode(context) + [eos_token_id] 
    completion_id = tokenizer.encode(completion) + [eos_token_id]

    sample_NLG = {
        "completion":completion_id,
        "context":context_id
    }
    sample_NLG_formatted = {
        "context":context,
        "completion":completion
    }
    return sample_NLG, sample_NLG_formatted

def show_sen_len_hist(list, p):
    l = sorted(list)
    for i in range(p):
        print(i, int((i/p)*len(l)), l[int((i/p)*len(l))])
    print("max:" + str(max(list)))
    print("--------------------------------")

def make_NLG_dataset(s2s, dataset, split, gen_path, gen_name):
    # 确保目标目录存在
    os.makedirs(gen_path, exist_ok=True)
    
    len_con = []
    len_com = []
    len_tot = []
    with jsonlines.open(gen_path + f"/{gen_name}.jsonl", 'w') as w:
        with jsonlines.open(gen_path + f"/{gen_name}_formatted.jsonl", 'w') as w_form:
            for i, sample in enumerate(tqdm(dataset[split])):
                sample_NLG, sample_NLG_formatted = s2s(sample)
                w.write(sample_NLG)
                w_form.write(sample_NLG_formatted)
                len_con.append(len(sample_NLG["context"]))
                len_com.append(len(sample_NLG["completion"]))
                len_tot.append(len(sample_NLG["context"]) + len(sample_NLG["completion"]))
    show_sen_len_hist(len_con, 20)
    show_sen_len_hist(len_com, 20)
    show_sen_len_hist(len_tot, 20)

s2s = e2e_sample_to_sampleNLG

print(len(raw_data['train']))
print(len(raw_data['validation']))
print(len(raw_data['test']))
print(raw_data['train'][0])

make_NLG_dataset(s2s, raw_data, "train", NLG_path, "train")
make_NLG_dataset(s2s, raw_data, "validation", NLG_path, "valid")
make_NLG_dataset(s2s, raw_data, "test", NLG_path, "test")
