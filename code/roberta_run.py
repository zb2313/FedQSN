import argparse
# hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type = str)
parser.add_argument("--model_name", type = str)
parser.add_argument("--dataset_name", type = str)
parser.add_argument("--alg", type = str)
parser.add_argument("--p1", type = float)
parser.add_argument("--p2", type = float)
parser.add_argument("--bit_width", type = int)
parser.add_argument("--block_size", type = int)
parser.add_argument("--N", type = int)
parser.add_argument("--C", type = int)
parser.add_argument("--R", type = int)
parser.add_argument("--local_epochs", type = int)
parser.add_argument("--lr", type = float)
parser.add_argument("--batch_size", type = int)
parser.add_argument("--gradient_accumulation_steps", type = int)
parser.add_argument("--seed", type = int)
parser.add_argument("--device", type = int)
parser.add_argument("--mask_type",type = str)

args = parser.parse_args()

run_name = str(args.run_name)
model_name = str(args.model_name)
dataset_name = str(args.dataset_name)
alg = str(args.alg)
p1 = float(args.p1)
p2 = float(args.p2)
bit_width = int(args.bit_width)
block_size = int(args.block_size)
N = int(args.N)
C = int(args.C)
R = int(args.R)
local_epochs = int(args.local_epochs)
lr = float(args.lr)
batch_size = int(args.batch_size)
gradient_accumulation_steps = int(args.gradient_accumulation_steps)
seed = int(args.seed)
device = int(args.device)

mask_type = str(args.mask_type)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

import torch
import copy
from datetime import datetime
from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaConfig, AutoModelForCausalLM
from tqdm import tqdm
import torch.nn as nn


import fed_utils
from data_utils import CLS_Dataset_Manager
from FedSN_roberta_model import FedSN_RoBERTa_Model
from FedSN_model_utils import create_server_mask, create_client_mask, update_averaged_state_dict
from trainer_utils import CLS_Trainer
from result_manager import ResultManager
from quant_utils import Quant_Manager

if model_name == "roberta_base":
    model_path = "/data/xzb/models/fedmodel/roberta_base/cls_"
    tokenizer_path = "/data/xzb/models/fedmodel/roberta_base/tokenizer"
    
if model_name == "roberta_large":
    model_path = "/data/xzb/models/fedmodel/roberta_large/cls_"
    tokenizer_path = "/data/xzb/models/fedmodel/roberta_large/tokenizer"

if dataset_name == "rte":
    dataset_path = "/data/xzb/datasets/feddataset/glue_CLS/rte_CLS"
    max_length = 160
    M = 2
    p_train_valid = 10
    weight_decay = 0.01
elif dataset_name == "sst2":
    dataset_path = "/data/xzb/datasets/feddataset/glue_CLS/sst2_CLS"
    max_length = 50
    M = 2
    p_train_valid = 100
    weight_decay = 0.01
elif dataset_name == "mrpc":
    dataset_path = "/data/xzb/datasets/feddataset/glue_CLS/mrpc_CLS"
    max_length = 80
    M = 2
    p_train_valid = 10
    weight_decay = 0.01
elif dataset_name == "qqp":
    dataset_path = "/data/xzb/datasets/feddataset/glue_CLS/qqp_CLS"
    max_length = 60
    M = 2
    p_train_valid = 500
    weight_decay = 0.01
elif dataset_name == "qnli":
    dataset_path = "/data/xzb/datasets/feddataset/glue_CLS/qnli_CLS"
    max_length = 90
    M = 2
    p_train_valid = 100
    weight_decay = 0.01

max_train_step = -1
max_eval_step = -1
max_test_step = -1


time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":", "").replace(" ", "").replace("-", "")
meta_dir = f"/data/xzb/FedQSN/[meta]/{run_name}/{model_name}_{dataset_name}_p1({p1})p2({p2})_bitwidth_{bit_width}_blocksize_{block_size}_R{R}_N{N}C{C}_localepochs{local_epochs}_time_{time}"
save_dir = meta_dir.replace("[meta]", "save_models")
os.makedirs(save_dir, exist_ok=True)
log_dir = meta_dir.replace("[meta]", "logs")
logger = fed_utils.set_log(log_dir)
logger.info(f"Args:\n \
              \tmeta:\n \
              \t\trun_name: {run_name}\n\
              \t\tmodel_name: {model_name}\n\
              \t\ttime: {time}\n\
              \t\tseed: {seed}\n\
              \tdataset:\n\
              \t\tN: {str(N)}\n\
              \t\tC: {str(C)}\n\
              \t\tM: {str(M)}\n\
              \t\tdataset_name: {str(dataset_name)}\n\
              \t\tmax_length: {str(max_length)}\n\
              \tserver:\n\
              \t\tp1: {str(p1)}\n\
              \t\tp2: {str(p2)}\n\
              \t\tbit_width: {str(bit_width)} \n\
              \t\tblock_size: {str(block_size)} \n\
              \t\tmask_type: {str(mask_type)}\n\
              \ttrain:\n\
              \t\tR: {str(R)}\n\
              \t\tlocal_epochs: {str(local_epochs)}\n\
              \t\tlr: {str(lr)}\n\
              \t\tbatch_size: {str(batch_size)}\n\
              \t\tgradient_accumulation_steps: {str(gradient_accumulation_steps)}\n\
              \t\tweight_decay: {str(weight_decay)}\n\
              \tearly_stop:\n\
              \t\tmax_train_step: {str(max_train_step)}\n\
              \t\tmax_eval_step: {str(max_eval_step)}\n\
              \t\tmax_test_step: {str(max_test_step)}")

base_model = RobertaForSequenceClassification.from_pretrained(
    model_path + str(M),
    num_labels=M,
    torch_dtype = torch.float
    # model_path
)
# base_model = AutoModelForCausalLM.from_pretrained(model_path)
robertaconfig = base_model.config
base_model_state_dict = base_model.state_dict().copy()
model_state_dict = {k: v.to(torch.device("cuda:0")) for k, v in base_model_state_dict.items()}

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)

dataset_manager = CLS_Dataset_Manager(dataset_path, max_length, batch_size, seed, N, p_train_valid)
result_manager = ResultManager(N, R, logger, dataset_name,"FedSN")

loss_fct = nn.CrossEntropyLoss()
trainer = CLS_Trainer(
    dataset_name,
    lr, local_epochs, gradient_accumulation_steps, 
    weight_decay, loss_fct, 
    max_train_step, max_eval_step, max_test_step,
)

choosen_client_ids = fed_utils.client_choose(R, N, C)


best_client_before_mask = None
best_client_after_mask = None

model_mask = create_server_mask(robertaconfig, model_name, 0) #不打mask
fed_model = FedSN_RoBERTa_Model(model_name, robertaconfig, model_mask, logger).to(torch.device("cuda:0"))



fed_model._model.load_state_dict(model_state_dict, strict=True)
model_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
better = result_manager._write_model_result(model_result, 0)
fed_utils.update_save_models(better, save_dir, "model", model_state_dict, None, None)



server_mask = create_server_mask(robertaconfig, model_name, p1) #p1 mask
fed_model._set_fed_mask(server_mask)
server_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
better = result_manager._write_server_result(server_result, 0)
fed_utils.update_save_models(better, save_dir, "server", model_state_dict, None, None)
quant_manager = Quant_Manager(logger, "FedQSN", model_name, robertaconfig, bit_width, block_size)

result_manager.print_result(-1)

for r in range(R):
    ava_state_dict = None
    for i in range(C):
    
        client_id = choosen_client_ids[r][i]
        
        ## 量化
        quant_state_dict = quant_manager._quantize_model(model_state_dict)
        # print(quant_state_dict.keys())
        fed_model._model.load_state_dict(quant_state_dict, strict=True)
        # fed_model._model.load_state_dict(model_state_dict, strict=True)
        client_mask = create_client_mask(robertaconfig, model_name, server_mask, p2, mask_type)
        fed_model._set_fed_mask(client_mask)
        
        result_before = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
        better = result_manager._write_clients_before_result(result_before, r, client_id)
        best_client_before_mask = fed_utils.update_save_models(better, save_dir, "client_before", model_state_dict, client_mask, best_client_before_mask)

        trainer._train(fed_model, dataset_manager._get_train_loader(client_id))
        client_state_dict = fed_model._model.state_dict().copy()

        result_after = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
        better = result_manager._write_clients_after_result(result_after, r, client_id)
        best_client_after_mask = fed_utils.update_save_models(better, save_dir, "client_after", client_state_dict, client_mask, best_client_after_mask)

        if i == 0:
            ava_state_dict = client_state_dict
        else:
            ava_state_dict = update_averaged_state_dict(ava_state_dict, client_state_dict, i)

    fed_model._model.load_state_dict(ava_state_dict, strict=True)
    fed_model._set_fed_mask(model_mask)
    model_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
    better = result_manager._write_model_result(model_result, r+1)
    fed_utils.update_save_models(better, save_dir, "model", ava_state_dict, None, None)
    
    fed_model._set_fed_mask(server_mask)
    server_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
    better = result_manager._write_server_result(server_result, r+1)
    fed_utils.update_save_models(better, save_dir, "server", ava_state_dict, None, None)

    result_manager.print_result(r)

    model_state_dict = ava_state_dict

del model_state_dict
del ava_state_dict

logger.info(result_manager.get_best_round())

best_model_state_dict = torch.load(save_dir + '/best_model.pth')
fed_model._model.load_state_dict(best_model_state_dict, strict=True)
fed_model._set_fed_mask(model_mask)
fed_model._model = fed_model._model.to('cuda:0')
model_result = trainer._evaluate(fed_model, dataset_manager._get_test_loader(), "test")
logger.info(f"best model: {str(model_result)}")
del best_model_state_dict

best_server_state_dict = torch.load(save_dir + '/best_server.pth')
fed_model._model.load_state_dict(best_server_state_dict, strict=True)
fed_model._set_fed_mask(server_mask)
fed_model._model = fed_model._model.to('cuda:0')
server_result = trainer._evaluate(fed_model, dataset_manager._get_test_loader(), "test")
logger.info(f"best server: {str(server_result)}")
del best_server_state_dict

best_client_before_state_dict = torch.load(save_dir + '/best_client_before.pth')
fed_model._model.load_state_dict(best_client_before_state_dict, strict=True)
fed_model._set_fed_mask(best_client_before_mask)
fed_model._model = fed_model._model.to('cuda:0')
client_before_result = trainer._evaluate(fed_model, dataset_manager._get_test_loader(), "test")
logger.info(f"best client_before: {str(client_before_result)}")
del best_client_before_state_dict

best_client_after_state_dict = torch.load(save_dir + '/best_client_after.pth')
fed_model._model.load_state_dict(best_client_after_state_dict, strict=True)
fed_model._set_fed_mask(best_client_after_mask)
fed_model._model = fed_model._model.to('cuda:0')
client_after_result = trainer._evaluate(fed_model, dataset_manager._get_test_loader(), "test")
logger.info(f"best client_after: {str(client_after_result)}")
del best_client_after_state_dict


# /home/zhujh/lzwenv/bin/python3 /home/zhujh/20240702_FedSN/code/roberta_run.py


"""
print(base_model)

print("\n\n【【【【【【【for n,p in base_model.named_parameters():】】】】】】")
for n,p in base_model.named_parameters():
    if ("dense.bias"in n)or("LayerNorm" in n):
        print(n, p.shape, p.requires_grad, p.tolist()[:3])
    else:
        print(n, p.shape, p.requires_grad)
raw_state_dict = base_model.state_dict().copy()
print("\n\n【【【【【【【for key in raw_state_dict:】】】】】】")
for key in raw_state_dict:
    print(key)

print("\n\n【���【eval】】】【【【【【【【for n,p in base_model.named_parameters():】】】】】】")
base_model.eval()
print(base_model)
for n,p in base_model.named_parameters():
    if ("dense.bias"in n)or("LayerNorm" in n):
        print(n, p.shape, p.requires_grad, p.tolist()[:3])
    else:
        print(n, p.shape, p.requires_grad)

print("\n\n【【【eval】】】【【【【【【【for key in raw_state_dict:】】】】】】")
raw_state_dict = base_model.state_dict().copy()
for key in raw_state_dict:
    print(key)

FedModel(
  (_model): RobertaForSequenceClassification(
    (roberta): RobertaModel(
      (embeddings): Fed_RobertaEmbeddings(
        (word_embeddings): Embedding(50265, 768, padding_idx=1)
        (position_embeddings): Embedding(512, 768, padding_idx=1)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): RobertaEncoder(
        (layer): ModuleList(
          (0-11): 12 x RobertaLayer(
            (attention): RobertaAttention(
              (self): Fed_RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): Fed_RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): Fed_RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
    )
    (classifier): RobertaClassificationHead(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (out_proj): Linear(in_features=768, out_features=2, bias=True)
    )
  )
)
【【【eval】】】【【【【【【【for n,p in base_model.named_parameters():】】】】】】
RobertaForSequenceClassification(
  (roberta): RobertaModel(
    (embeddings): RobertaEmbeddings(
      (word_embeddings): Embedding(50265, 768, padding_idx=1)
      (position_embeddings): Embedding(514, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): RobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x RobertaLayer(
          (attention): RobertaAttention(
            (self): RobertaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): RobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): RobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): RobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (classifier): RobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=2, bias=True)
  )
)
roberta.embeddings.word_embeddings.weight torch.Size([50265, 768]) True
roberta.embeddings.position_embeddings.weight torch.Size([514, 768]) True
roberta.embeddings.token_type_embeddings.weight torch.Size([1, 768]) True
roberta.embeddings.LayerNorm.weight torch.Size([768]) True [0.28955078125, 0.35595703125, 0.33154296875]
roberta.embeddings.LayerNorm.bias torch.Size([768]) True [-0.1275634765625, 0.00823974609375, -0.0089874267578125]

roberta.encoder.layer.0.attention.self.query.weight torch.Size([768, 768]) True
roberta.encoder.layer.0.attention.self.query.bias torch.Size([768]) True
roberta.encoder.layer.0.attention.self.key.weight torch.Size([768, 768]) True
roberta.encoder.layer.0.attention.self.key.bias torch.Size([768]) True
roberta.encoder.layer.0.attention.self.value.weight torch.Size([768, 768]) True
roberta.encoder.layer.0.attention.self.value.bias torch.Size([768]) True

roberta.encoder.layer.0.attention.output.dense.weight torch.Size([768, 768]) True
roberta.encoder.layer.0.attention.output.dense.bias torch.Size([768]) True [-0.1044921875, -0.1632080078125, -0.00592041015625]
roberta.encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768]) True [0.556640625, 0.64111328125, 0.60302734375]
roberta.encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768]) True [0.27587890625, 0.0404052734375, 0.1912841796875]

roberta.encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768]) True
roberta.encoder.layer.0.intermediate.dense.bias torch.Size([3072]) True [-0.166259765625, -0.148193359375, -0.189208984375]

roberta.encoder.layer.0.output.dense.weight torch.Size([768, 3072]) True
roberta.encoder.layer.0.output.dense.bias torch.Size([768]) True [-0.09271240234375, -0.024749755859375, 0.01438140869140625]
roberta.encoder.layer.0.output.LayerNorm.weight torch.Size([768]) True [0.41064453125, 0.5986328125, 0.52197265625]
roberta.encoder.layer.0.output.LayerNorm.bias torch.Size([768]) True [-0.05194091796875, 0.097900390625, -0.0081634521484375]

classifier.dense.weight torch.Size([768, 768]) True
classifier.dense.bias torch.Size([768]) True [0.0, 0.0, 0.0]
classifier.out_proj.weight torch.Size([2, 768]) True
classifier.out_proj.bias torch.Size([2]) True


【【【eval】】】【【【【【【【for key in raw_state_dict:】】】】】】
roberta.embeddings.word_embeddings.weight
roberta.embeddings.position_embeddings.weight
roberta.embeddings.token_type_embeddings.weight
roberta.embeddings.LayerNorm.weight
roberta.embeddings.LayerNorm.bias

roberta.encoder.layer.0.attention.self.query.weight
roberta.encoder.layer.0.attention.self.query.bias
roberta.encoder.layer.0.attention.self.key.weight
roberta.encoder.layer.0.attention.self.key.bias
roberta.encoder.layer.0.attention.self.value.weight
roberta.encoder.layer.0.attention.self.value.bias

roberta.encoder.layer.0.attention.output.dense.weight
roberta.encoder.layer.0.attention.output.dense.bias
roberta.encoder.layer.0.attention.output.LayerNorm.weight
roberta.encoder.layer.0.attention.output.LayerNorm.bias

roberta.encoder.layer.0.intermediate.dense.weight
roberta.encoder.layer.0.intermediate.dense.bias

roberta.encoder.layer.0.output.dense.weight
roberta.encoder.layer.0.output.dense.bias
roberta.encoder.layer.0.output.LayerNorm.weight
roberta.encoder.layer.0.output.LayerNorm.bias

classifier.dense.weight
classifier.dense.bias
classifier.out_proj.weight
classifier.out_proj.bias
"""