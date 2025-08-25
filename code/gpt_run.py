import argparse
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
torch.manual_seed(seed)
import copy
from datetime import datetime
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, GenerationConfig
from tqdm import tqdm
import torch.nn as nn


import fed_utils
from data_utils import NLG_Dataset_Manager
from FedSN_gpt_model import FedSN_GPTLM_Model
from FedSN_model_utils import create_server_mask, create_client_mask, update_averaged_state_dict
from trainer_utils import NLG_Trainer
from result_manager import ResultManager
from quant_utils import Quant_Manager

if alg in ["FedSN"]:
    bit_width = 0
    block_size = 0

if model_name == "gpt2_medium":
    model_path = "/data/xzb/models/gpt2-medium"
    tokenizer_path = "/data/xzb/models/gpt2-medium"
    

if model_name == "gpt2_large":
    model_path = "/data/xzb/models/gpt2-large"
    tokenizer_path = "/data/xzb/models/gpt2-large"
   

if model_name == "gpt2_xl":
    model_path = "/data/xzb/models/gpt2-xl"
    tokenizer_path = "/data/xzb/models/gpt2-xl"
    

if dataset_name == "e2e":
    dataset_path = "/data/xzb/datasets/feddataset/e2e_NLG"
    max_length = 128
    weight_decay = 0.01
    label_smoothing = 0.1
    max_train_step = -1
    max_eval_step = -1
    max_test_step = -1
    
elif dataset_name == "dart":
    dataset_path = "/data/xzb/datasets/feddataset/dart_NLG"
    max_length = 128
    weight_decay = 0.01
    label_smoothing = 0.1

    max_train_step = -1
    max_eval_step = -1
    max_test_step = -1


elif dataset_name == "dialogsum":
    dataset_path = "/data/xzb/datasets/feddataset/dialogsum_NLG"
    max_length = 400
    weight_decay = 0.01
    label_smoothing = 0.1

    max_train_step = -1
    max_eval_step = -1
    max_test_step = -1


elif dataset_name == "viggo":
    dataset_path = "/data/xzb/datasets/feddataset/viggo_NLG"
    max_length = 128
    weight_decay = 0.01
    label_smoothing = 0.1

    max_train_step = -1
    max_eval_step = -1
    max_test_step = -1

elif dataset_name == "tibetan":
    dataset_path = "/data/xzb/datasets/tibetan-to-english"
    max_length = 128
    weight_decay = 0.01
    label_smoothing = 0.1

    max_train_step = -1
    max_eval_step = -1
    max_test_step = -1

if alg == "FedAVG":
    p1 = 0
    p2 = 0

num_beams = 10
do_sample = False
no_repeat_ngram_size = 4
length_penalty = 0.9
generation_length = 64
generation_config = GenerationConfig(
    num_beams = num_beams,
    do_sample = do_sample,
    no_repeat_ngram_size = no_repeat_ngram_size,
    length_penalty = length_penalty,
)


time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":", "").replace(" ", "").replace("-", "")
meta_dir = f"/data/xzb/FedQSN/[meta]/{run_name}/{model_name}_{dataset_name}_p1{p1}_p2{p2}_bit_{bit_width}_block_{block_size}_R{R}_N{N}C{C}_localepochs{local_epochs}_time_{time}"
save_dir = meta_dir.replace("[meta]", "save_models")
os.makedirs(save_dir, exist_ok=True)
generation_dir = meta_dir.replace("[meta]", "generation")
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
              \t\tdataset_name: {str(dataset_name)}\n\
              \t\tmax_length: {str(max_length)}\n\
              \tserver:\n\
              \t\tp1: {str(p1)}\n\
              \t\tp2: {str(p2)}\n\
              \t\tp1: {str(bit_width)}\n\
              \t\tp2: {str(block_size)}\n\
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
              \t\tmax_test_step: {str(max_test_step)}"
            )

base_model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
gpt2_config = base_model.config
base_model_state_dict = base_model.state_dict().copy()
model_state_dict = {k: v.to(torch.device("cuda:0")) for k, v in base_model_state_dict.items()}

loss_fct = nn.CrossEntropyLoss(ignore_index = -1, reduce = False, label_smoothing = label_smoothing)
trainer = NLG_Trainer(
    lr, local_epochs, gradient_accumulation_steps, 
    weight_decay, loss_fct, 
    max_train_step, max_eval_step, max_test_step,
    generation_config, generation_length, generation_dir
)
choosen_client_ids = fed_utils.client_choose(R, N, C)

dataset_manager = NLG_Dataset_Manager(dataset_path, max_length, batch_size, seed, N)
result_manager = ResultManager(N, R, logger, dataset_name, alg)

if alg =="FedQSN":
    quant_manager = Quant_Manager(logger, alg, model_name, gpt2_config, bit_width, block_size)
    client_quant_manager = Quant_Manager(logger, alg, model_name, gpt2_config, 4, block_size)


best_client_before_mask = None
best_client_after_mask = None
model_mask = create_server_mask(gpt2_config, "gpt2", 0)
fed_model = FedSN_GPTLM_Model(model_name, gpt2_config, model_mask, logger).to(torch.device("cuda:0"))




fed_model._model.load_state_dict(model_state_dict, strict=True)
model_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader())
better = result_manager._write_model_result(model_result, 0)
fed_utils.update_save_models(better, save_dir, "model", model_state_dict, None, None)


if alg in ["FedSN","FedQSN"]:
    server_mask = create_server_mask(gpt2_config, "gpt2", p1)
    fed_model._set_fed_mask(server_mask)
    server_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader())
    better = result_manager._write_server_result(server_result, 0)
    fed_utils.update_save_models(better, save_dir, "server", model_state_dict, None, None)
    torch.save(server_mask, save_dir + f"/server_mask.pth")
else:
    server_mask = model_mask
    torch.save(server_mask, save_dir + f"/server_mask.pth")


result_manager.print_result(-1)


for r in range(R):
    ava_state_dict = None
    for i in range(C):

        client_id = choosen_client_ids[r][i]
        fed_model._model.load_state_dict(model_state_dict, strict=True)
        client_mask = create_client_mask(gpt2_config, "gpt2", server_mask, p2, mask_type)
        fed_model._set_fed_mask(client_mask)
        
        ###量化###
        if alg =="FedQSN":
            quant_state_dict = quant_manager._quantize_model(model_state_dict)
            fed_model._model.load_state_dict(quant_state_dict, strict=True)
        
        result_before = trainer._evaluate(fed_model, dataset_manager._get_valid_loader())
        better = result_manager._write_clients_before_result(result_before, r, client_id)
        best_client_before_mask = fed_utils.update_save_models(better, save_dir, "client_before", model_state_dict, client_mask, best_client_before_mask)
        
        trainer._train(fed_model, dataset_manager._get_train_loader(client_id))
        client_state_dict = fed_model._model.state_dict().copy()
        
        if alg =="FedQSN":
            quant_client = client_quant_manager._quantize_model(client_state_dict)

        # result_after = trainer._evaluate(fed_model, dataset_manager._get_valid_loader())
        # better = result_manager._write_clients_after_result(result_after, r, client_id)
        # best_client_after_mask = fed_utils.update_save_models(better, save_dir, "client_after", client_state_dict, client_mask, best_client_after_mask)

        # if i == 0:
        #     ava_state_dict = client_state_dict
        # else:
        #     ava_state_dict = update_averaged_state_dict(ava_state_dict, client_state_dict, i)
        
        if i == 0:
            ava_state_dict = quant_client
        else:
            ava_state_dict = update_averaged_state_dict(ava_state_dict, quant_client, i)


    fed_model._model.load_state_dict(ava_state_dict, strict=True)
    fed_model._set_fed_mask(model_mask)
    model_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader())
    better = result_manager._write_model_result(model_result, r+1)
    fed_utils.update_save_models(better, save_dir, "model", ava_state_dict, None, None)
    
    if alg in ["FedSN","FedQSN"]:
        fed_model._set_fed_mask(server_mask)
        server_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader())
        better = result_manager._write_server_result(server_result, r+1)
        fed_utils.update_save_models(better, save_dir, "server", ava_state_dict, None, None)

    result_manager.print_result(r)

    model_state_dict = ava_state_dict


del model_state_dict
del ava_state_dict

logger.info(result_manager.get_best_round())
test_data_list, context_pred_refs_dict, context_list = fed_utils.generate_prepare(dataset_manager._get_test_loader(), dataset_path, generation_dir)

best_model_state_dict = torch.load(save_dir + '/best_model.pth')
fed_model._model.load_state_dict(best_model_state_dict, strict=True)
fed_model._set_fed_mask(model_mask)
fed_model._model = fed_model._model.to('cuda:0')
trainer._generate(fed_model, tokenizer, test_data_list, copy.deepcopy(context_pred_refs_dict), context_list, "model", logger)
del best_model_state_dict

# server_mask_path = save_dir+"/server_mask.pth"
# best_mask_client_before_path = save_dir+"/best_mask_client_before.pth"
# server_mask = torch.load(server_mask_path)
# best_client_before_mask = torch.load(best_mask_client_before_path)
if alg in["FedSN","FedQSN"]:
    best_server_state_dict = torch.load(save_dir + '/best_server.pth')
    fed_model._model.load_state_dict(best_server_state_dict, strict=True)
    fed_model._set_fed_mask(server_mask)
    fed_model._model = fed_model._model.to('cuda:0')
    trainer._generate(fed_model, tokenizer, test_data_list, copy.deepcopy(context_pred_refs_dict), context_list, "server", logger)
    del best_server_state_dict

best_client_before_state_dict = torch.load(save_dir + '/best_client_before.pth')
fed_model._model.load_state_dict(best_client_before_state_dict, strict=True)
fed_model._set_fed_mask(best_client_before_mask)
fed_model._model = fed_model._model.to('cuda:0')
trainer._generate(fed_model, tokenizer, test_data_list, copy.deepcopy(context_pred_refs_dict), context_list, "client_before", logger)
del best_client_before_state_dict

# best_client_after_state_dict = torch.load(save_dir + '/best_client_after.pth')
# fed_model._model.load_state_dict(best_client_after_state_dict, strict=True)
# fed_model._set_fed_mask(best_client_after_mask)
# fed_model._model = fed_model._model.to('cuda:0')
# trainer._generate(fed_model, tokenizer, test_data_list, copy.deepcopy(context_pred_refs_dict), context_list, "client_after", logger)
# del best_client_after_state_dict


"""import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type = str)
parser.add_argument("--model_name", type = str)
parser.add_argument("--dataset_name", type = str)
parser.add_argument("--p1", type = float)
parser.add_argument("--p2", type = float)
parser.add_argument("--N", type = int)
parser.add_argument("--C", type = int)
parser.add_argument("--R", type = int)
parser.add_argument("--local_epochs", type = int)
parser.add_argument("--lr", type = float)
parser.add_argument("--seed", type = int)
parser.add_argument("--device", type = int)

args = parser.parse_args()

run_name = str(args.run_name)
model_name = str(args.model_name)
dataset_name = str(args.dataset_name)
p1 = float(args.p1)
p2 = float(args.p2)
N = int(args.N)
C = int(args.C)
R = int(args.R)
local_epochs = int(args.local_epochs)
lr = float(args.lr)
seed = int(args.seed)
device = int(args.device)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


import torch
torch.manual_seed(seed)
import copy
from datetime import datetime
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, GenerationConfig
from tqdm import tqdm
import torch.nn as nn


import fed_utils
from data_utils import NLG_Dataset_Manager
from FedSN_gpt_model import FedSN_GPTLM_Model
from FedSN_model_utils import create_server_mask, create_client_mask, update_averaged_state_dict
from trainer_utils import NLG_Trainer
from result_manager import ResultManager







if model_name == "gpt2_medium":
    model_path = "/home/zhujh/20240417_FedQLoRA/model/gpt2_lm_medium"
    tokenizer_path = "/home/zhujh/20240417_FedQLoRA/model/gpt2_lm_medium/tokenizer"
    batch_size = 10
    gradient_accumulation_steps = 1
if model_name == "gpt2_xl":
    model_path = "/home/zhujh/20240417_FedQLoRA/model/gpt2_lm_xl"
    tokenizer_path = "/home/zhujh/20240417_FedQLoRA/model/gpt2_lm_xl/tokenizer"
    batch_size = 4
    gradient_accumulation_steps = 2

if dataset_name == "e2e":
    dataset_path = "/home/zhujh/20240417_FedQLoRA/data/e2e_NLG"
    max_length = 128
    weight_decay = 0.01
    label_smoothing = 0.1
    max_train_step = -1
    max_eval_step = 100
    max_test_step = -1
elif dataset_name == "dart":
    dataset_path = "/home/zhujh/20240417_FedQLoRA/data/dart_NLG"
    max_length = 128
    weight_decay = 0.01
    label_smoothing = 0.1
    max_train_step = -1
    max_eval_step = 100
    max_test_step = 2000
elif dataset_name == "dialogsum":
    dataset_path = "/home/zhujh/20240417_FedQLoRA/data/dialogsum_NLG"
    max_length = 400
    weight_decay = 0.01
    label_smoothing = 0.1
    max_train_step = -1
    max_eval_step = 100
    max_test_step = -1
    batch_size = 5
    gradient_accumulation_steps = 2
elif dataset_name == "viggo":
    dataset_path = "/home/zhujh/20240417_FedQLoRA/data/viggo_NLG"
    max_length = 128
    weight_decay = 0.01
    label_smoothing = 0.1
    max_train_step = -1
    max_eval_step = 100
    max_test_step = 1000

num_beams = 10
do_sample = False
no_repeat_ngram_size = 4
length_penalty = 0.9
generation_length = 64
generation_config = GenerationConfig(
    num_beams = num_beams,
    do_sample = do_sample,
    no_repeat_ngram_size = no_repeat_ngram_size,
    length_penalty = length_penalty,
)


time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":", "").replace(" ", "").replace("-", "")
meta_dir = f"/home/zhujh/20240902_FedSN/[meta]/{run_name}/time_{time}"
save_dir = meta_dir.replace("[meta]", "save_models")
os.makedirs(save_dir, exist_ok=True)
generation_dir = meta_dir.replace("[meta]", "generation")
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
              \t\tdataset_name: {str(dataset_name)}\n\
              \t\tmax_length: {str(max_length)}\n\
              \tserver:\n\
              \t\tp1: {str(p1)}\n\
              \t\tp2: {str(p2)}\n\
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
              \t\tmax_test_step: {str(max_test_step)}"
            )

base_model = GPT2LMHeadModel.from_pretrained(model_path)
gpt2_config = base_model.config


base_model_state_dict = base_model.state_dict().copy()
model_state_dict = {k: v.to(torch.device("cuda:0")) for k, v in base_model_state_dict.items()}

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token_id = 0

dataset_manager = NLG_Dataset_Manager(dataset_path, max_length, batch_size, seed, N)
result_manager = ResultManager(N, R, logger, dataset_name)

loss_fct = nn.CrossEntropyLoss(ignore_index = -1, reduce = False, label_smoothing = label_smoothing)
trainer = NLG_Trainer(
    lr, local_epochs, gradient_accumulation_steps, 
    weight_decay, loss_fct, 
    max_train_step, max_eval_step, max_test_step,
    generation_config, generation_length, generation_dir
)

choosen_client_ids = fed_utils.client_choose(R, N, C)


best_client_before_mask = None #客户端模型在训练前的最佳模型对应的那个mask
best_client_after_mask = None #客户端模型在训练后的最佳模型对应的那个mask
#之所以不记录server和model的mask是因为不需要，这两个mask在训练过程中一直都是固定的

model_mask = create_server_mask(gpt2_config, 0)
fed_model = FedSN_GPTLM_Model(model_name, gpt2_config, model_mask).to(torch.device("cuda:0"))



fed_model._model.load_state_dict(model_state_dict, strict=True)
model_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
better = result_manager._write_model_result(model_result, 0)
fed_utils.update_save_models(better, save_dir, "model", model_state_dict, None, None)



server_mask = create_server_mask(gpt2_config, p1)
fed_model._set_fed_mask(server_mask)
server_result = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
better = result_manager._write_server_result(server_result, 0)
fed_utils.update_save_models(better, save_dir, "server", model_state_dict, None, None)


result_manager.print_result(-1)

for r in range(R):
    ava_state_dict = None
    for i in range(C):

        client_id = choosen_client_ids[r][i]
        fed_model._model.load_state_dict(model_state_dict, strict=True)
        client_mask = create_client_mask(gpt2_config, server_mask, p2)
        fed_model._set_fed_mask(client_mask)

        result_before = trainer._evaluate(fed_model, dataset_manager._get_valid_loader(), "valid")
        better = result_manager._write_clients_before_result(result_before, r, client_id)
        best_client_before_mask = fed_utils.update_save_models(better, save_dir, "client_before", model_state_dict, client_mask, best_client_before_mask)

        logger.info("Trainable parameters:")
        for name, param in fed_model.named_parameters():
            if param.requires_grad:
                logger.info(f"\t{name}")
        
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

best_model_state_dict = torch.load(save_dir + '/best_model.pth')
fed_model._model.load_state_dict(best_model_state_dict, strict=True)
fed_model._set_fed_mask(model_mask)
fed_model._model = fed_model._model.to('cuda:0')
#

del best_model_state_dict

best_server_state_dict = torch.load(save_dir + '/best_server.pth')
fed_model._model.load_state_dict(best_server_state_dict, strict=True)
fed_model._set_fed_mask(server_mask)
fed_model._model = fed_model._model.to('cuda:0')
#
del best_server_state_dict

best_client_before_state_dict = torch.load(save_dir + '/best_client_before.pth')
fed_model._model.load_state_dict(best_client_before_state_dict, strict=True)
fed_model._set_fed_mask(best_client_before_mask)
fed_model._model = fed_model._model.to('cuda:0')
#
del best_client_before_state_dict

best_client_after_state_dict = torch.load(save_dir + '/best_client_after.pth')
fed_model._model.load_state_dict(best_client_after_state_dict, strict=True)
fed_model._set_fed_mask(best_client_after_mask)
fed_model._model = fed_model._model.to('cuda:0')
#
del best_client_after_state_dict"""

"""
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 1024)
    (wpe): Embedding(1024, 1024)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-23): 24 x GPT2Block(
        (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1024, out_features=50257, bias=False)
)
"""
