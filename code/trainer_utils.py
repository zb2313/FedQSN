import torch
from tqdm import tqdm
from transformers import AdamW, get_scheduler

class NLG_Trainer():
    def __init__(
        self, lr, local_epochs, gradient_accumulation_steps, 
        weight_decay, loss_fct,
        max_train_step, max_eval_step, max_test_step,
        generation_config, generation_length, generation_dir):


        self._lr = lr
        self._local_epochs = local_epochs
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._weight_decay = weight_decay
        self._loss_fct = loss_fct
        self._max_train_step = max_train_step
        self._max_eval_step = max_eval_step
        self._max_test_step = max_test_step
        self._generation_config = generation_config
        self._generation_length = generation_length
        self._generation_dir = generation_dir


    def _train(self, fed_model, train_dataLoader):
        if self._max_train_step == -1:
            total_train_step = self._local_epochs * int(len(train_dataLoader) / self._gradient_accumulation_steps)
        else:
            total_train_step = self._max_train_step
        warmup_step = int(total_train_step * 0.06)+1
        optimizer = AdamW(params = fed_model._model.parameters(), lr = self._lr, weight_decay = self._weight_decay)
        lr_scheduler = get_scheduler("linear", optimizer = optimizer, num_warmup_steps = warmup_step, num_training_steps = total_train_step)
        

        fed_model._model.train()
        train_step = 0
        for epoch in range(self._local_epochs):
            if(train_step == self._max_train_step):
                break
            for step, batch in enumerate(tqdm(train_dataLoader)):
                data = {k:v.to(torch.device("cuda")) for k,v in batch.items()}

                _input = data['input']
                _batch, _len = _input.shape
                _target = data['target']
                _msk = data['mask']
                output = fed_model._model(_input)
                lm_logits = output.logits
                loss = self._loss_fct(lm_logits.view(-1, lm_logits.size(-1)), _target.view(-1)).view(_batch, _len)

                loss = loss * _msk 
                loss = loss.sum() / (_msk.sum() + 0.0001)
                loss = loss.mean() 
                
                loss = loss / self._gradient_accumulation_steps
                loss.backward()

                if ((step + 1) % self._gradient_accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1
                    lr_scheduler.step()
                if(train_step == self._max_train_step):
                    break

    

    def _evaluate(self, fed_model, valid_dataLoader):
        eval_loss = 0
        eval_step = 0

        fed_model._model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_dataLoader)):
                data = {k:v.to(torch.device("cuda")) for k,v in batch.items()}

                _input = data['input']
                _batch, _len = _input.shape
                _target = data['target']
                _msk = data['mask']
                output = fed_model._model(_input)
                lm_logits = output.logits
                loss = self._loss_fct(lm_logits.view(-1, lm_logits.size(-1)), _target.view(-1)).view(_batch, _len)

                loss = loss * _msk 
                loss = loss.sum() / (_msk.sum() + 0.0001)
                loss = loss.mean() 
                
                eval_loss+=loss
                eval_step+=1
                if(eval_step == self._max_eval_step):
                    break

        avg_eval_loss  = eval_loss / eval_step
        return {
            "loss":avg_eval_loss.item(),
        }

    
    def _generate(self, fed_model, tokenizer, test_data_list, context_pred_refs_dict, context_list, name, logger):
        fed_model._model.eval()

        with torch.no_grad():
            for step , data in enumerate(tqdm(test_data_list)):
                output = fed_model._model.generate(
                    input_ids=data["query"], 
                    generation_config = self._generation_config, 
                    max_length=data["query_len"].item() + self._generation_length
                )
                output = tokenizer.decode(output.tolist()[0])
                #logger.info(output)
                pred = output.split('<|endoftext|>')[1].split('\n\n')[0].strip() 
                id = data["id"]
                context = context_list[id]
                assert context_pred_refs_dict[context]["pred"] == "[No answer]"
                context_pred_refs_dict[context]["pred"] = pred
                if step == self._max_test_step:
                    break
        refss = [context_pred_refs_dict[context]['refs'] for context in context_pred_refs_dict]
        preds = [context_pred_refs_dict[context]['pred'] for context in context_pred_refs_dict]
        with open(f"{self._generation_dir}/refs.txt", 'w', encoding='utf8') as refs_writer, \
            open(f"{self._generation_dir}/pred_{name}.txt", 'w', encoding='utf8') as pred_writer:
            for refs, pred in zip(refss, preds):
                for r in refs:
                    refs_writer.write(r + '\n')
                refs_writer.write('\n')
                pred_writer.write(pred + '\n')

class CLS_Trainer():
    def __init__(
        self, 
        dataset_name,
        lr, local_epochs, gradient_accumulation_steps, 
        weight_decay, loss_fct,
        max_train_step, max_eval_step, max_test_step):

        self._dataset_name = dataset_name
        self._lr = lr
        self._local_epochs = local_epochs
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._weight_decay = weight_decay
        self._loss_fct = loss_fct
        self._max_train_step = max_train_step
        self._max_eval_step = max_eval_step
        self._max_test_step = max_test_step



    def _train(self, fed_model, train_dataLoader):
        if self._max_train_step == -1:
            total_train_step = self._local_epochs * int(len(train_dataLoader) / self._gradient_accumulation_steps + 1)
        else:
            total_train_step = self._max_train_step
        warmup_step = int(total_train_step * 0.06)+1
        optimizer = AdamW(params = fed_model._model.parameters(), lr = self._lr, weight_decay = self._weight_decay)
        lr_scheduler = get_scheduler("linear", optimizer = optimizer, num_warmup_steps = warmup_step, num_training_steps = total_train_step)
        
        fed_model._model.train()
        train_step = 0
        for epoch in range(self._local_epochs):
            if(train_step == self._max_train_step):
                break
            for step, batch in enumerate(tqdm(train_dataLoader)):
                data = {k:v.to(torch.device("cuda")) for k,v in batch.items()}

                input = data['input_ids']
                label = data['label']
                mask = data["mask"]

                output = fed_model._model(input,mask)
                logits = output.logits
                loss = self._loss_fct(logits, label)
                
                loss = loss / self._gradient_accumulation_steps
                loss.backward()

                if (((step + 1) % self._gradient_accumulation_steps) == 0)or(step == len(train_dataLoader)-1):
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1
                    lr_scheduler.step()
                if(train_step == self._max_train_step):
                    break


    

    def _evaluate(self, fed_model, valid_dataLoader, name):
        if name == "valid":
            max_step = self._max_eval_step
        elif name == "test":
            max_step = self._max_test_step
        eval_loss = 0
        eval_T = 0
        if(self._dataset_name in ['mrpc']):
            eval_TP = 0
            eval_TN = 0
            eval_FP = 0
            eval_FN = 0
        eval_num = 0
        eval_step = 0

        fed_model._model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_dataLoader)):
                data = {k:v.to(torch.device("cuda")) for k,v in batch.items()}

                input = data['input_ids']
                label = data['label']
                mask = data["mask"]

                output = fed_model._model(input,mask)
                logits = output.logits
                loss = self._loss_fct(logits, label)
                
                pred = torch.argmax(logits, dim = 1)

                eval_T += torch.sum(pred == label).item()
                eval_num += input.shape[0]
                if(self._dataset_name in ['mrpc']):
                    eval_TP += torch.sum((pred == 1) & (label == 1)).item()
                    eval_TN += torch.sum((pred == 0) & (label == 0)).item()
                    eval_FP += torch.sum((pred == 1) & (label == 0)).item()
                    eval_FN += torch.sum((pred == 0) & (label == 1)).item()

                eval_loss+=loss
                eval_step+=1
                if(eval_step == max_step):
                    break

        avg_eval_loss  = eval_loss / eval_step
        avg_acc = eval_T / eval_num
        if(self._dataset_name in ['mrpc']):
            precision = eval_TP/(eval_TP + eval_FP)
            recall = eval_TP/(eval_TP + eval_FN)
            f1score = 2*precision*recall/(precision + recall)
            return {
                "loss":avg_eval_loss.item(),
                "acc":avg_acc,
                "f1":f1score,
                "num":eval_num,
            }
        else:
            return {
                "loss":avg_eval_loss.item(),
                "acc":avg_acc,
                "num":eval_num
            }
    
    
    