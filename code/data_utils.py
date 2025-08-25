import json
from torch.utils.data import Dataset
import random
import torch
from torch.utils.data import DataLoader

class NLG_Dataset_Manager():
    def __init__(self, path, max_seq_length, batch_size, seed, N = 1):
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._rng = random.Random(seed)
        self._N = N
        self._train_samples = self._read_NLG_file(path + "/train.jsonl")
        self._valid_samples = self._read_NLG_file(path + "/valid.jsonl")
        self._test_samples = self._read_NLG_file(path + "/test.jsonl")
        self._rng.shuffle(self._train_samples)
        if self._N != 1:
            self._num_train = len(self._train_samples)
            self._num_client = int(self._num_train/self._N)
            self._train_samples_clients = []
            self._train_dataset_clients = []
            self._train_loader_clients = []
            for client_id in range(N):
                samples = self._train_samples[
                    client_id*self._num_client:
                    (client_id+1)*self._num_client
                ]
                self._train_samples_clients.append(samples)
                e2edataset = NLG_Dataset(samples, self._max_seq_length)
                self._train_dataset_clients.append(e2edataset)
                self._train_loader_clients.append(
                    DataLoader(
                        dataset = e2edataset, 
                        batch_size = self._batch_size,
                        shuffle=True
                    )
                )
        else:
            train_dataset = NLG_Dataset(self._train_samples, self._max_seq_length)
            self._train_loader = DataLoader(
                dataset = train_dataset, 
                batch_size = self._batch_size,
                shuffle=True
            )
        valid_dataset = NLG_Dataset(self._valid_samples, self._max_seq_length)
        test_dataset = NLG_Dataset(self._test_samples, self._max_seq_length, is_test = True)
        self._valid_loader = DataLoader(
            dataset = valid_dataset, 
            batch_size = self._batch_size,
        )
        self._test_loader = DataLoader(
            dataset = test_dataset, 
            batch_size = 1,
        )
    def _get_train_loader(self, client_id = -1):
        if self._N == 1:
            assert client_id == -1
            return self._train_loader
        else:
            return self._train_loader_clients[client_id]
    def _get_valid_loader(self):
        return self._valid_loader
    def _get_test_loader(self):
        return self._test_loader
        
    def _read_NLG_file(self, path):
        samples = []
        with open(path, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                samples.append([context, completion])
        return samples

class NLG_Dataset(Dataset):
    def __init__(self, data, max_seq_length, is_test = False):
        self._data = data
        self._max_seq_length = max_seq_length
        self._num_sample = len(self._data)
        self._is_test = is_test
    def _padding_tokens(self, tokens, pad_token, direct, max_context_length = 0):
        if max_context_length == 0:
            max_context_length = self._max_seq_length
        if len(tokens) > max_context_length:
            if direct > 0:
                pad_tokens = tokens[:max_context_length]
            else:
                pad_tokens = tokens[-max_context_length:]
        else:
            pad_tokens = tokens
        token_len = len(pad_tokens)
        pad_tokens = pad_tokens + [pad_token for _ in range(self._max_seq_length - token_len)]
        return pad_tokens, token_len
    def __len__(self):
        return len(self._data)
        
    def __getitem__(self, item):
        assert item < self._num_sample

        sample = self._data[item]
        conditions = sample[0]
        completion = sample[1]

        _input, _input_len = self._padding_tokens(conditions + completion, 0, 1)
        _target, _ = self._padding_tokens((conditions + completion)[1:], 0, 1)
        _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
        _msk, _ = self._padding_tokens(_msk, 0, 1)
        
        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)
        if(self._is_test):
            if len(conditions) > 1024 - 64 -1:
                conditions = conditions[:1024 - 64 -1]
            output["query"] = torch.tensor(conditions, dtype=torch.long)
            output["query_len"] = torch.tensor(len(conditions), dtype=torch.long)
        output["input"] = torch.tensor(_input, dtype=torch.long) 
        output["target"] = torch.tensor(_target, dtype=torch.long) 
        output["mask"] = torch.tensor(_msk, dtype=torch.float)
        return output
    




class CLS_Dataset_Manager():
    def __init__(self, dataset_path, max_length, batch_size, seed, N, p_train_valid):
        self._max_seq_length = max_length
        self._batch_size = batch_size
        self._rng = random.Random(seed)
        self._N = N
        self._train_valid_samples = self._read_CLS_file(dataset_path + "/train.jsonl")
        self._test_samples = self._read_CLS_file(dataset_path + "/valid.jsonl")
        
        self._num_valid = int(len(self._train_valid_samples)/p_train_valid)
        self._num_train = len(self._train_valid_samples) - self._num_valid
        self._num_client = int(self._num_train/self._N)
        self._train_samples = self._train_valid_samples[:self._num_train]
        self._valid_samples = self._train_valid_samples[self._num_train:]
        self._rng.shuffle(self._train_samples)
        self._train_samples_clients = []
        self._train_dataset_clients = []
        self._train_loader_clients = []
        for client_id in range(N):
            samples = self._train_samples[
                client_id*self._num_client:
                (client_id+1)*self._num_client
            ]
            self._train_samples_clients.append(samples)
            cls_dataset = CLS_Dataset(samples, self._max_seq_length)
            self._train_dataset_clients.append(cls_dataset)
            self._train_loader_clients.append(
                DataLoader(
                    dataset = cls_dataset, 
                    batch_size = self._batch_size,
                    shuffle=True
                )
            )
        valid_dataset = CLS_Dataset(self._valid_samples, self._max_seq_length)
        test_dataset = CLS_Dataset(self._test_samples, self._max_seq_length, is_test = True)
        self._valid_loader = DataLoader(
            dataset = valid_dataset, 
            batch_size = self._batch_size,
        )
        self._test_loader = DataLoader(
            dataset = test_dataset, 
            batch_size = 1,
        )
    def _get_train_loader(self, client_id):
        return self._train_loader_clients[client_id]
    def _get_valid_loader(self):
        return self._valid_loader
    def _get_test_loader(self):
        return self._test_loader
    def _read_CLS_file(name, path):
        samples = []
        with open(path, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                input_ids = items['input_ids']
                label = items['label']
                samples.append([input_ids, label])
        return samples


class CLS_Dataset(Dataset):
    def __init__(self, data, max_seq_length, is_test = False):
        self._data = data
        self._max_seq_length = max_seq_length
        self._num_sample = len(self._data)
        self._is_test = is_test
    def _padding_tokens(self, tokens, pad_token, direct, max_context_length = 0):
        if max_context_length == 0:
            max_context_length = self._max_seq_length
        if len(tokens) > max_context_length:
            if direct > 0:
                pad_tokens = tokens[:max_context_length]
            else:
                pad_tokens = tokens[-max_context_length:]
        else:
            pad_tokens = tokens
        token_len = len(pad_tokens)
        pad_tokens = pad_tokens + [pad_token for _ in range(self._max_seq_length - token_len)]
        return pad_tokens, token_len
    def __len__(self):
        return len(self._data)
        
    def __getitem__(self, item):
        assert item < self._num_sample

        sample = self._data[item]
        input_ids = sample[0]
        label = sample[1]

        _input, _input_len = self._padding_tokens(input_ids, 1, 1)
        _mask = [1]*len(input_ids)
        _mask, _ = self._padding_tokens(_mask, 0, 1)
        
        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)
        output["input_ids"] = torch.tensor(_input, dtype=torch.long) 
        output["label"] = torch.tensor(label, dtype=torch.long) 
        output["mask"] = torch.tensor(_mask, dtype=torch.long) 
        return output
    

class Llama_NLG_Dataset_Manager():
    def __init__(self, path, max_seq_length, batch_size, seed, N = 1):
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._rng = random.Random(seed)
        self._N = N
        self._train_samples = self._read_NLG_file(path + "/train.jsonl")
        self._valid_samples = self._read_NLG_file(path + "/valid.jsonl")
        self._test_samples = self._read_NLG_file(path + "/test.jsonl")
        self._rng.shuffle(self._train_samples)
        if self._N != 1:
            self._num_train = len(self._train_samples)
            self._num_client = int(self._num_train/self._N)
            self._train_samples_clients = []
            self._train_dataset_clients = []
            self._train_loader_clients = []
            for client_id in range(N):
                samples = self._train_samples[
                    client_id*self._num_client:
                    (client_id+1)*self._num_client
                ]
                self._train_samples_clients.append(samples)
                e2edataset = Llama_NLG_Dataset(samples, self._max_seq_length)
                self._train_dataset_clients.append(e2edataset)
                self._train_loader_clients.append(
                    DataLoader(
                        dataset = e2edataset, 
                        batch_size = self._batch_size,
                        shuffle=True
                    )
                )
        else:
            train_dataset = Llama_NLG_Dataset(self._train_samples, self._max_seq_length)
            self._train_loader = DataLoader(
                dataset = train_dataset, 
                batch_size = self._batch_size,
                shuffle=True
            )
        valid_dataset = Llama_NLG_Dataset(self._valid_samples, self._max_seq_length)
        test_dataset = Llama_NLG_Dataset(self._test_samples, self._max_seq_length, is_test = True)
        self._valid_loader = DataLoader(
            dataset = valid_dataset, 
            batch_size = self._batch_size,
        )
        self._test_loader = DataLoader(
            dataset = test_dataset, 
            batch_size = 1,
        )
    def _get_train_loader(self, client_id = -1):
        if self._N == 1:
            assert client_id == -1
            return self._train_loader
        else:
            return self._train_loader_clients[client_id]
    def _get_valid_loader(self):
        return self._valid_loader
    def _get_test_loader(self):
        return self._test_loader
        
    def _read_NLG_file(self, path):
        samples = []
        with open(path, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                samples.append([context, completion])
        return samples

class Llama_NLG_Dataset(Dataset):
    def __init__(self, data, max_seq_length, is_test = False):
        self._data = data
        self._max_seq_length = max_seq_length
        self._num_sample = len(self._data)
        self._is_test = is_test
    def _padding_tokens(self, tokens, pad_token, direct, max_context_length = 0):
        if max_context_length == 0:
            max_context_length = self._max_seq_length
        if len(tokens) > max_context_length:
            if direct > 0:
                pad_tokens = tokens[:max_context_length]
            else:
                pad_tokens = tokens[-max_context_length:]
        else:
            pad_tokens = tokens
        token_len = len(pad_tokens)
        pad_tokens = [pad_token for _ in range(self._max_seq_length - token_len)] + pad_tokens
        return pad_tokens, token_len
    def __len__(self):
        return len(self._data)
        
    def __getitem__(self, item):
        assert item < self._num_sample

        sample = self._data[item]
        conditions = sample[0]
        completion = sample[1]
# 0 0 0 1 2 3 a s d
# 0 0 0 2 3 a s d 0
# 0 0 0 0 0 1 1 1 0
        _input, _input_len = self._padding_tokens(conditions + completion, 0, 1)
        _target, _ = self._padding_tokens((conditions + completion)[1:] + [0], 0, 1)
        _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions)) + [0.0]
        _msk, _ = self._padding_tokens(_msk, 0, 1)
        
        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)
        if(self._is_test):
            if len(conditions) > 1024 - 64 -1:
                conditions = conditions[:1024 - 64 -1]
            output["query"] = torch.tensor(conditions, dtype=torch.long)
            output["query_len"] = torch.tensor(len(conditions), dtype=torch.long)
        output["input"] = torch.tensor(_input, dtype=torch.long) 
        output["target"] = torch.tensor(_target, dtype=torch.long) 
        output["mask"] = torch.tensor(_msk, dtype=torch.float)
        return output