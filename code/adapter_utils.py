from scipy.stats import norm
import torch
import math

class AdapterManager:
    def __init__(self, alg, adapter_lr, update_method):
        if alg == "FedAVG":
            quanti_bit = 0
            block_size = 0
        elif "FedQLoRA" in alg:
            quanti_bit = int(alg.split('_')[1])
            block_size = int(alg.split('_')[2])
        else:
            quanti_bit = 0
            block_size = 0
        quanti_bit = 0
        block_size = 0
        self._update_method = update_method

        self._alg = alg
        self._quanti_bit = quanti_bit
        self._block_size = block_size
        self._adapter_lr = adapter_lr
        if quanti_bit != 0:
            self._create_normal_map()
    def _create_normal_map(self):
        if self._quanti_bit >=2:
            self._offset = 0.9677083
            self._positive_num = 2**(self._quanti_bit-1)
            self._negetive_num = 2**(self._quanti_bit-1) - 1 
            v1 = norm.ppf(torch.linspace(self._offset, 0.5, self._positive_num + 1)[:-1]).tolist() # 正数部分
            v2 = [0] ## we have 15 non-zero values in this data type
            v3 = (-norm.ppf(torch.linspace(self._offset, 0.5, self._negetive_num + 1)[:-1])).tolist() #负数部分
            v = v1 + v2 + v3
            values = torch.Tensor(v)
            values = values.sort().values
            values /= values.max()
            self._Q = values.tolist()
        if self._quanti_bit == 1.5:
            self._Q = [-1.0, 0.0, 1.0]
        if self._quanti_bit == 1:
            self._Q = [-1.0, 0.0, 1.0]

    def _block(self, tensor):
        tensor_1D = tensor.view(-1)
        last_block_len = tensor_1D.shape[0] % self._block_size
        if(last_block_len!=0):
            last_block = tensor_1D[-last_block_len:].tolist()
            full_blocks = tensor_1D[:-last_block_len].view(-1, self._block_size).tolist()
            return full_blocks + [last_block]
        else:
            full_blocks = tensor_1D.view(-1, self._block_size).tolist()
            return full_blocks
    def _deblcok(self, blocks, shape, deivce, dtype):
        list_1D = []
        for block in blocks:
            list_1D = list_1D + block
        return torch.tensor(list_1D, device = deivce, dtype = dtype).reshape(shape)


    def _quantize(self, tensor):
        shape = tensor.shape
        deivce = tensor.device
        dtype = tensor.dtype
        blocked_tensor = self._block(tensor)
        quantize_result = []
        quantize_constant = []
        for block in blocked_tensor:
            c = max([abs(val) for val in block])
            if(c == 0):
                c = 1e-8
            quantize_constant.append(c)
            norm_block = [val/c for val in block]
            block_result = []
            for norm_val in norm_block:
                min_sim = math.inf
                idx = -1
                for j, q in enumerate(self._Q): # 寻找Q中最近值的索引
                    sim = abs(norm_val - q)
                    if sim < min_sim:
                        min_sim = sim
                        idx = j
                block_result.append(idx)
            quantize_result.append(block_result)
        return quantize_constant, quantize_result, shape, deivce, dtype
    
    def _dequantize(self, quantize_constant, quantize_result, shape, device, dtype):
        dequantize_result = []
        for idx, quantized_block in enumerate(quantize_result):
            c = quantize_constant[idx]
            dequantize_result.append([self._Q[val] * c for val in quantized_block])
        dequanti_tensor = self._deblcok(dequantize_result, shape, device, dtype)
        return dequanti_tensor
    
    def _encode_adapter(self, adapter):
        encoded_adapter = {}
        for n in adapter:
            quantize_constant, quantize_result, shape, deivce, dtype = self._quantize(adapter[n])
            encoded_adapter[n] = self._dequantize(quantize_constant, quantize_result, shape, deivce, dtype)
        return encoded_adapter


    def _get_adapter_from_model(self, model):
        adapter = {}
        for n, p in model.named_parameters():
            if (("lora_A.default" in n) or ("lora_B.default" in n)):
                adapter[n] = p.clone()
        return adapter
    def _set_adapter_of_model(self, adapter, model):
        for n, p in model.named_parameters():
            if n in adapter:
                p.data = adapter[n].clone()
    def _get_contributed_adapter(self, client_adapters):
        contributed_adapter = {}
        for n in client_adapters[0]:
            adapters = [client_adapters[i][n] for i in range(len(client_adapters))]
            mean_adapter = torch.mean(torch.stack(adapters), dim=0)  
            contributed_adapter[n] = mean_adapter 
        return contributed_adapter
    def _get_server_adapter(self, server_adapter, distributed_adapter, contributed_adapter, r, R):
        ada_lr = self._get_adapter_lr_(r, R)
        for n in distributed_adapter:
            delta = contributed_adapter[n] - distributed_adapter[n]
            if self._update_method == 1:
                server_adapter[n] = server_adapter[n] + ada_lr * delta
            elif self._update_method == 2:
                server_adapter[n] = distributed_adapter[n] + ada_lr * delta
            elif self._update_method in [3,4]:
                server_adapter[n] = server_adapter[n] + ada_lr * (contributed_adapter[n] - server_adapter[n])
        return server_adapter
    def _get_distributed_adapter(self, server_adapter):
        if "FedQLoRA" in self._alg:
            return  self._encode_adapter(server_adapter)
        elif self._alg == "FedAVG":
            return server_adapter
        else:
            return server_adapter
    def _get_adapter_lr_(self, r, R):
        if self._update_method == 1:
            return self._adapter_lr
        elif self._update_method == 2:
            return self._adapter_lr
        elif self._update_method == 3:
            return self._adapter_lr
        elif self._update_method == 4:
            return self._adapter_lr * (R-r) / R





