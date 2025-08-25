class ResultManager:
    def __init__(self, N, R, logger, dataset_name, alg):
        self._N = N
        self._R = R
        self._logger = logger
        self._dataset_name = dataset_name
        self._alg = alg

        self._model_result = {}
        if alg in ["FedSN","FedQSN"]:
            self._server_result = {}
        self._clients_before_result = {}
        self._clients_after_result = {}

        self._best_model_loss = float('inf')
        if alg in ["FedSN","FedQSN"]:
            self._best_server_loss = float('inf')
        self._best_client_before_loss = float('inf')
        self._best_client_after_loss = float('inf')

        if dataset_name in ["rte", "sst2", "qqp", "mnli", "qnli"]: 
            self._metrics = ["loss", "acc"]
        if dataset_name in ["mrpc"]:
            self._metrics = ["loss", "acc", "f1"]
        elif dataset_name in ["e2e", "dart", "viggo", "dialogsum", "tibetan"]: 
            self._metrics = ["loss"]
        for metric in self._metrics:
            self._model_result[metric] = [-1 for r in range(self._R + 1)]
            if alg in ["FedSN","FedQSN"]:
                self._server_result[metric] = [-1 for r in range(self._R + 1)]
            self._clients_before_result[metric] = [[-1 for i in range(self._R)] for r in range(self._N)]
            self._clients_after_result[metric] = [[-1 for i in range(self._R)] for r in range(self._N)]

    
    def _write_model_result(self, result, r):
        for metric in self._metrics:
            self._model_result[metric][r] = result[metric]
        if result["loss"]<self._best_model_loss:
            self._best_model_loss = result["loss"]
            return True
        else:
            return False
    
    def _write_server_result(self, result, r):
        assert self._alg in ["FedSN","FedQSN"] 
        for metric in self._metrics:
            self._server_result[metric][r] = result[metric]
        if result["loss"]<self._best_server_loss:
            self._best_server_loss = result["loss"]
            return True
        else:
            return False

    def _write_clients_before_result(self, result, r, i):
        for metric in self._metrics:
            self._clients_before_result[metric][i][r] = result[metric]
        if result["loss"]<self._best_client_before_loss:
            self._best_client_before_loss = result["loss"]
            return True
        else:
            return False
                
    def _write_clients_after_result(self, result, r, i):
        for metric in self._metrics:
            self._clients_after_result[metric][i][r] = result[metric]
        if result["loss"]<self._best_client_after_loss:
            self._best_client_after_loss = result["loss"]
            return True
        else:
            return False

    def print_list(self, list, dp, name):
        formatted_row = '[' + ', '.join([f'{num:.{dp}f}' if num != -1 else '0' for num in list]) + ']'
        self._logger.info("\t\t" + name + ": " + formatted_row)
    def print_result(self, r):
        self._logger.info(f'round{str(r)}')
        for metric in self._metrics:
            self._logger.info("\t" + metric)
            dp = 4 if "loss" in metric else 4
            self.print_list(self._model_result[metric], dp, "model")
            if self._alg in ["FedSN","FedQSN"]:
                self.print_list(self._server_result[metric], dp, "server")
            for i in range(self._N):
                self.print_list(self._clients_before_result[metric][i], dp, f"clients before {i}")
                self.print_list(self._clients_after_result[metric][i], dp, f"clients after {i}")
    def find_min_element_index(self, arr):
        min_value = float('inf')  # 初始设定一个无穷大的值作为最小值
        min_index = (-1, -1)  # 初始设定一个无效的下标

        # 遍历二维数组
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                if arr[i][j] != -1 and arr[i][j] < min_value:
                    min_value = arr[i][j]
                    min_index = (i, j)

        return min_index
    def get_best_round(self):


        model_loss = self._model_result["loss"]
        best_model = model_loss.index(min(model_loss))  

        if self._alg in ["FedSN","FedQSN"]:
            server_loss = self._server_result["loss"]
            best_server = server_loss.index(min(server_loss))

        clients_before_loss = self._clients_before_result["loss"]
        best_clients_before = self.find_min_element_index(clients_before_loss)
        
        clients_after_loss = self._clients_after_result["loss"]
        best_clients_after = self.find_min_element_index(clients_after_loss)

        if self._alg in ["FedSN","FedQSN"]:
            return {
                "best_server": best_server,
                "best_model": best_model,
                "best_clients_before": best_clients_before,
                "best_clients_after": best_clients_after,
            }
        else:
            return {
                "best_model": best_model,
                "best_clients_before": best_clients_before,
                "best_clients_after": best_clients_after,
            } 


