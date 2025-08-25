## FedQSN_EMNLP2025
### Abstract
The primary goal of traditional federated learn-
ing is to protect data privacy by enabling dis-
tributed edge devices to collaboratively train a
shared global model while keeping raw data de-
centralized at local clients. The rise of large lan-
guage models (LLMs) has introduced new chal-
lenges in distributed systems, as their substan-
tial computational requirements and the need
for specialized expertise raise critical concerns
about protecting intellectual property (IP). This
highlights the need for a federated learning ap-
proach that can safeguard both sensitive data
and proprietary models. To tackle this chal-
lenge, we propose FedQSN, a federated learn-
ing approach that leverages random masking to
obscure a subnetwork of model parameters and
applies quantization to the remaining parame-
ters. Consequently, the server transmits only a
privacy-preserving proxy of the global model to
clients during each communication round, thus
enhancing the model’s confidentiality. Experi-
mental results across various models and tasks
demonstrate that our approach not only main-
tains strong model performance in federated
learning settings but also achieves enhanced
protection of model parameters compared to
baseline methods.
![main_graph](source/main.png)
