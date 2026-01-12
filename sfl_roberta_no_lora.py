import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import time
import csv
import random
import copy
import os
from dataclasses import dataclass, field, asdict
from typing import List

# ============================== 1. 导入你自己的数据工具 ==============================
from data_utils import FT_Dataset, get_tokenizer

# ============================== 2. 配置和日志 ==============================
@dataclass
class ExperimentConfig:
    """实验配置"""
    exp_name: str = "dynamic_split_feddw_baseline_no_lora"
    rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 4
    local_epochs: int = 1
    batch_size: int = 16
    max_seq_length: int = 128
    lr: float = 2e-5
    # 请确认这个路径存在，或者直接改为 "roberta-base" 以自动下载
    model_name: str = "./roberta-base-local"
    seed: int = 42
    aggregation_method: str = "FedDW"  # 聚合方法: 'FedAvg' 或 'FedDW'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ============================== 3. 数据和模型定义 (无需修改) ==============================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs): self.dataset, self.idxs = dataset, list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item): return self.dataset[self.idxs[item]]

class ClientModel(nn.Module):
    def __init__(self, embeddings, encoder_layers, attention_mask_getter):
        super().__init__()
        self.embeddings = embeddings
        self.encoder_layers = encoder_layers
        self.get_extended_attention_mask = attention_mask_getter
    def forward(self, input_ids, attention_mask, device):
        hidden_states = self.embeddings(input_ids)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        for layer in self.encoder_layers: hidden_states = layer(hidden_states, extended_attention_mask)[0]
        return hidden_states

class ServerModel(nn.Module):
    def __init__(self, encoder_layers, classifier, attention_mask_getter):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.classifier = classifier
        self.get_extended_attention_mask = attention_mask_getter
    def forward(self, hidden_states, attention_mask, device, labels=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, hidden_states.size()[:-1], device)
        for layer in self.encoder_layers: hidden_states = layer(hidden_states, extended_attention_mask)[0]
        # RoBERTa 的分类器需要池化后的输出
        sequence_output = hidden_states
        # 通常分类器作用于 [CLS] token 的输出
        logits = self.classifier(sequence_output[:, 0, :])
        loss = nn.CrossEntropyLoss()(logits.view(-1, 2), labels.view(-1)) if labels is not None else None
        return logits, loss

def get_tensor_size_mb(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor): return 0
    return tensor.numel() * tensor.element_size() / (1024 * 1024)

def get_trainable_params_size_mb(model: nn.Module):
    total_size = 0
    for param in model.parameters():
        if param.requires_grad:
            total_size += param.numel() * param.element_size()
    return total_size / (1024 * 1024)

# ============================== 4. 训练器类 ==============================
class SFLTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config.seed)
        try:
            base_model_full = RobertaForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        except OSError:
            logging.error(f"找不到模型 '{config.model_name}'。请确认该文件夹存在或改为 Hugging Face 上的模型名。")
            raise
        self.net_glob_full = base_model_full.to(self.device)
        self.history = []
        self.client_resource_profiles = {'high': {'split_layer': 8}, 'medium': {'split_layer': 6}, 'low': {'split_layer': 4}}
        self.client_profiles = [random.choice(list(self.client_resource_profiles.keys())) for _ in range(config.num_clients)]
        logging.info(f"使用设备: {self.device}, 实验配置: {asdict(config)}")
        logging.info("客户端资源模拟分布:")
        for i, profile in enumerate(self.client_profiles):
            logging.info(f"  - 客户端 {i}: {profile} (分割点: {self.client_resource_profiles[profile]['split_layer']})")

    def set_seed(self, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def aggregate(self, w_locals, client_weights=None):
        if not w_locals: return
        if client_weights is None or self.config.aggregation_method == 'FedAvg':
            num_clients = len(w_locals)
            client_weights = [1.0 / num_clients] * num_clients
            logging.info(f"使用 FedAvg 聚合, 权重: {[f'{w:.2f}' for w in client_weights]}")
        else:
            logging.info(f"使用 FedDW 聚合, 权重: {[f'{w:.2f}' for w in client_weights]}")
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            if w_avg[k].dtype.is_floating_point:
                w_avg[k] *= client_weights[0]
        for i in range(1, len(w_locals)):
            for k in w_avg.keys():
                if w_avg[k].dtype.is_floating_point:
                    w_avg[k] += w_locals[i][k] * client_weights[i]
        self.net_glob_full.load_state_dict(w_avg)

    # [优化] 将 local_full_model 作为参数传入，逻辑更清晰
    def train_client(self, client_idx, local_full_model, local_net, server_model_for_client, train_loader):
        optimizer = AdamW(local_full_model.parameters(), lr=self.config.lr)
        total_loss, num_batches = 0, 0
        comm_stats = {'uplink_sfl': 0, 'downlink_sfl': 0}
        
        local_full_model.train() # 确保模型处于训练模式
        for epoch in range(self.config.local_epochs):
            pbar = tqdm(train_loader, desc=f"Client {client_idx} (Epoch {epoch+1})", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                fx = local_net(batch['input_ids'], batch['attention_mask'], self.device)
                comm_stats['uplink_sfl'] += get_tensor_size_mb(fx.detach())
                client_fx = fx.clone().detach().requires_grad_(True)
                logits, loss = server_model_for_client(client_fx, batch['attention_mask'], self.device, batch['labels'])
                if loss is None: continue
                loss.backward()
                if client_fx.grad is not None:
                    comm_stats['downlink_sfl'] += get_tensor_size_mb(client_fx.grad)
                    fx.backward(client_fx.grad)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss, comm_stats, local_full_model.state_dict()

    def evaluate(self, test_loader, split_layer_for_eval=6):
        full_model = self.net_glob_full
        full_model.eval()
        full_roberta_model = full_model.roberta
        client_net = ClientModel(full_roberta_model.embeddings, nn.ModuleList(full_roberta_model.encoder.layer[:split_layer_for_eval]), full_model.get_extended_attention_mask).to(self.device).eval()
        server_net = ServerModel(nn.ModuleList(full_roberta_model.encoder.layer[split_layer_for_eval:]), full_model.classifier, full_model.get_extended_attention_mask).to(self.device).eval()
        all_preds, all_labels = [], []; total_eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                fx = client_net(batch['input_ids'], batch['attention_mask'], self.device)
                logits, loss = server_net(fx, batch['attention_mask'], self.device, batch['labels'])
                if loss is not None: total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(batch['labels'].cpu().numpy())
        metrics = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        avg_eval_loss = total_eval_loss / len(test_loader) if len(test_loader) > 0 else 0
        return {'acc': accuracy_score(all_labels, all_preds), 'precision': metrics[0], 'recall': metrics[1], 'f1': metrics[2], 'eval_loss': avg_eval_loss}

    def run(self, dataset_train, dataset_test):
        dict_users = [set(indices) for indices in np.array_split(range(len(dataset_train)), self.config.num_clients)]
        train_loaders = [DataLoader(DatasetSplit(dataset_train, list(idxs)), batch_size=self.config.batch_size, shuffle=True) for idxs in dict_users]
        test_loader_global = DataLoader(dataset_test, batch_size=self.config.batch_size, shuffle=False)
        logging.info(f"开始训练: {self.config.rounds} 轮")
        for round_num in range(1, self.config.rounds + 1):
            logging.info(f"\n--- Round {round_num}/{self.config.rounds} ---")
            selected_clients = np.random.choice(range(self.config.num_clients), self.config.clients_per_round, replace=False)
            logging.info(f"选中的客户端: {selected_clients}")
            w_locals, local_losses = [], []
            round_comm_stats = {'total_uplink': 0, 'total_downlink': 0}
            client_split_layers = []
            for idx in selected_clients:
                profile = self.client_profiles[idx]
                split_layer = self.client_resource_profiles[profile]['split_layer']
                client_split_layers.append(split_layer)
                logging.info(f"  客户端 {idx} (资源: {profile}) -> 动态分割点: {split_layer}")
                local_full_model = copy.deepcopy(self.net_glob_full)
                full_roberta_model = local_full_model.roberta
                client_embeddings, client_encoder_layers = full_roberta_model.embeddings, nn.ModuleList(full_roberta_model.encoder.layer[:split_layer])
                server_encoder_layers, server_classifier = nn.ModuleList(full_roberta_model.encoder.layer[split_layer:]), local_full_model.classifier
                local_net = ClientModel(client_embeddings, client_encoder_layers, local_full_model.get_extended_attention_mask).to(self.device)
                server_model_for_client = ServerModel(server_encoder_layers, server_classifier, local_full_model.get_extended_attention_mask).to(self.device)
                
                # [优化] 将 local_full_model 传入 train_client
                avg_loss, comm_stats, client_trained_state_dict = self.train_client(
                    idx, local_full_model, local_net, server_model_for_client, train_loaders[idx]
                )
                
                round_comm_stats['total_uplink'] += comm_stats['uplink_sfl']
                round_comm_stats['total_downlink'] += comm_stats['downlink_sfl']
                round_comm_stats['total_uplink'] += get_trainable_params_size_mb(local_full_model)
                local_losses.append(avg_loss)
                w_locals.append(client_trained_state_dict)

            client_weights = None
            if self.config.aggregation_method == 'FedDW':
                total_split_layers = sum(client_split_layers)
                if total_split_layers > 0: client_weights = [sl / total_split_layers for sl in client_split_layers]
            self.aggregate(w_locals, client_weights)
            round_comm_stats['total_downlink'] += get_trainable_params_size_mb(self.net_glob_full) * len(selected_clients)
            avg_round_loss = np.mean(local_losses) if local_losses else 0
            metrics = self.evaluate(test_loader_global)
            round_summary = {'round': round_num, 'avg_train_loss': avg_round_loss, 'total_uplink_mb': round_comm_stats['total_uplink'], 'total_downlink_mb': round_comm_stats['total_downlink'], **metrics}
            self.history.append(round_summary)
            print()
            logging.info(f"--- Round {round_num} 总结 ---")
            logging.info(f"  平均客户端训练损失: {avg_round_loss:.4f}")
            logging.info(f"  本轮上行通信量 (MB): {round_comm_stats['total_uplink']:.4f}")
            logging.info(f"  本轮下行通信量 (MB): {round_comm_stats['total_downlink']:.4f}")
            logging.info(f"  全局模型评估损失: {metrics['eval_loss']:.4f}")
            logging.info(f"  全局模型评估准确率: {metrics['acc']:.4f}")
            logging.info(f"  全局模型评估 F1 分数: {metrics['f1']:.4f}")
            print("-" * (len(f"--- Round {round_num} 总结 ---") + 2))

    def save_results(self, output_dir="results_baseline_no_lora"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"{self.config.exp_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            config_dict = asdict(self.config)
            for key, value in config_dict.items(): writer.writerow([f'# {key}', value])
            writer.writerow([])
            if self.history:
                header = list(self.history[0].keys())
                writer.writerow(header)
                for record in self.history:
                    writer.writerow([f"{record.get(k, ''):.4f}" if isinstance(record.get(k, ''), (int, float)) else record.get(k, '') for k in header])
        logging.info(f"结果已保存到: {filepath}")

# ============================== 5. 主函数 ==============================
def main():
    config = ExperimentConfig()
    
    # 使用你自己的 data_utils.py 中的函数
    tokenizer = get_tokenizer(config.model_name)
    
    # 请确保数据路径正确
    logging.info("正在加载训练数据...")
    dataset_train = FT_Dataset('processed_data/train_data.jsonl', config.batch_size, config.max_seq_length, tokenizer)
    logging.info("正在加载测试数据...")
    dataset_test = FT_Dataset('processed_data/test_data.jsonl', config.batch_size, config.max_seq_length, tokenizer)
        
    trainer = SFLTrainer(config)
    trainer.run(dataset_train, dataset_test)
    trainer.save_results()
    logging.info("\n动态分割联邦学习 (无LoRA微调基线) 实验完成!")

if __name__ == "__main__":
    main()

