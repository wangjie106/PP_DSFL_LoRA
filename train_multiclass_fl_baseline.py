import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, matthews_corrcoef
import logging
import time
import copy
import random
import os
import csv
from dataclasses import dataclass, field
from typing import List

from data_utils import FT_Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@dataclass
class FedConfig:
    exp_name: str = "fedavg_baseline_smote"
    rounds: int = 100
    num_clients: int = 10
    clients_per_round: int = 4
    local_epochs: int = 1
    batch_size: int = 16
    max_seq_length: int = 128
    num_classes: int = 10
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    lr: float = 2e-5
    model_name: str = "./roberta-base-local"
    seed: int = 42

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]

class FedAvgTrainer:
    def __init__(self, config: FedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config.seed)
        
        try:
            base_model = RobertaForSequenceClassification.from_pretrained(
                config.model_name, 
                num_labels=config.num_classes,
                ignore_mismatched_sizes=True
            )
        except OSError:
            logging.error(f"Error: Unable to load model from '{config.model_name}'.")
            raise

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=config.lora_r, 
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout, 
            target_modules=config.target_modules, 
            modules_to_save=["classifier"], 
            bias="none"
        )
        self.global_model = get_peft_model(base_model, peft_config).to(self.device)
        self.history = []
        logging.info(f"Using device: {self.device}")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_client(self, global_weights, train_loader):
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(global_weights)
        local_model.train()
        
        optimizer = AdamW(local_model.parameters(), lr=self.config.lr)
        total_loss = 0
        num_batches = 0
        
        for _ in range(self.config.local_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = local_model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                
        return local_model.state_dict(), (total_loss / num_batches if num_batches > 0 else 0)

    def aggregate(self, w_locals):
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            if w_avg[k].dtype.is_floating_point:
                for i in range(1, len(w_locals)):
                    w_avg[k] += w_locals[i][k]
                w_avg[k] = w_avg[k] / len(w_locals)
        self.global_model.load_state_dict(w_avg)

    def evaluate(self, test_loader):
        self.global_model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.global_model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels']
                )
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds)
        return {'eval_loss': total_loss/len(test_loader) if len(test_loader) > 0 else 0, 'mcc': mcc, 'report': report}

    def run(self, dataset_train, dataset_test):
        dict_users_indices = np.array_split(range(len(dataset_train)), self.config.num_clients)
        train_loaders = [DataLoader(DatasetSplit(dataset_train, idxs), batch_size=self.config.batch_size, shuffle=True) for idxs in dict_users_indices]
        test_loader = DataLoader(dataset_test, batch_size=self.config.batch_size, shuffle=False)
        
        for round_num in range(1, self.config.rounds + 1):
            round_start = time.time()
            w_locals, local_losses = [], []
            
            selected_clients = random.sample(range(self.config.num_clients), self.config.clients_per_round)
            global_weights = self.global_model.state_dict()
            
            for client_idx in selected_clients:
                w, loss = self.train_client(global_weights, train_loaders[client_idx])
                w_locals.append(w)
                local_losses.append(loss)
            
            self.aggregate(w_locals)
            metrics = self.evaluate(test_loader)
            duration = time.time() - round_start
            
            self.history.append({
                'round': round_num,
                'avg_train_loss': np.mean(local_losses),
                'eval_loss': metrics['eval_loss'],
                'accuracy': metrics['report']['accuracy'],
                'mcc': metrics['mcc'],
                'weighted_f1': metrics['report']['weighted avg']['f1-score'],
                'duration': duration
            })
            logging.info(f"Round {round_num} | Acc: {metrics['report']['accuracy']:.4f} | MCC: {metrics['mcc']:.4f}")

    def save_results(self, output_dir="results_fedavg"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.config.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'avg_train_loss', 'eval_loss', 'accuracy', 'mcc', 'weighted_f1', 'duration'])
            for rec in self.history:
                writer.writerow([
                    rec['round'], rec['avg_train_loss'], rec['eval_loss'],
                    rec['accuracy'], rec['mcc'], rec['weighted_f1'], rec['duration']
                ])
        logging.info(f"Results saved to {filename}")

def main():
    config = FedConfig()
    train_data_path = 'processed_data_SMOTE/train_data.jsonl'
    test_data_path = 'processed_data_SMOTE/test_data.jsonl'
    
    if not all(os.path.exists(f) for f in [train_data_path, test_data_path]):
        logging.error("Data files not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset_train = FT_Dataset(train_data_path, config.batch_size, config.max_seq_length, tokenizer)
    dataset_test = FT_Dataset(test_data_path, config.batch_size, config.max_seq_length, tokenizer)
    
    trainer = FedAvgTrainer(config)
    trainer.run(dataset_train, dataset_test)
    trainer.save_results()

if __name__ == "__main__":
    main()
