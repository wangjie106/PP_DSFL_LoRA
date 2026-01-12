import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    matthews_corrcoef
)
import logging
import time
import csv
import random
import copy
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

from data_utils import FT_Dataset

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

@dataclass
class ExperimentConfig:
    exp_name: str = "multiclass_fl_baseline_fedavg"
    rounds: int = 50
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
    
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: List[float] = field(default_factory=lambda: [
        1.0, 5.0, 5.0, 1.5, 1.0, 1.5, 1.0, 1.5, 8.0, 10.0
    ])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs): self.dataset, self.idxs = dataset, list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item): return self.dataset[self.idxs[item]]

def get_trainable_params_size_mb(model: nn.Module):
    total_size = 0
    for param in model.parameters():
        if param.requires_grad:
            total_size += param.numel() * param.element_size()
    return total_size / (1024 * 1024)

class FLTrainer:
    def __init__(self, config: ExperimentConfig):
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
            logging.error(f"Error: Unable to load model from local path '{config.model_name}'.")
            raise

        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=config.target_modules, modules_to_save=["classifier"], bias="none")
        self.global_model = get_peft_model(base_model, peft_config).to(self.device)
        self.history = []
        
        if config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        else:
            weights = torch.tensor(config.focal_alpha) if config.focal_alpha else None
            self.loss_fct = nn.CrossEntropyLoss(weight=weights)
            
        logging.info(f"Using device: {self.device}, Loss Function: {'Focal Loss' if config.use_focal_loss else'Cross Entropy'}, Config: {asdict(config)}")

    def set_seed(self, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def aggregate(self, w_locals):
        if not w_locals: return
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            if w_avg[k].dtype.is_floating_point:
                for i in range(1, len(w_locals)):
                    w_avg[k] += w_locals[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w_locals))
        self.global_model.load_state_dict(w_avg)

    def train_client(self, client_idx, train_loader):
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        optimizer = AdamW(local_model.parameters(), lr=self.config.lr)
        
        total_loss, num_batches = 0, 0
        
        for epoch in range(self.config.local_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                
                logits = local_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                loss = self.loss_fct(logits.view(-1, self.config.num_classes), batch['labels'].view(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return local_model.state_dict(), avg_loss

    def evaluate(self, test_loader):
        self.global_model.eval()
        all_preds, all_labels, total_eval_loss = [], [], 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                logits = self.global_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                loss = self.loss_fct(logits.view(-1, self.config.num_classes), batch['labels'].view(-1))
                
                if loss is not None: total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(batch['labels'].cpu().numpy())
        
        report_dict = classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(self.config.num_classes)], zero_division=0, output_dict=True)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        logging.info(f"\nClassification Evaluation Report:\n{classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(self.config.num_classes)], zero_division=0)}")
        
        return {
            'eval_loss': total_eval_loss / len(test_loader) if len(test_loader) > 0 else 0,
            'mcc': mcc,
            'report': report_dict 
        }

    def run(self, dataset_train, dataset_test):
        dict_users_indices = np.array_split(range(len(dataset_train)), self.config.num_clients)
        train_loaders = [DataLoader(DatasetSplit(dataset_train, list(idxs)), self.config.batch_size, shuffle=True) for idxs in dict_users_indices]
        test_loader_global = DataLoader(dataset_test, self.config.batch_size, shuffle=False)
        
        logging.info(f"Start FL training {self.config.rounds} rounds...")
        
        for round_num in range(1, self.config.rounds + 1):
            round_start_time = time.time()
            logging.info(f"\n--- Round {round_num}/{self.config.rounds} ---")
            
            selected_clients = sorted(random.sample(range(self.config.num_clients), self.config.clients_per_round))
            logging.info(f"Selected clients: {selected_clients}")
            
            w_locals, local_losses = [], []
            total_uplink = 0
            
            for idx in selected_clients:
                w, loss = self.train_client(idx, train_loaders[idx])
                w_locals.append(w)
                local_losses.append(loss)
                total_uplink += get_trainable_params_size_mb(self.global_model)

            self.aggregate(w_locals)
            
            total_downlink = get_trainable_params_size_mb(self.global_model) * len(selected_clients)
            
            metrics = self.evaluate(test_loader_global)
            round_duration = time.time() - round_start_time
            
            self.history.append({
                'round': round_num, 
                'avg_train_loss': np.mean(local_losses), 
                'round_duration_sec': round_duration,
                'total_uplink_mb': total_uplink,
                'total_downlink_mb': total_downlink,
                **metrics
            })
            
            logging.info(f"--- Round {round_num} Summary ---")
            logging.info(f"  Accuracy: {metrics['report']['accuracy']:.4f} | Weighted F1: {metrics['report']['weighted avg']['f1-score']:.4f} | MCC: {metrics['mcc']:.4f}")
            logging.info(f"  Round Duration: {round_duration:.2f}s | Uplink: {total_uplink:.2f}MB | Downlink: {total_downlink:.2f}MB")

    def save_results(self, output_dir="results_multiclass_fl_baseline"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.config.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.history: return

        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for k, v in asdict(self.config).items(): writer.writerow([f'# {k}', v])
            writer.writerow([])

            header = [
                'round', 'avg_train_loss', 'eval_loss', 'round_duration_sec', 
                'accuracy', 'mcc', 'weighted_avg_f1-score',
                'total_uplink_mb', 'total_downlink_mb'
            ]
            for i in range(self.config.num_classes):
                header.extend([f'class_{i}_precision', f'class_{i}_recall', f'class_{i}_f1-score', f'class_{i}_support'])
            writer.writerow(header)

            for record in self.history:
                report = record['report']
                row = [
                    record.get('round'), record.get('avg_train_loss'), record.get('eval_loss'), record.get('round_duration_sec'),
                    report.get('accuracy'), record.get('mcc'), report.get('weighted avg', {}).get('f1-score'),
                    record.get('total_uplink_mb'), record.get('total_downlink_mb')
                ]
                for i in range(self.config.num_classes):
                    class_metrics = report.get(f'Class {i}', {})
                    row.extend([class_metrics.get('precision'), class_metrics.get('recall'), class_metrics.get('f1-score'), class_metrics.get('support')])
                writer.writerow(row)
        logging.info(f"Results saved to: {os.path.join(output_dir, filename)}")

def main():
    config = ExperimentConfig()
    
    if not all(os.path.exists(f) for f in ['processed_data/train_data.jsonl', 'processed_data/test_data.jsonl']):
        logging.error("Error: Data files not found.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except OSError:
        logging.error(f"Error: Unable to load tokenizer from local path '{config.model_name}'.")
        return

    dataset_train = FT_Dataset('processed_data/train_data.jsonl', config.batch_size, config.max_seq_length, tokenizer)
    dataset_test = FT_Dataset('processed_data/test_data.jsonl', config.batch_size, config.max_seq_length, tokenizer)
        
    trainer = FLTrainer(config)
    trainer.run(dataset_train, dataset_test)
    trainer.save_results()
    logging.info("\nFL Baseline Experiment completed!")

if __name__ == "__main__":
    main()
