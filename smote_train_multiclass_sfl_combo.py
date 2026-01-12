# train_multiclass_sfl_final_fixed_smote.py

import torch
import torch.nn as nn
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
from sklearn.cluster import KMeans
import logging
import time
import csv
import random
import copy
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

from data_utils import FT_Dataset


@dataclass
class ExperimentConfig:
    # --- [推荐修改] ---
    exp_name: str = "multiclass_clustered_ga_splitfed_smote"
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
    aggregation_method: str = "FedDW" 
    num_data_clusters: int = 3
    clustering_interval: int = 5
    ga_population_size: int = 20
    ga_generations: int = 10
    ga_mutation_rate: float = 0.1
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


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
    def __init__(self, encoder_layers, classifier, attention_mask_getter, num_classes: int):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.classifier = classifier
        self.get_extended_attention_mask = attention_mask_getter
        self.num_classes = num_classes

    def forward(self, hidden_states, attention_mask, device, labels=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, hidden_states.size()[:-1], device)
        for layer in self.encoder_layers: hidden_states = layer(hidden_states, extended_attention_mask)[0]
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
       
            loss_fct = nn.CrossEntropyLoss() 
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            
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


class SFLTrainer:
    def __init__(self, config: ExperimentConfig):
      
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config.seed)
        
        try:
            base_model_full = RobertaForSequenceClassification.from_pretrained(
                config.model_name, 
                num_labels=config.num_classes,
                ignore_mismatched_sizes=True
            )
        except OSError:
            logging.error(f"Error: Unable to retrieve from local path '{config.model_name}' 加载模型。")
            raise

        peft_config_full = LoraConfig(task_type=TaskType.SEQ_CLS, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=config.target_modules, modules_to_save=["classifier"], bias="none")
        self.net_glob_full = get_peft_model(base_model_full, peft_config_full).to(self.device)
        self.history = []
        
        self.client_resource_profiles = {
            'high': {'split_layer': 8, 'weight_factor': 1.2},
            'medium': {'split_layer': 6, 'weight_factor': 1.0},
            'low': {'split_layer': 4, 'weight_factor': 0.8}
        }
        self.client_profiles = [random.choice(list(self.client_resource_profiles.keys())) for _ in range(config.num_clients)]
        self.client_lora_weights = [None] * config.num_clients 
        self.data_cluster_labels = [-1] * config.num_clients
        self.client_data_sizes = []
        self.clustered_clients_by_resource = {}
        
        logging.info(f"Use equipment: {self.device}, Experimental configuration: {asdict(config)}")

    def set_seed(self, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

   
    def aggregate(self, w_locals, client_weights=None):
        if not w_locals: return
        
        if client_weights is None or self.config.aggregation_method == 'FedAvg':
            client_weights = [1.0 / len(w_locals)] * len(w_locals)
        
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            if w_avg[k].dtype.is_floating_point: w_avg[k] *= client_weights[0]

        for i in range(1, len(w_locals)):
            for k in w_avg.keys():
                if w_avg[k].dtype.is_floating_point: w_avg[k] += w_locals[i][k] * client_weights[i]
        
        self.net_glob_full.load_state_dict(w_avg)

    def train_client(self, client_idx, local_net, server_model_for_client, train_loader):
        optimizer = AdamW(list(local_net.parameters()) + list(server_model_for_client.parameters()), lr=self.config.lr)
        total_loss, num_batches, comm_stats = 0, 0, {'uplink_sfl': 0, 'downlink_sfl': 0}
        local_net.train(); server_model_for_client.train()
        
        for epoch in range(self.config.local_epochs):
            pbar = tqdm(train_loader, desc=f"Client {client_idx} (Epoch {epoch+1}/{self.config.local_epochs})", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                fx = local_net(batch['input_ids'], batch['attention_mask'], self.device)
                comm_stats['uplink_sfl'] += get_tensor_size_mb(fx)
                client_fx = fx.clone().detach().requires_grad_(True)
                _, loss = server_model_for_client(client_fx, batch['attention_mask'], self.device, batch['labels'])
                if loss is None: continue
                loss.backward()
                if client_fx.grad is not None:
                    comm_stats['downlink_sfl'] += get_tensor_size_mb(client_fx.grad)
                    fx.backward(client_fx.grad)
                optimizer.step()
                total_loss += loss.item(); num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        return (total_loss / num_batches if num_batches > 0 else 0), comm_stats
    
    def evaluate(self, test_loader, split_layer_for_eval=6):
        self.net_glob_full.eval()
        full_roberta_model = self.net_glob_full.base_model.model.roberta
        client_net = ClientModel(full_roberta_model.embeddings, nn.ModuleList(full_roberta_model.encoder.layer[:split_layer_for_eval]), self.net_glob_full.get_extended_attention_mask).to(self.device).eval()
        server_net = ServerModel(nn.ModuleList(full_roberta_model.encoder.layer[split_layer_for_eval:]), self.net_glob_full.classifier, self.net_glob_full.get_extended_attention_mask, self.config.num_classes).to(self.device).eval()
        all_preds, all_labels, total_eval_loss = [], [], 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                fx = client_net(batch['input_ids'], batch['attention_mask'], self.device)
                logits, loss = server_net(fx, batch['attention_mask'], self.device, batch['labels'])
                if loss is not None: total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(batch['labels'].cpu().numpy())
        report_dict = classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(self.config.num_classes)], zero_division=0, output_dict=True)
        mcc = matthews_corrcoef(all_labels, all_preds)
        logging.info(f"\nClassification Assessment Report:\n{classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(self.config.num_classes)], zero_division=0)}")
        return {'eval_loss': total_eval_loss / len(test_loader) if len(test_loader) > 0 else 0, 'mcc': mcc, 'report': report_dict }

    def _extract_lora_weights_vector(self, model_state_dict):
        lora_params = {k: v for k, v in model_state_dict.items() if ('lora_A' in k or 'lora_B' in k or 'classifier' in k) and 'original_module' not in k}
        if not lora_params: return None
        return torch.cat([lora_params[k].view(-1) for k in sorted(lora_params.keys())]).cpu().numpy()

    def perform_double_clustering(self):
        logging.info("Implement a dual clustering strategy...")
        client_features, available_clients = [], []
        for i, weight in enumerate(self.client_lora_weights):
            if weight is not None: client_features.append(weight); available_clients.append(i)
        if len(available_clients) < self.config.num_data_clusters:
            logging.warning(f"Number of available clients({len(available_clients)})Insufficient for clustering, skip.")
            self.data_cluster_labels = [0] * self.config.num_clients
            return
        kmeans = KMeans(n_clusters=self.config.num_data_clusters, random_state=self.config.seed, n_init='auto').fit(np.array(client_features))
        for i, client_idx in enumerate(available_clients): self.data_cluster_labels[client_idx] = kmeans.labels_[i]
        logging.info(f"Data clustering results: {list(zip(available_clients, kmeans.labels_))}")
        self.clustered_clients_by_resource = {c_id: {res: [] for res in self.client_resource_profiles} for c_id in range(self.config.num_data_clusters)}
        for i in range(self.config.num_clients):
            if self.data_cluster_labels[i] != -1: self.clustered_clients_by_resource[self.data_cluster_labels[i]][self.client_profiles[i]].append(i)
        logging.info("Internal resource grouping completed.")

    def genetic_algorithm_client_selection(self, candidates: List[int], num_to_select: int) -> List[int]:
        if not candidates or num_to_select <= 0: return []
        if len(candidates) <= num_to_select:
            logging.info(f"GA: Number of candidates ({len(candidates)}) Less than or equal to the number of selections ({num_to_select})，Return all candidates directly: {candidates}")
            return candidates
        num_to_select = min(num_to_select, len(candidates))
        def fitness(individual):
            selected = [candidates[i] for i, gene in enumerate(individual) if gene == 1]
            if not selected: return 0.0
            data_size = sum(self.client_data_sizes[i] for i in selected)
            res_factor = sum(self.client_resource_profiles[self.client_profiles[i]]['weight_factor'] for i in selected)
            return (data_size * res_factor) - abs(len(selected) - num_to_select) * 0.5
        population = []
        for _ in range(self.config.ga_population_size):
            individual = [0] * len(candidates)
            for i in random.sample(range(len(candidates)), num_to_select): individual[i] = 1
            population.append(individual)
        best_individual = population[0]
        for _ in range(self.config.ga_generations):
            fitness_scores = [fitness(ind) for ind in population]
            best_individual = population[np.argmax(fitness_scores)]
            total_fit = sum(max(0, f) for f in fitness_scores)
            if total_fit == 0: parents = random.choices(population, k=self.config.ga_population_size)
            else: parents = random.choices(population, weights=[max(0, f) / total_fit for f in fitness_scores], k=self.config.ga_population_size)
            next_pop = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[i+1 if i+1 < len(parents) else 0]
                pt = random.randint(1, len(p1) - 1)
                c1, c2 = p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
                next_pop.extend([c1, c2])
            for ind in next_pop:
                if random.random() < self.config.ga_mutation_rate:
                    i = random.randrange(len(ind)); ind[i] = 1 - ind[i]
                num_ones = sum(ind)
                if num_ones > num_to_select:
                    ones = [i for i, g in enumerate(ind) if g == 1]
                    for i in random.sample(ones, num_ones - num_to_select): ind[i] = 0
                elif num_ones < num_to_select:
                    zeros = [i for i, g in enumerate(ind) if g == 0]
                    for i in random.sample(zeros, min(len(zeros), num_to_select-num_ones)): ind[i] = 1
            population = next_pop[:self.config.ga_population_size]
        final_selection = [candidates[i] for i, gene in enumerate(best_individual) if gene == 1]
        return sorted(final_selection)
        
    def run(self, dataset_train, dataset_test):
      
        dict_users_indices = np.array_split(range(len(dataset_train)), self.config.num_clients)
        self.client_data_sizes = [len(idxs) for idxs in dict_users_indices]
        train_loaders = [DataLoader(DatasetSplit(dataset_train, list(idxs)), self.config.batch_size, shuffle=True) for idxs in dict_users_indices]
        test_loader_global = DataLoader(dataset_test, self.config.batch_size, shuffle=False)
        logging.info(f"Start training {self.config.rounds} rounds...")
        self.data_cluster_labels = [0] * self.config.num_clients
        self.perform_double_clustering()
        for round_num in range(1, self.config.rounds + 1):
            round_start_time = time.time()
            logging.info(f"\n--- Round {round_num}/{self.config.rounds} ---")
            if round_num > 1 and (round_num % self.config.clustering_interval == 0): self.perform_double_clustering()
            clients_by_cluster = {cid: [] for cid in range(self.config.num_data_clusters)}
            for i, cid in enumerate(self.data_cluster_labels):
                if cid != -1: clients_by_cluster[cid].append(i)
            active_clusters = [cid for cid, clist in clients_by_cluster.items() if clist]
            if not active_clusters:
                logging.warning("No data clusters available, skip this round."); continue
            selected_clients = []
            num_per_cluster = max(1, self.config.clients_per_round // len(active_clusters))
            for cid in active_clusters:
                candidates = clients_by_cluster[cid]
                if candidates:
                    num_to_get = min(num_per_cluster, len(candidates))
                    selected_clients.extend(self.genetic_algorithm_client_selection(candidates, num_to_get))
            selected_clients = sorted(list(set(selected_clients)))
            if len(selected_clients) < self.config.clients_per_round:
                all_possible = [c for clist in clients_by_cluster.values() for c in clist]
                needed = self.config.clients_per_round - len(selected_clients)
                potential_adds = list(set(all_possible) - set(selected_clients))
                selected_clients.extend(random.sample(potential_adds, min(len(potential_adds), needed)))
            if not selected_clients:
                logging.warning("If no client is selected, skip this round."); continue
            logging.info(f"The client ultimately selected in this round: {selected_clients}")
            w_locals, local_losses, round_comm_stats = [], [], {'total_uplink_mb': 0, 'total_downlink_mb': 0}
            for idx in selected_clients:
                split_layer = self.client_resource_profiles[self.client_profiles[idx]]['split_layer']
                local_model = copy.deepcopy(self.net_glob_full)
                roberta = local_model.base_model.model.roberta
                local_net = ClientModel(roberta.embeddings, nn.ModuleList(roberta.encoder.layer[:split_layer]), local_model.get_extended_attention_mask).to(self.device)
                server_net = ServerModel(nn.ModuleList(roberta.encoder.layer[split_layer:]), local_model.classifier, local_model.get_extended_attention_mask, self.config.num_classes).to(self.device)
                loss, comms = self.train_client(idx, local_net, server_net, train_loaders[idx])
                round_comm_stats['total_uplink_mb'] += comms['uplink_sfl'] + get_trainable_params_size_mb(local_model)
                round_comm_stats['total_downlink_mb'] += comms['downlink_sfl']
                local_losses.append(loss); w_locals.append(local_model.state_dict())
                self.client_lora_weights[idx] = self._extract_lora_weights_vector(local_model.state_dict())
            weights = None
            if self.config.aggregation_method == 'FedDW' and selected_clients:
                total_sl = sum(self.client_resource_profiles[self.client_profiles[i]]['split_layer'] for i in selected_clients)
                weights = [(self.client_resource_profiles[self.client_profiles[i]]['split_layer'] / total_sl) for i in selected_clients] if total_sl > 0 else None
            self.aggregate(w_locals, weights)
            round_comm_stats['total_downlink_mb'] += get_trainable_params_size_mb(self.net_glob_full) * len(selected_clients)
            metrics = self.evaluate(test_loader_global)
            round_duration = time.time() - round_start_time
            self.history.append({'round': round_num, 'avg_train_loss': np.mean(local_losses) if local_losses else 0, 'round_duration_sec': round_duration, **round_comm_stats, **metrics})
            logging.info(f"--- Round {round_num} Summarize ---")
            logging.info(f"  Global model evaluation - Accuracy: {metrics['report']['accuracy']:.4f}, F1 (Weighted): {metrics['report']['weighted avg']['f1-score']:.4f}, MCC: {metrics['mcc']:.4f}")
            logging.info(f"  This round takes time: {round_duration:.2f} 秒")

    def save_results(self, output_dir="results_multiclass_sfl_smote"): 
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.config.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.history:
            logging.warning("The history is empty, so the result cannot be saved.")
            return

        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for k, v in asdict(self.config).items(): writer.writerow([f'# {k}', v])
            writer.writerow([])

            header = ['round', 'avg_train_loss', 'eval_loss', 'round_duration_sec', 'accuracy', 'mcc', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1-score','weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1-score','total_uplink_mb', 'total_downlink_mb']
            for i in range(self.config.num_classes): header.extend([f'class_{i}_precision', f'class_{i}_recall', f'class_{i}_f1-score', f'class_{i}_support'])
            writer.writerow(header)

            for record in self.history:
                report = record['report']
                row = [
                    record.get('round'), record.get('avg_train_loss'), record.get('eval_loss'), record.get('round_duration_sec'),
                    report.get('accuracy'), record.get('mcc'),
                    report.get('macro avg', {}).get('precision'), report.get('macro avg', {}).get('recall'), report.get('macro avg', {}).get('f1-score'),
                    report.get('weighted avg', {}).get('precision'), report.get('weighted avg', {}).get('recall'), report.get('weighted avg', {}).get('f1-score'),
                    record.get('total_uplink_mb'), record.get('total_downlink_mb')
                ]
                for i in range(self.config.num_classes):
                    class_metrics = report.get(f'Class {i}', {})
                    row.extend([class_metrics.get('precision'), class_metrics.get('recall'), class_metrics.get('f1-score'), class_metrics.get('support')])
                writer.writerow(row)
        logging.info(f"Detailed evaluation results have been saved to: {os.path.join(output_dir, filename)}")

def main():
    config = ExperimentConfig()
 
    train_data_path = 'processed_data_SMOTE/train_data.jsonl'
    test_data_path = 'processed_data_SMOTE/test_data.jsonl'

    if not all(os.path.exists(f) for f in [train_data_path, test_data_path]):
        logging.error(f"Error: Data file processed by SMOTE not found. '{train_data_path}' 或 '{test_data_path}'。")
        logging.error("Please run the script `UNSW_NB15_..._processed_llm.py`, which contains SMOTE processing, first.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except OSError:
        logging.error(f"Error: Unable to load tokenizer from local path '{config.model_name}'.")
        return

    dataset_train = FT_Dataset(train_data_path, config.batch_size, config.max_seq_length, tokenizer)
    dataset_test = FT_Dataset(test_data_path, config.batch_size, config.max_seq_length, tokenizer)
        
    trainer = SFLTrainer(config)
    trainer.run(dataset_train, dataset_test)
    trainer.save_results()
    logging.info("\nExperiment completed!")

if __name__ == "__main__":
    main()
