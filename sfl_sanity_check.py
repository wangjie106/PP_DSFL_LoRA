# file: sfl_sanity_check.py
# 用途：分割联邦学习（SFL）主训练脚本，使用 LoRA 微调 RoBERTa 模型
# 改进版：增强训练强度 + 类别权重 + 详细评估

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import time
import csv
import random
from collections import OrderedDict, Counter
import copy
import os

from data_utils import FT_Dataset, get_tokenizer

# ============================== 配置 ==============================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

# ==================== 模式切换（增强版）====================
MODE = 'QUICK'  # 'QUICK' 用于快速逻辑检查, 'NORMAL' 用于完整训练

if MODE == 'QUICK':
    logging.warning("<<<<< 正在以 SFL QUICK 模式运行 >>>>>")
    # ⬇️⬇️ 增强训练：10轮，每轮5个epoch ⬇️⬇️
    ROUNDS, NUM_CLIENTS, CLIENTS_PER_ROUND, LOCAL_EPOCHS, DEBUG_DATA_SIZE, SPLIT_LAYER = 10, 4, 2, 5, 2000, 4
else:
    logging.info(">>>>> 正在以 SFL NORMAL 模式运行 <<<<<")
    ROUNDS, NUM_CLIENTS, CLIENTS_PER_ROUND, LOCAL_EPOCHS, DEBUG_DATA_SIZE, SPLIT_LAYER = 50, 20, 5, 5, None, 4

# ==================== 使用本地模型 ====================
MODEL_NAME = './roberta-base-local'
MODEL_NAME_ON_HUB = 'roberta-base'

# ⬇️⬇️ 提高学习率到 1e-4（原来是 2e-5）⬇️⬇️
LR, BATCH_SIZE, MAX_SEQ_LENGTH = 1e-4, 16, 128
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.1
RESULTS_FILENAME = f"SFL_{MODE.lower()}_R{ROUNDS}_C{NUM_CLIENTS}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# ============================== SFL 模型定义 ==============================
class ClientModelSFL(nn.Module):
    """客户端模型：包含嵌入层和前 N 层编码器"""
    def __init__(self, full_model, split_layer):
        super().__init__()
        base_model = full_model.base_model.model if hasattr(full_model, 'base_model') else full_model
        self.embeddings = base_model.roberta.embeddings
        self.encoder_layers = base_model.roberta.encoder.layer[:split_layer]
    
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None: 
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states = self.embeddings(input_ids=input_ids)
        for layer in self.encoder_layers: 
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
        return hidden_states, extended_attention_mask


class ServerModelSFL(nn.Module):
    """服务器端模型：包含剩余编码器层和分类头（支持类别权重）"""
    def __init__(self, full_model, split_layer):
        super().__init__()
        base_model = full_model.base_model.model if hasattr(full_model, 'base_model') else full_model
        self.encoder_layers = base_model.roberta.encoder.layer[split_layer:]
        self.classifier = base_model.classifier
    
    def forward(self, hidden_states, attention_mask, labels=None, class_weights=None):
        """
        前向传播，支持类别权重
        
        Args:
            hidden_states: 隐藏状态
            attention_mask: 注意力掩码
            labels: 标签
            class_weights: 类别权重（用于处理不平衡数据）
        """
        for layer in self.encoder_layers: 
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            # ⬇️⬇️ 使用类别权重 ⬇️⬇️
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return logits, loss


def create_and_split_model(split_layer, model_path):
    """创建并分割模型"""
    logging.info(f"正在从路径 '{model_path}' 加载模型...")
    logging.info(f"分割层设置为: {split_layer}")
    
    full_model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT, 
        target_modules=["query", "key", "value"], 
        modules_to_save=["classifier"]
    )
    full_model_peft = get_peft_model(full_model, peft_config)
    
    logging.info("--- 应用 PEFT 后的完整模型可训练参数 ---")
    full_model_peft.print_trainable_parameters()
    
    client_model = ClientModelSFL(full_model_peft, split_layer)
    server_model = ServerModelSFL(full_model_peft, split_layer)
    
    logging.info("✅ 模型分割完成。")
    return client_model.to(device), server_model.to(device)


# ============================== 权重处理与聚合 ==============================
def get_trainable_state_dict(model: nn.Module) -> dict:
    """获取模型的可训练参数"""
    return {k: v.clone() for k, v in model.state_dict().items() if v.requires_grad}


def set_trainable_state_dict(model: nn.Module, state_dict: dict):
    """设置模型的可训练参数"""
    model.load_state_dict(state_dict, strict=False)


def FedAvg(w_list):
    """联邦平均算法"""
    if not w_list: 
        return None
    aggregated_weights = OrderedDict()
    for key in w_list[0].keys():
        aggregated_weights[key] = torch.stack([w[key] for w in w_list]).mean(dim=0)
    return aggregated_weights


# ============================== SFL 训练与评估 ==============================
def client_sfl_train(client_id, client_model, server_model, train_loader, current_lr, 
                     class_weights=None, is_first_client_in_round=False):
    """
    客户端 SFL 训练函数（支持类别权重）
    
    Args:
        client_id: 客户端ID
        client_model: 客户端模型
        server_model: 服务器端模型
        train_loader: 训练数据加载器
        current_lr: 当前学习率
        class_weights: 类别权重（用于处理不平衡）
        is_first_client_in_round: 是否是本轮第一个客户端
    """
    client_model.train()
    server_model.train()
    
    trainable_client_params = [p for p in client_model.parameters() if p.requires_grad]
    trainable_server_params = [p for p in server_model.parameters() if p.requires_grad]
    
    # 调试信息（只在第一轮第一个客户端打印）
    if is_first_client_in_round:
        print("\n" + "="*80)
        print(f"!!! DEBUG: 第一个客户端 (ID: {client_id}) 的训练前检查 !!!")
        print(f"    客户端模型找到的可训练参数组数量: {len(trainable_client_params)}")
        print(f"    服务器模型找到的可训练参数组数量: {len(trainable_server_params)}")
        
        client_param_count = sum(p.numel() for p in trainable_client_params)
        server_param_count = sum(p.numel() for p in trainable_server_params)
        
        print(f"    客户端模型可训练参数总数: {client_param_count:,}")
        print(f"    服务器模型可训练参数总数: {server_param_count:,}")

        if not trainable_client_params:
            print("    !!!!!! 警告: 客户端优化器没有需要优化的参数 !!!!!!")
        else:
            print("    ✅ OK: 客户端优化器有参数可优化。")

        if not trainable_server_params:
            print("    !!!!!! 警告: 服务器端优化器没有需要优化的参数 !!!!!!")
        else:
            print("    ✅ OK: 服务器端优化器有参数可优化。")
        
        if class_weights is not None:
            print(f"    ✅ 使用类别权重: {class_weights.cpu().numpy()}")
        
        print("="*80 + "\n")

    # 如果没有任何可训练参数，训练是无意义的
    if not trainable_client_params and not trainable_server_params:
        logging.error(f"客户端 {client_id}: 致命错误 - 客户端和服务器端均未找到可训练参数。跳过训练。")
        return get_trainable_state_dict(client_model), 0.0

    optimizer_client = AdamW(trainable_client_params, lr=current_lr) if trainable_client_params else None
    optimizer_server = AdamW(trainable_server_params, lr=current_lr) if trainable_server_params else None

    total_loss, num_batches = 0, 0
    for epoch in range(LOCAL_EPOCHS):
        for batch in train_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            labels = batch_on_device['labels']
            
            if optimizer_client: optimizer_client.zero_grad()
            if optimizer_server: optimizer_server.zero_grad()

            smashed_data, extended_attention_mask = client_model(
                input_ids=batch_on_device['input_ids'], 
                attention_mask=batch_on_device['attention_mask']
            )
            smashed_data_server = smashed_data.detach().requires_grad_(True)
            
            # ⬇️⬇️ 传入类别权重 ⬇️⬇️
            logits, loss = server_model(smashed_data_server, extended_attention_mask, labels, 
                                       class_weights=class_weights)
            
            if loss is None: 
                continue
            
            loss.backward()
            
            # 只有在 smashed_data 上有梯度时才反向传播到客户端
            if smashed_data_server.grad is not None:
                smashed_data.backward(smashed_data_server.grad)
            
            if optimizer_client: optimizer_client.step()
            if optimizer_server: optimizer_server.step()
            
            total_loss += loss.item()
            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logging.info(f"  客户端 {client_id} (LR={current_lr:.2e}) 训练完成, 平均损失: {avg_loss:.4f}")

    return get_trainable_state_dict(client_model), avg_loss


def evaluate_sfl(client_model, server_model, dataloader):
    """
    评估 SFL 模型（增强版，显示详细预测分布）
    """
    client_model.eval()
    server_model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估SFL模型", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            smashed_data, extended_attention_mask = client_model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            logits, _ = server_model(smashed_data, extended_attention_mask)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # ========== 详细预测分析 ==========
    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)
    
    logging.info(f"  📊 预测分布: 类别0={pred_counts.get(0, 0):>6}, 类别1={pred_counts.get(1, 0):>6}")
    logging.info(f"  📊 真实分布: 类别0={label_counts[0]:>6}, 类别1={label_counts[1]:>6}")
    
    # 计算每个类别的准确率
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    
    correct_0 = sum((all_preds_np == 0) & (all_labels_np == 0))
    correct_1 = sum((all_preds_np == 1) & (all_labels_np == 1))
    total_0 = label_counts[0]
    total_1 = label_counts[1]
    
    acc_0 = correct_0 / total_0 * 100 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 * 100 if total_1 > 0 else 0
    
    logging.info(f"  📊 类别0准确率: {correct_0:>6}/{total_0:>6} = {acc_0:>5.2f}%")
    logging.info(f"  📊 类别1准确率: {correct_1:>6}/{total_1:>6} = {acc_1:>5.2f}%")
    # ===================================
    
    metrics = {'accuracy': accuracy_score(all_labels, all_preds)}
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    metrics.update({'precision': precision, 'recall': recall, 'f1': f1})
    return metrics


# ============================== 其他工具函数 ==============================
def split_data_for_clients(train_dataset, num_clients):
    """将数据集分割给多个客户端"""
    client_datasets, all_indices = [], list(range(len(train_dataset)))
    random.shuffle(all_indices)
    for i in range(num_clients):
        subset_indices = all_indices[i::num_clients]
        client_datasets.append(Subset(train_dataset, subset_indices))
    return client_datasets


def save_results_to_csv(all_results, filename, config):
    """保存结果到 CSV 文件"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['# Configuration'])
            for k, v in config.items():
                writer.writerow([f'# {k}', v])
            writer.writerow([])
            
            fieldnames = ['round', 'loss', 'accuracy', 'f1', 'precision', 'recall']
            writer.writerow(fieldnames)
            
            for result in all_results:
                writer.writerow([
                    result['round'], 
                    f"{result['loss']:.4f}", 
                    f"{result['accuracy']:.4f}", 
                    f"{result['f1']:.4f}", 
                    f"{result['precision']:.4f}", 
                    f"{result['recall']:.4f}"
                ])
        logging.info(f"✅ SFL评估结果已成功保存到文件: {filename}")
    except Exception as e:
        logging.error(f"保存SFL结果文件失败: {e}")


def print_trainable_parameters_manually(model: nn.Module):
    """手动打印可训练参数统计"""
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
        return

    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_param if all_param > 0 else 0
    logging.info(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {percentage:.4f}%")


def verify_local_model(model_path):
    """验证本地模型文件是否完整"""
    print("\n" + "="*70)
    print("🔍 正在验证本地模型...")
    
    if not os.path.exists(model_path):
        logging.error(f"❌ 本地模型路径不存在: {model_path}")
        logging.error("请先运行 download_roberta.py 下载模型")
        return False
    
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'vocab.json', 'merges.txt']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logging.info(f"  ✅ {file:<25} ({size_mb:>8.2f} MB)")
    
    if missing_files:
        logging.error(f"❌ 缺失关键文件: {missing_files}")
        return False
    
    logging.info(f"✅ 本地模型验证通过: {model_path}")
    print("="*70 + "\n")
    return True


# ============================== 主函数 ==============================
def main():
    print("="*70)
    print("分割联邦学习 (SFL): 使用 LoRA 微调 RoBERTa 模型")
    print(f"当前模式: {MODE}")
    print(f"改进: 增强训练强度 + 类别权重 + 详细评估")
    print("="*70 + "\n")
    
    # ==================== 验证本地模型 ====================
    if not verify_local_model(MODEL_NAME):
        logging.error("模型验证失败，程序退出。")
        return
    
    # ==================== 自动数据检查与处理 ====================
    logging.info("步骤 0: 检查数据文件...")
    
    train_file = 'processed_data/train_data.jsonl'
    test_file = 'processed_data/test_data.jsonl'
    
    data_exists = os.path.exists(train_file) and os.path.exists(test_file)
    
    if not data_exists:
        logging.warning("⚠️  未找到已处理的数据文件，尝试自动处理...")
        
        try:
            import UNSW_NB15_processed_llm as data_processor
            
            if MODE == 'QUICK':
                debug_rows = 10000
                logging.info(f"QUICK 模式：使用 {debug_rows} 行数据进行快速测试")
            else:
                debug_rows = None
                logging.info("NORMAL 模式：使用完整数据集")
            
            success = data_processor.check_and_prepare_data(
                data_path='data/UNSW-NB15.csv',
                debug_rows=debug_rows,
                force_reprocess=False
            )
            
            if not success:
                logging.error("❌ 数据处理失败，程序退出。")
                return
                
        except ImportError:
            logging.error("❌ 无法导入数据处理模块 'UNSW_NB15_processed_llm.py'")
            return
        except Exception as e:
            logging.error(f"❌ 数据处理过程中发生错误: {e}")
            return
    else:
        logging.info(f"✅ 发现已处理的数据文件")
    
    # ==================== 配置信息 ====================
    training_config = {
        "Mode": MODE,
        "Framework": "Split Federated Learning",
        "Model": MODEL_NAME_ON_HUB,
        "Model_Path": MODEL_NAME,
        "Split_Layer": SPLIT_LAYER,
        "LoRA_Rank_(r)": LORA_R,
        "LoRA_Alpha_(alpha)": LORA_ALPHA,
        "Communication_Rounds": ROUNDS,
        "Clients_per_Round": CLIENTS_PER_ROUND,
        "Total_Clients": NUM_CLIENTS,
        "Local_Epochs": LOCAL_EPOCHS,
        "Learning_Rate_(LR)": LR
    }

    # ==================== 加载数据 ====================
    logging.info("步骤 1: 加载数据并划分...")
    
    tokenizer = get_tokenizer(MODEL_NAME)
    
    full_train_dataset = FT_Dataset(train_file, BATCH_SIZE, MAX_SEQ_LENGTH, tokenizer)
    test_dataset = FT_Dataset(test_file, BATCH_SIZE, MAX_SEQ_LENGTH, tokenizer)
    
    # ========== 计算类别权重 ==========
    logging.info("正在计算类别权重...")
    all_labels = []
    for i in range(len(full_train_dataset)):
        all_labels.append(full_train_dataset[i]['labels'].item())
    
    label_counts = Counter(all_labels)
    logging.info(f"训练集标签分布: 类别0={label_counts[0]}, 类别1={label_counts[1]}")
    
    # 计算类别权重（反比于类别频率）
    total_samples = len(all_labels)
    class_weights = torch.tensor([
        total_samples / (2 * label_counts[0]),  # 类别0的权重
        total_samples / (2 * label_counts[1])   # 类别1的权重
    ], dtype=torch.float32).to(device)
    
    logging.info(f"类别权重: 类别0={class_weights[0]:.3f}, 类别1={class_weights[1]:.3f}")
    logging.info("（权重越大，该类别在损失计算中越重要）")
    # ====================================
    
    if DEBUG_DATA_SIZE is not None and DEBUG_DATA_SIZE < len(full_train_dataset):
        indices = torch.randperm(len(full_train_dataset))[:DEBUG_DATA_SIZE]
        train_subset = Subset(full_train_dataset, indices.tolist())
        client_datasets = split_data_for_clients(train_subset, NUM_CLIENTS)
    else:
        client_datasets = split_data_for_clients(full_train_dataset, NUM_CLIENTS)
    
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==================== 初始化模型 ====================
    logging.info("步骤 2: 初始化全局模型与全局调度器...")
    
    global_client_model, global_server_model = create_and_split_model(SPLIT_LAYER, MODEL_NAME)
    
    logging.info("--- (手动检查) 客户端可训练参数 ---")
    print_trainable_parameters_manually(global_client_model)
    logging.info("--- (手动检查) 服务器端可训练参数 ---")
    print_trainable_parameters_manually(global_server_model)

    dummy_optimizer = AdamW([torch.zeros(1)], lr=LR)
    global_scheduler = get_linear_schedule_with_warmup(
        dummy_optimizer, 
        num_warmup_steps=0,
        num_training_steps=ROUNDS
    )

    # ==================== 开始训练 ====================
    logging.info("步骤 3: 开始 SFL 训练...")
    all_round_results = []
    
    for round_num in range(1, ROUNDS + 1):
        logging.info(f"\n{'='*70}")
        logging.info(f"通信轮次 {round_num}/{ROUNDS}")
        logging.info("="*70)
        
        selected_client_ids = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        logging.info(f"本轮参与客户端: {selected_client_ids}")
        
        current_round_lr = global_scheduler.get_last_lr()[0]
        logging.info(f"当前学习率: {current_round_lr:.2e}")
        
        round_server_model = copy.deepcopy(global_server_model)
        
        local_client_weights, local_losses = [], []
        
        for i, client_id in enumerate(selected_client_ids):
            local_client_model = copy.deepcopy(global_client_model)
            
            # 只在全局的第一轮第一个客户端打印调试信息
            is_first = (i == 0 and round_num == 1)
            
            # ⬇️⬇️ 传入类别权重 ⬇️⬇️
            client_w, loss = client_sfl_train(
                client_id, 
                local_client_model, 
                round_server_model, 
                client_loaders[client_id], 
                current_round_lr,
                class_weights=class_weights,  # ⬅️ 关键：传入类别权重
                is_first_client_in_round=is_first
            )
            local_client_weights.append(client_w)
            local_losses.append(loss)
            
        # 聚合客户端权重
        global_client_weights = FedAvg(local_client_weights)
        if global_client_weights: 
            set_trainable_state_dict(global_client_model, global_client_weights)
        
        # 更新全局服务器模型
        final_round_server_weights = get_trainable_state_dict(round_server_model)
        set_trainable_state_dict(global_server_model, final_round_server_weights)
        
        logging.info(f"服务器聚合完成，全局模型已更新。")
        
        # 评估
        logging.info("正在评估...")
        metrics = evaluate_sfl(global_client_model, global_server_model, test_loader)
        avg_round_loss = np.mean(local_losses)
        round_result = {'round': round_num, 'loss': avg_round_loss, **metrics}
        all_round_results.append(round_result)
        
        global_scheduler.step()
        
        # 打印轮次总结
        print()
        logging.info(f"{'='*70}")
        logging.info(f"轮次 {round_num} 总结")
        logging.info("="*70)
        logging.info(f"  平均训练损失: {avg_round_loss:.4f}")
        logging.info(f"  测试集 准确率: {metrics['accuracy']:.4f}")
        logging.info(f"  测试集 F1 分数: {metrics['f1']:.4f}")
        logging.info(f"  测试集 精确率: {metrics['precision']:.4f}")
        logging.info(f"  测试集 召回率: {metrics['recall']:.4f}")
        logging.info("="*70)
        print()

    # ==================== 保存结果 ====================
    logging.info("步骤 4: 训练完成，保存结果...")
    save_results_to_csv(all_round_results, RESULTS_FILENAME, training_config)
    
    print("\n" + "="*70)
    logging.info("✅ SFL 训练完成！")
    logging.info(f"📊 结果已保存到: {RESULTS_FILENAME}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
