# PP-DSFL-LoRA: A Privacy-Preserving Intrusion Detection Framework

This repository contains the official implementation, supplementary materials, and reproducibility details for the paper: "A Privacy-Preserving Intrusion Detection Framework for Heterogeneous Edge Devices Based on Dynamic Split Federated Learning and LoRA".

## 📌 Abstract
This work proposes a comprehensive framework that integrates Dynamic Split Federated Learning (SFL), Low-Rank Adaptation (LoRA), and SMOTE to address the trilemma of communication overhead, client-side burden, and system heterogeneity in federated Pre-trained Language Model (PLM) fine-tuning.

---

## 🛠 Experimental Environment

All experiments were conducted in a simulated heterogeneous federated learning environment.

* Server Side: Equipped with an NVIDIA V100 GPU and high-performance CPUs to simulate a cloud aggregation server with sufficient computational power.
* Client Side: Based on PyTorch multi-process simulation. We configured 10 clients as virtual edge devices with varying computational capabilities (High, Medium, and Low tiers), corresponding to different model split depths ($L_{split} \in \{2, 4, 8\}$).
* Software Dependencies:
    * Python 3.8+
    * PyTorch 1.12 (Deep Learning Framework)
    * Transformers 4.x (HuggingFace Model Loading)
    * PEFT (LoRA Implementation)
    * Scikit-learn (Data Processing & Evaluation)

---

## ⚙️ Hyperparameter Configuration

Due to page limits in the main paper, the detailed hyperparameter settings used for reproduction are listed below.

| Category | Parameter Name | Value / Description |
| :--- | :--- | :--- |
| Model Config | Backbone Model | `roberta-base` (12-layer, 768-hidden) |
| | LoRA Rank ($r$) | 8 |
| | LoRA Alpha ($\alpha$) | 16 |
| | LoRA Dropout | 0.1 |
| | Target Modules | Query, Value Projection Matrices |
| Federated Training | Communication Rounds | 100 |
| | Total Clients ($K$) | 10 |
| | Active Clients per Round | 4 (Selected via Genetic Algorithm) |
| | Local Epochs | 1 |
| | Batch Size | 16 |
| | Learning Rate | $2 \times 10^{-5}$ (AdamW Optimizer) |
| | Max Sequence Length | 128 |
| Strategy Config | Aggregation Algorithm | FedDW (Dynamic Weighting) |
| | GA Population Size | 20 |

---

## 🔍 Detailed Preprocessing Pipeline

To bridge the modality gap between structured network traffic data (UNSW-NB15) and the input requirements of PLMs, and to address severe class imbalance, we implemented a strict 4-stage preprocessing pipeline:

### Stage 1: Feature Cleaning and Numericalization
* Cleaning: Removed identifier columns irrelevant to classification (e.g., `id`).
* Categorical Features: Applied One-Hot Encoding for `proto`, `service`, and `state`.
* Numerical Features: Applied Z-score Standardization ($x' = (x - \mu) / \sigma$) to eliminate scale discrepancies.
* Imputation: Missing or infinite values were replaced with the mean of the corresponding column.

### Stage 2: Strategic Resampling via SMOTE
* Strategy: Applied exclusively to the training set (70% split). The testing set (30%) retains its original imbalanced distribution to reflect real-world scenarios.
* Method: The Synthetic Minority Over-sampling Technique (SMOTE) interpolates minority class samples (e.g., "Worms", "Shellcode") in the feature space, forcing a balanced distribution across all 10 categories in the training data.

### Stage 3: Semantic Serialization (Prompt Engineering)
To leverage RoBERTa's semantic understanding capabilities, numerical features are transformed into natural language descriptions.
* Threshold Filtering: Only significant features with a standardized absolute value > 0.1 are retained to filter noise.
* Template Format:
    > "Network flow features: [Feature-Name] is [Value]; ..."
* Length Constraint: If a single sample contains more than 25 significant features, 25 are randomly sampled to accommodate the model's maximum sequence length constraints.

### Stage 4: Tokenization
* Processed using `RobertaTokenizerFast`.
* Max Sequence Length: Set to 128.
* Outputs `input-ids` and `attention-mask` tensors for federated training.

---

## 📖 Supplementary Discussion

This section provides extended analysis complementing the "Discussion" section of the main paper, delving into the mechanisms behind the experimental observations.

### 1. Performance Volatility: A Trade-off for Inclusivity
In the training dynamics presented in the paper (specifically Macro metrics), distinct "sawtooth" fluctuations were observed. This is not an algorithmic defect but a structural compromise of Dynamic Split Federated Learning (Dynamic SFL).
* Cause: The server receives intermediate features (Smashed Data) from clients with varying split depths ($L_{split} \in \{2, 4, 8\}$). This means the server-side model must adapt to feature streams with inconsistent semantic abstraction levels, leading to gradient oscillation.
* Benefit: This short-term instability is exchanged for system inclusivity. Enforcing a static split point would cause low-power devices ("stragglers") to drop out due to timeouts. Our method ensures data from all heterogeneous devices participates in aggregation, enhancing robust generalization.

### 2. Deep Analysis of Class Imbalance: The Boundaries of SMOTE
Experiments show that while Weighted F1 reached 0.95, Macro F1 hovered around 0.55.
* Role & Limit of SMOTE: SMOTE successfully prevented the model from "complete collapse" on minority classes (avoiding 0 F1 scores). However, for extremely scarce attacks with complex patterns (e.g., Worms), simple linear interpolation may introduce noise or overlapping decision boundaries.
* Future Direction: This suggests that future work needs to integrate Focal Loss or GAN-based advanced data augmentation techniques to further enhance sensitivity to ultra-minority classes.

### 3. Limitations
* Privacy of Intermediate Data: While raw data is not transmitted, "Smashed Data" (intermediate activations) is theoretically vulnerable to Model Inversion Attacks. We did not integrate Differential Privacy (DP) in this version to prioritize computational efficiency.
* Semi-Honest Assumption: The framework operates under the assumption that the server is "semi-honest" (honest-but-curious).
* Simulated Network: Communication overhead analysis is based on theoretical packet sizes and does not yet account for physical layer effects like packet loss or jitter in real Industrial IoT environments.
