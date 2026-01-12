# file: UNSW_NB15_processed_llm_smote_excel_custom_folder.py
# 用途：处理 UNSW-NB15 数据，使用 SMOTE，转换为 JSONL，并生成 Excel 报告到自定义文件夹

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import json
import os
from collections import Counter
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_distribution_to_excel(train_dist, test_dist, label_map, output_path):
    """
    将训练集和测试集的数据分布情况保存到 Excel 文件中。
    """
    id_to_label = {v: k for k, v in label_map.items()}
    all_labels = sorted(label_map.values())

    data_for_df = []
    for label_id in all_labels:
        data_for_df.append({
            'Category': id_to_label[label_id],
            'Label ID': label_id,
            'Train Count': train_dist.get(label_id, 0),
            'Test Count': test_dist.get(label_id, 0)
        })

    df = pd.DataFrame(data_for_df)
    
    total_row = {
        'Category': 'TOTAL',
        'Label ID': '',
        'Train Count': df['Train Count'].sum(),
        'Test Count': df['Test Count'].sum()
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        logging.info(f"💾 Successfully saved data distribution report to {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save Excel file {output_path}. Error: {e}")


def convert_to_text_and_save(df, label_series, output_path, task_type='classification'):
    """
    将数值和分类数据转换为自然文本格式并保存
    """
    logging.info(f"Converting data to natural text format for task '{task_type}' and saving to {output_path}")
    
    lines = []
    prompt_template = "Network flow features: "
    label_series_list = label_series.tolist()

    for i in tqdm(range(len(df)), desc=f"Generating text for {os.path.basename(output_path)}"):
        row = df.iloc[i]
        feature_parts = []
        non_zero_features = row[row.abs() > 0.1]
        
        if len(non_zero_features) > 25:
            selected_features = non_zero_features.sample(n=25, random_state=i)
        else:
            selected_features = non_zero_features

        for feature, value in selected_features.items():
            feature_parts.append(f"{feature.replace('_', ' ')} is {value:.2f}")
            
        feature_str = "; ".join(feature_parts)
        text_content = prompt_template + feature_str

        if task_type == 'classification':
            record = {'text': text_content, 'label': int(label_series_list[i])}
            lines.append(json.dumps(record, ensure_ascii=False))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logging.info(f"Successfully saved {len(lines)} lines to {output_path}")


def process_and_save_data(filepath, output_folder, debug_rows=None, task_type='classification'):
    """
    主数据处理函数
    """
    logging.info(f"Starting data loading from: {filepath}")
    full_df = pd.read_csv(filepath, encoding='latin1', header=0, low_memory=False)
    
    # ... (基本预处理和标签处理部分保持不变) ...
    full_df.columns = [col.strip().lower() for col in full_df.columns]
    if 'id' in full_df.columns: full_df = full_df.drop('id', axis=1)

    label_col = 'label'
    if 'attack_cat' in full_df.columns:
        full_df['attack_cat'] = full_df['attack_cat'].str.strip()
        unique_categories = sorted(full_df['attack_cat'].unique())
        if 'Normal' in unique_categories:
            unique_categories.remove('Normal')
            unique_categories.insert(0, 'Normal')
        label_map = {cat: i for i, cat in enumerate(unique_categories)}
        num_classes = len(label_map)
        full_df[label_col] = full_df['attack_cat'].map(label_map)
        full_df = full_df.drop('attack_cat', axis=1)
        logging.info("="*60 + "\n📊 标签映射完成\n" + "="*60)
    else:
        logging.error("❌ 未找到 'attack_cat' 列。")
        return None

    # ... (处理混合类型、缺失值、Debug模式部分保持不变) ...
    for col in full_df.select_dtypes(include=['object']).columns:
        if col not in ['proto', 'service', 'state']:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df = full_df.dropna(subset=[label_col])
    
    if debug_rows:
        logging.info(f"Debug mode: Using {debug_rows} rows")
        _, df = train_test_split(full_df, train_size=debug_rows, stratify=full_df[label_col], random_state=42)
    else:
        df = full_df
        
    y = df.pop(label_col)
    X = df
    
    # ... (数值处理、编码、归一化部分保持不变) ...
    X = pd.get_dummies(X, columns=['proto', 'service', 'state'])
    for col in X.select_dtypes(include=np.number).columns:
        if X[col].isnull().any(): X[col].fillna(X[col].mean(), inplace=True)
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- [修改] ---
    # 创建指定的输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 保存 SMOTE 前的数据分布到 Excel
    save_distribution_to_excel(
        y_train.value_counts().to_dict(),
        y_test.value_counts().to_dict(),
        label_map,
        os.path.join(output_folder, 'data_distribution_before_smote.xlsx')
    )
    
    # 使用 SMOTE 处理训练集
    logging.info("="*60 + "\n⚖️ 开始处理类别不平衡问题 (SMOTE)...\n" + "="*60)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 保存 SMOTE 后的数据分布到 Excel
    save_distribution_to_excel(
        pd.Series(y_train_resampled).value_counts().to_dict(),
        y_test.value_counts().to_dict(),
        label_map,
        os.path.join(output_folder, 'data_distribution_after_smote.xlsx')
    )
    
    # --- [修改] ---
    # 生成并保存文本文件到指定文件夹
    convert_to_text_and_save(X_train_resampled, pd.Series(y_train_resampled), os.path.join(output_folder, 'train_data.jsonl'), task_type)
    convert_to_text_and_save(X_test, y_test, os.path.join(output_folder, 'test_data.jsonl'), task_type)

    logging.info("✅ Data pre-processing and natural text conversion complete!")
    
    # ... (返回统计信息部分保持不变) ...
    stats = {
        'num_classes': num_classes, 'label_map': label_map,
        'train_size_before_smote': len(X_train), 'train_size_after_smote': len(X_train_resampled),
        'test_size': len(X_test), 'train_distribution_before_smote': y_train.value_counts().to_dict(),
        'train_distribution_after_smote': pd.Series(y_train_resampled).value_counts().to_dict(),
        'test_distribution': y_test.value_counts().to_dict()
    }
    return stats


def check_and_prepare_data(data_path, output_folder, debug_rows=None, force_reprocess=False):
    """
    检查数据是否存在，如果不存在则自动处理
    """
    # --- [修改] ---
    # 更新检查文件的路径
    train_file = os.path.join(output_folder, 'train_data.jsonl')
    test_file = os.path.join(output_folder, 'test_data.jsonl')
    
    if os.path.exists(train_file) and os.path.exists(test_file) and not force_reprocess:
        logging.info(f"✅ 发现已处理的数据文件于 '{output_folder}'，跳过处理。")
        return True
    
    if force_reprocess: logging.warning("⚠️  强制重新处理数据...")
    else: logging.warning(f"⚠️  未找到已处理的数据文件，开始处理原始数据至 '{output_folder}'...")
    
    if not os.path.exists(data_path):
        logging.error(f"❌ 原始数据文件不存在: {data_path}")
        return False
    
    try:
        # --- [修改] ---
        # 将 output_folder 传递给主处理函数
        stats = process_and_save_data(data_path, output_folder, debug_rows=debug_rows)
        
        # ... (日志打印部分保持不变) ...
        if stats:
            logging.info("\n" + "="*60)
            logging.info("📊 数据处理统计 (多分类, SMOTE):")
            logging.info(f"  训练集 (SMOTE后): {stats['train_size_after_smote']} 条 (原为 {stats['train_size_before_smote']} 条)")
            logging.info(f"  测试集: {stats['test_size']} 条 (保持不变)")
            logging.info("="*60 + "\n")
        return True
    except Exception as e:
        logging.error(f"❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    data_path = 'data/UNSW-NB15.csv'
    # --- [修改] ---
    # 定义新的输出文件夹
    output_folder = 'processed_data_SMOTE'
    debug_rows = 20000 
    
    print("="*60)
    print(f"UNSW-NB15 数据处理脚本 (输出至: {output_folder})")
    print("="*60 + "\n")
    
    success = check_and_prepare_data(
        data_path=data_path,
        output_folder=output_folder, # 传递文件夹名称
        debug_rows=debug_rows,
        force_reprocess=True 
    )
    
    if success:
        print("\n✅ 数据准备完成！可以开始训练了。")
        # --- [修改] ---
        # 更新最终提示信息
        print(f"📊 JSONL 数据和 Excel 报告已生成于 '{output_folder}' 文件夹中。")
    else:
        print("\n❌ 数据准备失败，请检查错误信息。")
