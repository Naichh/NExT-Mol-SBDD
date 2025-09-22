# utils/audit_embeddings.py

import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import os

def audit_main():
    """
    只读不写。
    严格按照原始split文件，检查每个条目对应的.pt文件是否存在，
    并打印一份详细的审计报告，包括缺失文件的列表。
    """
    parser = argparse.ArgumentParser(description="审计数据集，报告有效和缺失的embedding文件。")
    parser.add_argument("--dataset_root", type=str, required=True, help="指向 'crossdocked_pocket' 数据集的根目录")
    args = parser.parse_args()

    print("="*80)
    print("  开始进行数据集完整性审计 (只读模式)...")
    print("="*80)

    dataset_root = Path(args.dataset_root)
    split_file_path = dataset_root / "split_by_name.pt"

    # 1. 检查原始文件是否存在
    if not split_file_path.exists():
        print(f"❌ 错误: 原始划分文件不存在: {split_file_path}")
        return

    # 2. 加载原始划分文件
    print(f"[*] 正在加载原始划分文件: {split_file_path}")
    original_split = torch.load(split_file_path)
    
    missing_files_report = {'train': [], 'test': []}
    
    # 3. 逐一检查条目，统计有效和缺失的数量
    for split_type in ['train', 'test']:
        print(f"\n[*] 正在审计 '{split_type}' 部分...")
        original_entries = original_split.get(split_type, [])
        if not original_entries:
            print(f"[*] '{split_type}' 部分为空，跳过。")
            continue
        
        valid_count = 0
        for entry in tqdm(original_entries, desc=f"检查 {split_type} 条目"):
            pocket_fn = entry[0]
            embedding_path = (dataset_root / pocket_fn).with_suffix('.pt')
            
            if embedding_path.exists():
                valid_count += 1
            else:
                missing_files_report[split_type].append(str(embedding_path))
        
        total_count = len(original_entries)
        missing_count = total_count - valid_count
        
        print(f"[*] '{split_type}' 部分审计完成。")
        print(f"    - 原始清单样本数: {total_count}")
        print(f"    - 经验证有效样本数: {valid_count}")
        print(f"    - 已识别无效/缺失样本数: {missing_count}")
        if total_count > 0:
            valid_percentage = (valid_count / total_count) * 100
            print(f"    - 数据完整度: {valid_percentage:.2f}%")

    # 4. 打印详细的缺失文件报告
    print("\n" + "="*80)
    print("  审计报告总结")
    print("="*80)

    for split_type in ['train', 'test']:
        missing_list = missing_files_report[split_type]
        if missing_list:
            print(f"\n[!] 在 '{split_type}' 集合中，发现 {len(missing_list)} 个缺失的embedding文件。")
            # 只打印前10个作为示例，避免刷屏
            print("    缺失文件路径示例：")
            for i, path in enumerate(missing_list[:]):
                print(f"      - {path}")
            if len(missing_list) > 10:
                print("      - ... (更多文件未显示)")
        else:
            print(f"\n[✅] '{split_type}' 集合数据完整，未发现缺失文件。")
    print("="*80)

if __name__ == "__main__":
    audit_main()