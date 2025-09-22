# verify_embeddings.py
import torch
import os
import random
import pickle
from pathlib import Path

def verify():
    dataset_root = Path("/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket")
    index_path = dataset_root / 'index.pkl'
    
    print(f"[*] 正在从 {index_path} 加载索引...")
    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)
    
    pocket_files_relative = sorted(list(set([item[0] for item in master_index if item[0] is not None])))
    
    if len(pocket_files_relative) < 2:
        print("[!] 数据集中的口袋数量不足2，无法进行比较。")
        return

    # 从列表中随机抽取两个不同的文件进行比较
    sample_files = random.sample(pocket_files_relative, 2)
    
    print(f"[*] 正在比较文件: {sample_files[0]} 和 {sample_files[1]}")

    try:
        # 加载两个embedding文件
        emb1_path = (dataset_root / sample_files[0]).with_suffix('.pt')
        emb2_path = (dataset_root / sample_files[1]).with_suffix('.pt')
        
        emb1 = torch.load(emb1_path)
        emb2 = torch.load(emb2_path)

        # 将它们处理成单个向量 (L, D) -> (D)
        avg_emb1 = torch.mean(emb1.float(), dim=0)
        avg_emb2 = torch.mean(emb2.float(), dim=0)

        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(avg_emb1, avg_emb2, dim=0)
        
        print("\n" + "="*50)
        print("  VERIFICATION RESULT")
        print("="*50)
        print(f"  两个随机口袋表征的余弦相似度为: {similarity.item():.4f}")
        
        if similarity.item() < 0.9:
            print("  ✅ 验证通过！相似度处于健康范围。")
        else:
            print("  ❌ 验证失败！相似度仍然过高，请检查预处理脚本。")
        print("="*50)

    except Exception as e:
        print(f"[!] 验证过程中发生错误: {e}")

if __name__ == "__main__":
    verify()