# check_embedding_hypothesis.py

import os
import torch
import pickle
from pathlib import Path
import random
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

def get_residue_ids_from_pdb(pdb_path):
    """
    我们自己编写的简单、快速的PDB残基解析器。
    """
    try:
        with open(pdb_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        residue_keys = []
        seen_keys = set()
        for line in lines:
            if line.startswith("ATOM"):
                chain_id = line[21].strip()
                res_id = int(line[22:26].strip())
                key = (chain_id, res_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    residue_keys.append(key)
        return residue_keys
    except Exception:
        return None

def check_hypothesis(num_samples=5):
    """
    主验证函数。
    """
    print("="*80)
    print("  开始验证 ESM-3 Embedding 首尾向量假设...")
    print("="*80)

    # --- 路径和模型设置 ---
    pocket_data_root = Path("/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket")
    full_protein_root = Path("/data/share/liuzhiyuan/nai/NExT-Mol/datasets/crossdocked_v1.1_rmsd1.0")
    index_path = pocket_data_root / "index.pkl"
    device = torch.device("cuda")

    print("[*] 正在加载 ESM-3 模型 (这可能需要一点时间)...")
    model = ESM3.from_pretrained("esm3-sm-open-v1").to(device).eval()
    print("[*] 模型加载成功。")

    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    # --- 随机抽样 ---
    protein_paths = list(set([item[2] for item in master_index if item[2] is not None]))
    if len(protein_paths) < num_samples:
        print(f"[!] 警告: 数据集中的独立蛋白质数量不足 {num_samples} 个。")
        num_samples = len(protein_paths)
    
    samples_to_check = random.sample(protein_paths, num_samples)
    print(f"[*] 已随机抽取 {num_samples} 个蛋白质样本进行验证。")

    # --- 收集向量 ---
    first_vectors = []
    last_vectors = []
    core_mean_vectors = []

    for i, protein_fn in enumerate(samples_to_check):
        print(f"\n--- 正在处理样本 {i+1}/{num_samples}: {protein_fn} ---")
        full_path = full_protein_root / protein_fn
        
        # 1. 获取物理残基数量
        physical_residues = get_residue_ids_from_pdb(full_path)
        if not physical_residues:
            print("[!] 解析PDB失败，跳过此样本。")
            continue
        num_physical_residues = len(physical_residues)

        # 2. 获取完整Embedding
        try:
            protein = ESMProtein.from_pdb(str(full_path))
            protein_tensor = model.encode(protein).to(device)
            embedding_config = LogitsConfig(return_embeddings=True)
            with torch.no_grad():
                output = model.logits(protein_tensor, embedding_config)
            full_embedding = output.embeddings.squeeze(0).cpu()
            num_embedding_vectors = full_embedding.shape[0]
            
            print(f"  - PDB中物理残基数: {num_physical_residues}")
            print(f"  - ESM-3输出Embedding长度: {num_embedding_vectors}")

            # 3. 验证长度关系并提取向量
            if num_embedding_vectors == num_physical_residues + 2:
                print("  - ✅ 长度符合 N+2 假设。")
                first_vectors.append(full_embedding[0, :])
                last_vectors.append(full_embedding[-1, :])
                core_mean_vectors.append(full_embedding[1:-1, :].mean(dim=0))
            else:
                print(f"  - ❌ 长度不符合 N+2 假设！差异为 {num_embedding_vectors - num_physical_residues}。跳过此样本的对比。")

        except Exception as e:
            print(f"  - ❌ 处理时发生错误: {e}")

    # --- 对比分析 ---
    print("\n" + "="*80)
    print("  向量相似度对比分析结果")
    print("="*80)

    if len(first_vectors) < 2:
        print("[!] 收集到的有效样本不足2个，无法进行对比。")
        return

    # 封装对比逻辑
    def compare_vectors(name, vector_list):
        vec_tensor = torch.stack(vector_list)
        # 归一化以计算余弦相似度
        vec_tensor_norm = torch.nn.functional.normalize(vec_tensor, p=2, dim=1)
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(vec_tensor_norm, vec_tensor_norm.T)
        # 提取上三角（不含对角线）的元素
        indices = torch.triu_indices(len(vector_list), len(vector_list), offset=1)
        pairwise_similarities = similarity_matrix[indices[0], indices[1]]
        
        avg_sim = pairwise_similarities.mean().item()
        print(f"[*] {name} 的平均余弦相似度: {avg_sim:.6f}")
        return avg_sim

    sim_first = compare_vectors("首向量 (BOS)", first_vectors)
    sim_last = compare_vectors("尾向量 (EOS)", last_vectors)
    sim_core = compare_vectors("核心向量 (Core)", core_mean_vectors)

    print("\n--- 最终结论 ---")
    if sim_first > 0.99 and sim_last > 0.99 and sim_core < 0.9:
        print("✅ 假设得到证实！不同蛋白的首、尾向量高度一致，核心内容则不同。")
        print("   我们可以安全地使用 [1:-1] 切片方案。")
    else:
        print("❌ 假设未被证实。输出向量的行为比预想的更复杂，需要进一步分析。")

if __name__ == "__main__":
    check_hypothesis()