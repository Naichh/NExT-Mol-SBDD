import os
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import random
from collections import Counter
from Bio.PDB import PDBParser
from Bio.Data import PDBData
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

AA_3_TO_1 = PDBData.protein_letters_3to1
def get_residue_count_from_pdb(pdb_path: str) -> int | None:
    """快速、准确地计算PDB文件中的标准氨基酸残基数量。"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        count = 0
        for residue in structure.get_residues():
            if residue.get_resname() in AA_3_TO_1:
                count += 1
        return count
    except Exception as e:
        print(f"\n[!] BioPython解析PDB {Path(pdb_path).name} 时出错: {e}")
        return None

def analyze_generated_embeddings(dataset_root: str, sample_size: int = 5000):
    """
    对已生成的 embedding 文件进行质量分析，并打印第一个样本的形状。
    """
    index_path = Path(dataset_root) / "index.pkl"
    if not index_path.exists():
        print(f"错误: 在指定路径下找不到 index.pkl 文件: {dataset_root}")
        return

    print(f"正在从 {index_path} 加载索引...")
    with open(index_path, "rb") as f:
        master_index = pickle.load(f)

    all_pocket_fns = [item[0] for item in master_index if item[0] is not None]

    successfully_generated = []
    for pocket_fn in all_pocket_fns:
        pt_path = (Path(dataset_root) / pocket_fn).with_suffix('.pt')
        if pt_path.exists():
            successfully_generated.append(pocket_fn)

    if not successfully_generated:
        print("错误: 未找到任何已生成的 .pt 文件。")
        return

    num_to_sample = min(sample_size, len(successfully_generated))
    print(f"共找到 {len(successfully_generated)} 个成功生成的 embedding。将随机抽样 {num_to_sample} 个进行质量分析...")
    sampled_pockets = random.sample(successfully_generated, num_to_sample)

    # 初始化问题追踪器
    corrupted_files, nan_files, inf_files, shape_mismatch_files = [], [], [], []
    dimensions_found = []
    global_means, global_stds, global_mins, global_maxs = [], [], [], []

    first_embedding_analyzed = False # 用于标记是否已分析过第一个样本

    for pocket_fn in tqdm(sampled_pockets, desc="正在分析 Embedding 质量"):
        pt_path = (Path(dataset_root) / pocket_fn).with_suffix('.pt')
        pdb_path = (Path(dataset_root) / pocket_fn)

        try:
            # --- 核心改动 1: 加入 weights_only=True 参数消除警告 ---
            embedding = torch.load(pt_path, map_location='cpu', weights_only=True)
            if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
                corrupted_files.append(pocket_fn)
                continue

            # --- 核心改动 2: 打印第一个样本的详细信息 ---
            if not first_embedding_analyzed:
                print("\n\n--- 单个样本 Embedding 形状分析 ---")
                n_residues, d_embedding = embedding.shape
                print(f"第一个成功加载的样本: {pocket_fn}")
                print(f"  - 张量形状 (Tensor Shape): {embedding.shape}")
                print(f"  - 口袋中的残基数量 (N): {n_residues}")
                print(f"  - Embedding 维度 (D): {d_embedding}  <--- 这是我们要确认的维度")
                print("------------------------------------\n")
                first_embedding_analyzed = True

            if torch.isnan(embedding).any(): nan_files.append(pocket_fn)
            if torch.isinf(embedding).any(): inf_files.append(pocket_fn)

            n_embed, d_embed = embedding.shape
            dimensions_found.append(d_embed)

            n_pdb = get_residue_count_from_pdb(str(pdb_path))
            if n_pdb is None or n_embed != n_pdb:
                shape_mismatch_files.append(f"{pocket_fn} (Emb: {n_embed}, PDB: {n_pdb})")

            global_means.append(embedding.mean().item())
            global_stds.append(embedding.std().item())
            global_mins.append(embedding.min().item())
            global_maxs.append(embedding.max().item())

        except Exception as e:
            print(f"\n[!] 读取或处理文件 {pocket_fn} 时出错: {e}")
            corrupted_files.append(pocket_fn)

    # --- 打印最终的分析报告 (这部分保持不变) ---
    print("\n" + "="*60)
    print("          Embedding 质量分析报告")
    # ... (报告的其余部分与之前相同) ...
    print("="*60)
    print(f"总计抽样检查文件数: {num_to_sample}")
    print("-" * 60)
    print(f"1. 文件格式检查: {len(corrupted_files)} 个问题文件")
    print(f"2. 数值稳定性检查: {len(nan_files)} 个含NaN, {len(inf_files)} 个含inf")
    print(f"3. 维度一致性检查 (N vs PDB): {len(shape_mismatch_files)} 个不匹配文件")
    dim_counts = Counter(dimensions_found)
    print("\n4. Embedding 维度统计 (D):")
    if not dim_counts:
        print("  - 未能成功统计维度。")
    else:
        print(f"  - 发现的维度及其频次: {dict(dim_counts)}")
    print("\n5. 全局数值统计 (抽样平均):")
    if global_means:
        print(f"  - 平均最小值: {np.mean(global_mins):.4f}, 平均最大值: {np.mean(global_maxs):.4f}")
        print(f"  - 平均均值:   {np.mean(global_means):.4f}, 平均标准差: {np.mean(global_stds):.4f}")
    print("="*60)

    print("\n分析完成。")

if __name__ == "__main__":
    POCKET_DATA_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
    analyze_generated_embeddings(POCKET_DATA_ROOT)
