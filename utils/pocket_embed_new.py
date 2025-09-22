# pocket_embed_new.py (最终并行版)

import os
import torch
import io
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

def get_pocket_residue_ids(pocket_pdb_path):
    """从口袋PDB文件中解析并返回一个唯一的残基序号列表。"""
    try:
        with open(pocket_pdb_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        residue_ids = set()
        for line in lines:
            if line.startswith("ATOM"):
                res_id = int(line[22:26].strip())
                residue_ids.add(res_id)
        return sorted(list(residue_ids))
    except Exception as e:
        print(f"\n[!] 解析口袋 {pocket_pdb_path} 时出错: {e}")
        return None

def process_full_protein(protein_pdb_path, model, device):
    """处理一个完整的蛋白质PDB文件，返回完整的embedding和残基ID列表。"""
    try:
        protein = ESMProtein.from_pdb(protein_pdb_path)
        full_protein_res_ids = [res_id for chain_id, res_id in protein.residue_indexes]
        protein_tensor = model.encode(protein).to(device)
        embedding_config = LogitsConfig(return_embeddings=True)
        with torch.no_grad():
            output = model.logits(protein_tensor, embedding_config)
        full_embedding = output.embeddings.squeeze(0).cpu()
        
        if full_embedding.shape[0] != len(full_protein_res_ids):
             print(f"\n[!] WARNING: Embedding长度 ({full_embedding.shape[0]}) 与残基数 ({len(full_protein_res_ids)}) 不匹配！ 文件: {protein_pdb_path}")
             return None, None
        return full_embedding, full_protein_res_ids
    except Exception as e:
        print(f"\n[!] 处理完整蛋白 {protein_pdb_path} 时出错: {e}")
        return None, None

def main(args):
    # --- 1. 明确的路径设置 ---
    pocket_data_root = Path(args.dataset_root)
    full_protein_root = Path(args.full_protein_root)
    index_path = pocket_data_root / "index.pkl"
    
    device = torch.device("cuda") # 自动使用被CUDA_VISIBLE_DEVICES指定的GPU
    
    shard_info = f"[分片 {args.shard_id}/{args.num_shards}]"
    print(f"{shard_info} -> 口袋数据根目录: {pocket_data_root}")
    print(f"{shard_info} -> 完整蛋白根目录: {full_protein_root}")

    # --- 2. 加载模型 ---
    print(f"{shard_info} 正在加载 ESM-3 模型到 {device}...")
    model = ESM3.from_pretrained("esm3-sm-open-v1").to(device).eval()
    print(f"{shard_info} 模型加载成功。")

    # --- 3. 加载并分片索引 ---
    if not index_path.exists():
        raise FileNotFoundError(f"未找到主索引文件: {index_path}")
    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None and item[2] is not None}
    all_pockets = sorted(list(pocket_to_protein_map.keys()))
    
    # 关键并行化步骤：每个分片只处理自己的任务
    pockets_for_this_shard = [p for i, p in enumerate(all_pockets) if i % args.num_shards == args.shard_id]
    
    # --- 4. 按完整蛋白质分组（仅针对当前分片的任务） ---
    protein_to_pockets_map = {}
    for pocket_fn in pockets_for_this_shard:
        protein_fn = pocket_to_protein_map.get(pocket_fn)
        if protein_fn:
            if protein_fn not in protein_to_pockets_map:
                protein_to_pockets_map[protein_fn] = []
            protein_to_pockets_map[protein_fn].append(pocket_fn)
            
    print(f"{shard_info} 总共分配到 {len(pockets_for_this_shard)} 个口袋，涉及 {len(protein_to_pockets_map)} 个独立蛋白质。")

    # --- 5. 主循环 ---
    for protein_fn, pocket_fns in tqdm(protein_to_pockets_map.items(), desc=f"{shard_info} 处理中", position=args.shard_id):
        all_done = all([(pocket_data_root / fn).with_suffix('.pt').exists() for fn in pocket_fns])
        if all_done:
            continue
            
        full_protein_path = full_protein_root / protein_fn
        full_embedding, full_res_ids = process_full_protein(str(full_protein_path), model, device)
        
        if full_embedding is None:
            # 省略了错误日志写入，简化并行下的文件写入冲突，可以直接看标准输出
            continue

        res_id_to_idx_map = {res_id: i for i, res_id in enumerate(full_res_ids)}
        
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists():
                continue

            pocket_res_ids = get_pocket_residue_ids(pocket_data_root / pocket_fn)
            if pocket_res_ids is None:
                continue

            indices_to_select = [res_id_to_idx_map[res_id] for res_id in pocket_res_ids if res_id in res_id_to_idx_map]
            
            if not indices_to_select:
                print(f"\n{shard_info} WARNING: 对于口袋 {pocket_fn}, 未在完整蛋白中找到任何对应残基。")
                continue

            pocket_embedding = full_embedding[indices_to_select, :]
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pocket_embedding, output_path)

    print(f"{shard_info} 所有任务已成功处理完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取蛋白质embeddings (最终并行版)。")
    parser.add_argument("--dataset_root", type=str, required=True, help="指向 'crossdocked_pocket' 数据集的根目录")
    parser.add_argument("--full_protein_root", type=str, required=True, help="指向包含完整蛋白质的 'crossdocked_v1.1_rmsd1.0' 目录")
    parser.add_argument("--num-shards", type=int, default=1, help="并行进程的总数")
    parser.add_argument("--shard-id", type=int, default=0, help="当前进程的ID (0-indexed)")
    args = parser.parse_args()
    main(args)