# preprocess_embeddings.py (最终确认版)

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
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[!] 解析口袋 {pocket_pdb_path} 时出错: {e}")
        return None

def process_full_protein(protein_pdb_path, model, device):
    """
    处理一个完整的蛋白质PDB文件，返回完整的embedding和它所代表的结构中的残基ID列表。
    """
    try:
        # from_pdb 可以直接处理真实的PDB文件，无需我们手动添加MODEL/ENDMDL
        protein = ESMProtein.from_pdb(protein_pdb_path, chain_ids=None)

        # ESMProtein对象有一个 .residue_indexes 属性
        # 它是一个元组列表，包含了embedding每个位置对应的 (链ID, 残基ID)
        # 我们只提取残基ID
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
    # --- 路径设置 ---
    # 根据您之前的确认，我们在这里明确路径结构
    # 注意：dataset_root 现在指向包含所有数据的大目录，例如 /.../datasets/sbdd/
    dataset_root = Path(args.dataset_root)
    full_protein_root = dataset_root.parent / "crossdocked_v1.1_rmsd1.0"
    pocket_data_root = dataset_root
    index_path = pocket_data_root / "index.pkl"

    print(f"[*] 口袋数据根目录: {pocket_data_root}")
    print(f"[*] 完整蛋白根目录: {full_protein_root}")

    # 1. 加载模型
    print("[*] 正在加载 ESM-3 模型...")
    device = torch.device(args.device)
    model = ESM3.from_pretrained("esm3-sm-open-v1").to(device).eval()
    print("[*] 模型加载成功。")

    # 2. 加载索引
    if not index_path.exists():
        raise FileNotFoundError(f"未找到主索引文件: {index_path}")
    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    # 3. 创建从口袋到完整蛋白的映射
    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None and item[2] is not None}
    
    # 4. 按完整蛋白质进行分组，避免重复处理
    protein_to_pockets_map = {}
    for pocket, protein in pocket_to_protein_map.items():
        if protein not in protein_to_pockets_map:
            protein_to_pockets_map[protein] = []
        protein_to_pockets_map[protein].append(pocket)

    print(f"[*] 共涉及 {len(protein_to_pockets_map)} 个独立的完整蛋白质，对应 {len(pocket_to_protein_map)} 个口袋。")

    # 5. 主循环：遍历每一个完整的蛋白质
    for protein_fn, pocket_fns in tqdm(protein_to_pockets_map.items(), desc="处理完整蛋白质"):
        # 检查是否所有相关的口袋都已处理完毕
        all_done = all([(pocket_data_root / pocket_fn).with_suffix('.pt').exists() for pocket_fn in pocket_fns])
        if all_done:
            continue
            
        full_protein_path = full_protein_root / protein_fn
        full_embedding, full_res_ids = process_full_protein(str(full_protein_path), model, device)
        
        if full_embedding is None:
            with open(pocket_data_root / "embedding_errors.log", "a") as error_log:
                error_log.write(f"{full_protein_path}\tPROCESS_FULL_PROTEIN_FAILED\n")
            continue

        res_id_to_idx_map = {res_id: i for i, res_id in enumerate(full_res_ids)}
        
        # 遍历这个蛋白质对应的所有口袋
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists():
                continue

            pocket_res_ids = get_pocket_residue_ids(pocket_data_root / pocket_fn)
            if pocket_res_ids is None:
                with open(pocket_data_root / "embedding_errors.log", "a") as error_log:
                    error_log.write(f"{pocket_data_root / pocket_fn}\tGET_POCKET_RES_IDS_FAILED\n")
                continue

            indices_to_select = [res_id_to_idx_map[res_id] for res_id in pocket_res_ids if res_id in res_id_to_idx_map]
            
            if not indices_to_select:
                print(f"\n[!] WARNING: 对于口袋 {pocket_fn}, 未在完整蛋白中找到任何对应残基。")
                continue

            pocket_embedding = full_embedding[indices_to_select, :]
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pocket_embedding, output_path)

    print("[*] 所有口袋已成功处理完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取蛋白质embeddings (最终解决方案)。")
    # 注意这里 --dataset_root 应该指向包含 index.pkl 的 crossdocked_pocket 目录
    parser.add_argument("--dataset_root", type=str, required=True, help="指向 'crossdocked_pocket' 数据集的根目录")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)