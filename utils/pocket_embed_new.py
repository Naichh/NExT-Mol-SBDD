# preprocess_unified.py (最终修正版)

import os
import torch
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

from accelerate import Accelerator

warnings.filterwarnings('ignore', category=FutureWarning)

from Bio.PDB import PDBParser
from Bio.Data import PDBData

from esm.pretrained import ESM3_sm_open_v0

AA_3_TO_1 = PDBData.protein_letters_3to1

def get_sequence_and_res_keys_from_pdb(pdb_path):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        sequence = ""
        residue_keys = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    if res_name in AA_3_TO_1:
                        sequence += AA_3_TO_1[res_name]
                        res_id = residue.get_id()[1]
                        chain_id = chain.get_id()
                        residue_keys.append((chain_id, res_id))
        return sequence, residue_keys
    except Exception as e:
        print(f"\n[!] 解析PDB {pdb_path.name} 出错: {e}")
        return None, None

def process_protein(sequence, res_keys, model, tokenizer, device):
    try:
        tokens = tokenizer.encode(sequence)
        token_ids = torch.tensor(tokens, dtype=torch.int64).to(device).unsqueeze(0)

        with torch.no_grad():
            output = model.forward(sequence_tokens=token_ids)
        
        full_embedding = output.embeddings.squeeze(0).cpu()
        core_embedding = full_embedding[1:-1, :]

        if core_embedding.shape[0] != len(res_keys):
             print(f"\n[!] 长度不匹配! Emb: {core_embedding.shape[0]}, Keys: {len(res_keys)}.")
             return None
             
        return core_embedding

    except Exception as e:
        print(f"\n[!] 推理时出错: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def main():
    # --- 关键修正：使用 parse_known_args() ---
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取蛋白质embeddings (统一并行版)。")
    parser.add_argument("--dataset_root", type=str, required=True, help="指向 'crossdocked_pocket' 数据集的根目录")
    parser.add_argument("--full_protein_root", type=str, required=True, help="指向 'crossdocked_v1.1_rmsd1.0' 目录")
    args, unknown = parser.parse_known_args() # <--- 在这里修改
    # --- 修正结束 ---

    accelerator = Accelerator()
    device = accelerator.device
    shard_info = f"[进程 {accelerator.process_index}/{accelerator.num_processes} on {str(device)}]"
    
    if accelerator.is_main_process:
        print(f"[*] 正在使用 Accelerate 加载 ESM-3 模型 (device_map='auto')...")
    
    model = ESM3_sm_open_v0(device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    sequence_tokenizer = model.tokenizers.sequence
    
    if accelerator.is_main_process:
        print(f"[*] 模型和分词器加载成功。")

    pocket_data_root = Path(args.dataset_root)
    full_protein_root = Path(args.full_protein_root)
    index_path = pocket_data_root / "index.pkl"

    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None and item[2] is not None}
    all_pockets = sorted(list(pocket_to_protein_map.keys()))
    
    protein_to_pockets_map = {}
    for pocket, protein in pocket_to_protein_map.items():
        if protein not in protein_to_pockets_map:
            protein_to_pockets_map[protein] = []
        protein_to_pockets_map[protein].append(pocket)

    all_proteins = sorted(list(protein_to_pockets_map.keys()))
    
    proteins_for_this_process = all_proteins[accelerator.process_index::accelerator.num_processes]
    
    if accelerator.is_main_process:
        print(f"[*] 总共发现 {len(all_proteins)} 个独立蛋白质。")
    print(f"{shard_info} 分配到 {len(proteins_for_this_process)} 个蛋白质进行处理。")
    
    accelerator.wait_for_everyone()

    for protein_fn in tqdm(proteins_for_this_process, desc=shard_info, position=accelerator.process_index):
        pocket_fns = protein_to_pockets_map[protein_fn]
        all_done = all([(pocket_data_root / fn).with_suffix('.pt').exists() for fn in pocket_fns])
        if all_done:
            continue
            
        full_protein_path = full_protein_root / protein_fn
        sequence, full_res_keys = get_sequence_and_res_keys_from_pdb(str(full_protein_path))
        if not sequence:
            continue
        
        core_embedding = process_protein(sequence, full_res_keys, model, sequence_tokenizer, device)
        
        if core_embedding is None:
            continue

        res_key_to_idx_map = {res_key: i for i, res_key in enumerate(full_res_keys)}
        
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists(): continue
            
            _, pocket_res_keys = get_sequence_and_res_keys_from_pdb(pocket_data_root / pocket_fn)
            if not pocket_res_keys: continue

            indices_to_select = [res_key_to_idx_map.get(res_key) for res_key in pocket_res_keys]
            indices_to_select = [idx for idx in indices_to_select if idx is not None]

            if not indices_to_select: continue
            
            pocket_embedding = core_embedding[indices_to_select, :]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pocket_embedding, output_path)

    print(f"{shard_info} 所有任务已成功处理完毕。")

if __name__ == "__main__":
    main()