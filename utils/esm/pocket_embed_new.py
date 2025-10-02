
import os
import torch
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
from collections import defaultdict

from esm.models.esm3 import ESM3
import esm
from esm.sdk.api import ESMProtein, SamplingConfig, ESM3InferenceClient
from esm.utils.structure.protein_chain import ProteinChain
from biotite.structure.io.pdb import PDBFile
import biotite.structure as bs
from esm.utils.residue_constants import restype_3to1 as ESM_AA_3_TO_1

warnings.filterwarnings("ignore", category=UserWarning)


#获取某个pocket相关链的集合
def get_pocket_info_esm(pdb_path: str) -> tuple[set, list] | None:
    try:
        pdb_file = PDBFile.read(pdb_path)
        structure = pdb_file.get_structure(model=1)

        chain_ids = set()
        residue_keys = []


        for res in bs.residue_iter(structure):
            if res.res_name[0] in ESM_AA_3_TO_1:
                chain_id = res.chain_id[0]
                res_num = res.res_id[0]
                ins_code = res.ins_code[0].strip()

                chain_ids.add(chain_id)
                residue_keys.append((chain_id, (' ', res_num, ins_code)))
        return chain_ids, residue_keys
    except Exception as e:
        print(f"\n[!] Biotite解析PDB {Path(pdb_path).name} 获取信息时出错: {e}")
        return None

#获取residue索引
def get_keys_from_esm_chain(protein_chain: ProteinChain) -> list:
    return [
        (protein_chain.chain_id, (' ', int(rid), icode.strip()))
        for rid, icode in zip(protein_chain.residue_index, protein_chain.insertion_code)
    ]

#根据chain提取表征
def get_structure_embedding(protein: ESMProtein, model) -> torch.Tensor | None:
    try:
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            protein_tensor = model.encode(protein)
            output = model.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=True)
            )
        full_embedding = output.per_residue_embedding.squeeze(0).to(torch.float32).cpu()
        core_embedding = full_embedding[1:-1, :]
        return core_embedding
    except Exception as e:
        print(f"\n[!] ESM-3 推理时出错: {e}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

def main():
    parser = argparse.ArgumentParser(description="使用 ESM-3 为蛋白质口袋提取结构感知表征")
    parser.add_argument("--dataset_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--full_protein_root", type=str, required=True, help="完整蛋白质PDB目录")
    parser.add_argument("--num-shards", type=int, required=True, help="总分片数量")
    parser.add_argument("--shard-id", type=int, required=True, help="当前分片ID")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard_info = f"[分片 {args.shard_id + 1}/{args.num_shards}]"
    #加载模型
    print(f"{shard_info} 正在加载 ESM-3 模型到 {device}...")
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    model.eval()
    print(f"{shard_info} 模型加载成功。")
    #读取数据
    pocket_data_root = Path(args.dataset_root)
    full_protein_root = Path(args.full_protein_root)
    index_path = pocket_data_root / "index.pkl"

    with open(index_path, "rb") as f: master_index = pickle.load(f)
    #分配蛋白质
    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None and item[2] is not None}
    all_proteins = sorted(list(set(pocket_to_protein_map.values())))
    proteins_for_this_process = all_proteins[args.shard_id::args.num_shards]
    print(f"{shard_info} 分配到 {len(proteins_for_this_process)} 个蛋白质进行处理。")
    #构建protein-pocket的映射（一对多）
    protein_to_pockets_map = defaultdict(list)
    for pocket, protein in pocket_to_protein_map.items():
        protein_to_pockets_map[protein].append(pocket)

    #对每个protein处理
    for protein_fn in tqdm(proteins_for_this_process, desc=shard_info, position=args.shard_id):
        pocket_fns = protein_to_pockets_map.get(protein_fn, [])
        if all((pocket_data_root / fn).with_suffix('.pt').exists() for fn in pocket_fns):
            continue

        full_protein_path = full_protein_root / protein_fn
        if not full_protein_path.exists():
            print(f"\n[!] 警告: 找不到完整蛋白质文件 {full_protein_path}, 跳过。")
            continue
        # 存储提取出的单个残基表征，键为(pocket_fn, res_key)
        partial_embeddings = {}
        pocket_meta = {}
        pockets_by_chain = defaultdict(list)
        #对每个pocket处理
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists(): continue
            #获取每个pocket相关链和残基索引
            pocket_info = get_pocket_info_esm(str(pocket_data_root / pocket_fn))
            if not pocket_info: continue

            chain_ids, ordered_keys = pocket_info
            #记录每个pocket的chain/key映射
            pocket_meta[pocket_fn] = {"chains": chain_ids, "keys": ordered_keys}
            for chain_id in chain_ids:
                pockets_by_chain[chain_id].append(pocket_fn)
        #对每条链处理，然后将表征分发给链上的所有口袋
        for chain_id, chain_pocket_fns in pockets_by_chain.items():
            try:
                full_protein_chain = ProteinChain.from_pdb(str(full_protein_path), chain_id=chain_id)
                protein_for_model = ESMProtein.from_protein_chain(full_protein_chain)
                # 获取完整链的有序残基键
                full_res_keys = get_keys_from_esm_chain(full_protein_chain)

                core_embedding = get_structure_embedding(protein_for_model, model)

                if core_embedding is None or core_embedding.shape[0] != len(full_res_keys):
                    print(f"\n[!] 警告: {protein_fn} (链 {chain_id}) 长度不匹配或推理失败，跳过。")
                    torch.cuda.empty_cache()
                    continue

                res_key_to_idx_map = {res_key: i for i, res_key in enumerate(full_res_keys)}
                #将整条链的表征分配给这条链上的每个口袋
                for pocket_fn in chain_pocket_fns:
                    if (pocket_data_root / pocket_fn).with_suffix('.pt').exists(): continue

                    meta = pocket_meta[pocket_fn]

                    for res_key in meta["keys"]:
                        if res_key[0] == chain_id:
                            idx = res_key_to_idx_map.get(res_key)
                            if idx is not None:
                                partial_embeddings[(pocket_fn, res_key)] = core_embedding[idx]
            except Exception as e:
                print(f"\n[!] 处理蛋白质 {protein_fn} 的链 {chain_id} 时发生严重错误: {e}")
                continue
        # 将从各条链中提取的零散残基表征，按口袋重新组合
        final_pockets_to_save = defaultdict(list)
        for (pocket_fn, res_key), embedding_vec in partial_embeddings.items():
            final_pockets_to_save[pocket_fn].append((res_key, embedding_vec))

        for pocket_fn, data in final_pockets_to_save.items():
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists(): continue

            original_order_keys = pocket_meta[pocket_fn]["keys"]
            vec_map = {res_key: vec for res_key, vec in data}
            ordered_vecs = [vec_map[res_key] for res_key in original_order_keys if res_key in vec_map]

            if len(ordered_vecs) != len(original_order_keys):
                 print(f"\n[!] 警告: 口袋 {pocket_fn} 的部分残基在合并时丢失。")

            if not ordered_vecs: continue

            final_embedding = torch.stack(ordered_vecs, dim=0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(final_embedding, output_path)

    print(f"{shard_info} 所有任务已成功处理完毕。")

if __name__ == "__main__":
    main()
