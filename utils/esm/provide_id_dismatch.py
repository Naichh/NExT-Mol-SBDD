import argparse
from pathlib import Path
import pickle
from collections import defaultdict

# 导入 esm 自己的 ProteinChain 解析器
from esm.utils.structure.protein_chain import ProteinChain

def get_full_keys_from_chain(protein_chain: ProteinChain) -> list:
    """从 esm 的 ProteinChain 对象中生成包含插入码的完整残基key。"""
    keys = []
    for rid, icode in zip(protein_chain.residue_index, protein_chain.insertion_code):
        # 创建一个与 BioPython residue.get_id() 格式兼容的元组 (' ', res_num, ins_code)
        res_tuple = (' ', int(rid), icode.strip())
        keys.append((protein_chain.chain_id, res_tuple))
    return keys

def prove_hypothesis(pocket_data_root: str, full_protein_root: str, pocket_fn: str):
    """
    对一个指定的口袋文件，验证其残基ID是否在对应的完整蛋白PDB中发生了变化。
    """
    print("="*60)
    print(f"🔬 正在调查口袋: {pocket_fn}")
    print("="*60)

    # --- 1. 定位文件 ---
    index_path = Path(pocket_data_root) / "index.pkl"
    with open(index_path, "rb") as f:
        master_index = pickle.load(f)

    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None}

    if pocket_fn not in pocket_to_protein_map:
        print(f"错误: 在 index.pkl 中找不到口袋 {pocket_fn}")
        return

    protein_fn = pocket_to_protein_map[pocket_fn]
    pocket_path = Path(pocket_data_root) / pocket_fn
    full_protein_path = Path(full_protein_root) / protein_fn

    print(f"口袋文件路径: {pocket_path}")
    print(f"完整蛋白路径: {full_protein_path}")

    try:
        # --- 2. 解析文件，获取两份残基ID列表 ---
        print("\n[1/3] 正在解析口袋PDB...")
        pocket_chain = ProteinChain.from_pdb(str(pocket_path))
        pocket_keys = get_full_keys_from_chain(pocket_chain)
        pocket_keys_set = set(pocket_keys)
        print(f" -> 口袋 '{pocket_chain.chain_id}' 链解析出 {len(pocket_keys)} 个残基。")

        target_chain_id = pocket_chain.chain_id
        print(f"\n[2/3] 正在解析完整蛋白PDB (只关注链 '{target_chain_id}')...")
        full_protein_chain = ProteinChain.from_pdb(str(full_protein_path), chain_id=target_chain_id)
        full_protein_keys = get_full_keys_from_chain(full_protein_chain)
        full_protein_keys_set = set(full_protein_keys)
        print(f" -> 完整蛋白 '{target_chain_id}' 链解析出 {len(full_protein_keys)} 个残基。")

        # --- 3. 寻找证据 ---
        print("\n[3/3] 正在交叉比对，寻找不匹配的证据...")
        unmatched_pocket_keys = pocket_keys_set - full_protein_keys_set

        if not unmatched_pocket_keys:
            print("\n✅ 未发现ID不匹配项。此文件的失败原因可能不同。")
            print("="*60)
            return

        # 创建一个从 (chain, res_num) 到 完整ID列表 的映射，用于查找嫌疑对象
        full_protein_simple_map = defaultdict(list)
        for key in full_protein_keys:
            simple_key = (key[0], key[1][1]) # (chain_id, res_num)
            full_protein_simple_map[simple_key].append(key)

        evidence_found = []
        for p_key in unmatched_pocket_keys:
            p_simple_key = (p_key[0], p_key[1][1])
            if p_simple_key in full_protein_simple_map:
                evidence = {
                    "pocket_key": p_key,
                    "potential_source_keys": full_protein_simple_map[p_simple_key]
                }
                evidence_found.append(evidence)

        # --- 4. 打印报告 ---
        if not evidence_found:
            print("\n❌ 未找到明确的因插入码不同而导致不匹配的证据。")
            print("   不匹配的口袋残基ID如下:")
            for key in unmatched_pocket_keys: print(f"    - {key}")
        else:
            print("\n--- 证据清单 ---")
            print("在完整蛋白中找到了以下潜在的ID不匹配项：\n")
            for evidence in evidence_found:
                pk_chain, (pk_het, pk_num, pk_icode) = evidence['pocket_key']
                print(f"- 口袋中的残基ID:           (链: '{pk_chain}', 编号: {pk_num}, 插入码: '{pk_icode}')")

                for sk in evidence['potential_source_keys']:
                    sk_chain, (sk_het, sk_num, sk_icode) = sk
                    print(f"  在完整蛋白中找到的对应ID: (链: '{sk_chain}', 编号: {sk_num}, 插入码: '{sk_icode}')")
                print("  => 证据：标识信息（很可能是插入码）在口袋文件中发生了变化。\n")

            print("-" * 60)
            print("✅ 假设得到证实：")
            print("匹配失败的根本原因是口袋PDB文件中的残基ID信息")
            print("与其在完整蛋白质PDB文件中的原始信息不一致。")

    except Exception as e:
        print(f"\n[!] 在处理过程中发生错误: {e}")

    print("="*60)
    print("调查完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证口袋和完整蛋白PDB文件之间残基ID不一致的假设。")
    parser.add_argument(
        "pocket_filename",
        type=str,
        nargs='?', # 使参数成为可选的
        default="ACAC_YEAST_1477_2229_inhibitor_0/5cte_B_rec_3tv5_rcp_lig_tt_docked_2_pocket10.pdb", # 使用一个您日志中的失败案例作为默认值
        help="要调查的口袋文件的相对路径 (相对于 POCKET_DATA_ROOT)。"
    )
    args = parser.parse_args()

    POCKET_DATA_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
    FULL_PROTEIN_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/crossdocked_v1.1_rmsd1.0"

    prove_hypothesis(POCKET_DATA_ROOT, FULL_PROTEIN_ROOT, args.pocket_filename)
