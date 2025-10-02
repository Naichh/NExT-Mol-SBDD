import argparse
from pathlib import Path

# BioPython用于逐链分析
from Bio.PDB import PDBParser

# esm自己的解析器，用于模拟我们脚本中的行为
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants.residue_constants import restype_3to1

def analyze_pdb_parsers(pdb_path: str):
    """
    使用两种不同的解析器分析同一个PDB文件，并报告它们的残基计数差异。
    """
    if not Path(pdb_path).exists():
        print(f"错误: 文件不存在 -> {pdb_path}")
        return

    print("="*80)
    print(f"🔍 正在诊断文件: {Path(pdb_path).name}")
    print("="*80)

    # --- 1. 使用 esm 的内部解析器 (默认行为) ---
    print("\n--- 1. ESM 内部解析器结果 ---")
    try:
        # 这会模拟 ESMProtein.from_pdb(pdb_path) 的默认行为
        esm_chain = ProteinChain.from_pdb(pdb_path, chain_id="detect")
        esm_residue_count = len(esm_chain.sequence)
        detected_chain_id = esm_chain.chain_id
        print(f"✅ ESM 成功加载。")
        print(f"   - 它选择处理的链 (Chain ID): '{detected_chain_id}'")
        print(f"   - 它统计出的残基数量: {esm_residue_count}")
    except Exception as e:
        print(f"❌ ESM 加载失败: {e}")
        esm_residue_count = -1

    # --- 2. 使用 BioPython 解析器 (逐链分析) ---
    print("\n--- 2. BioPython 解析器结果 ---")
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        chain_counts = {}
        total_residues = 0

        for chain in structure.get_chains():
            chain_id = chain.get_id()
            count = 0
            for residue in chain.get_residues():
                # 只统计标准的氨基酸残基
                if residue.get_resname() in restype_3to1:
                    count += 1
            if count > 0:
                chain_counts[chain_id] = count
                total_residues += count

        print(f"✅ BioPython 成功加载。")
        print(f"   - 它在文件中找到的链和各自的残基数: {chain_counts}")
        print(f"   - 它统计出的总残基数量: {total_residues}")

    except Exception as e:
        print(f"❌ BioPython 加载失败: {e}")
        total_residues = -1

    # --- 3. 最终结论 ---
    print("\n--- 3. 结论 ---")
    if esm_residue_count != -1 and total_residues != -1:
        if esm_residue_count == total_residues:
            print("两边解析的残基总数一致。")
        else:
            print("‼️ 两边解析的残基总数不一致！")
            print(f"   - ESM 模型实际处理的残基数: {esm_residue_count} (来自链 '{detected_chain_id}')")
            print(f"   - 我们之前脚本中错误统计的总数: {total_residues} (来自所有链的总和)")
            print("\n   => 这就精确解释了为什么 Embedding 长度与 PDB 解析残基数不匹配。")
    else:
        print("由于某个解析器失败，无法进行比较。")

    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="精确诊断PDB文件解析差异的工具。")
    parser.add_argument("pdb_file", type=str, help="需要诊断的PDB文件的完整路径。")
    args = parser.parse_args()

    analyze_pdb_parsers(args.pdb_file)
