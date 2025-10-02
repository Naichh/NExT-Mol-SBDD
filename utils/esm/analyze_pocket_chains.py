import os
import pickle
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.Data import PDBData
import warnings

# 忽略 PDB 解析中的一些常见警告
warnings.filterwarnings("ignore", category=UserWarning)

def analyze_pockets(dataset_root: str):
    """
    分析数据集中所有口袋的链构成情况。
    """
    index_path = Path(dataset_root) / "index.pkl"
    if not index_path.exists():
        print(f"错误: 在指定路径下找不到 index.pkl 文件: {dataset_root}")
        return

    print(f"正在从 {index_path} 加载索引...")
    with open(index_path, "rb") as f:
        master_index = pickle.load(f)

    # 从索引中提取所有口袋文件的相对路径
    pocket_filenames = [item[0] for item in master_index if item[0] is not None]
    total_pockets = len(pocket_filenames)
    print(f"在索引中找到 {total_pockets} 个口袋文件，开始分析...")

    # 初始化计数器
    single_chain_count = 0
    multi_chain_count = 0
    error_count = 0
    multi_chain_examples = [] # 记录一些多链口袋的例子

    parser = PDBParser(QUIET=True)

    for pocket_fn in tqdm(pocket_filenames, desc="正在分析口袋构成"):
        pocket_path = Path(dataset_root) / pocket_fn
        if not pocket_path.exists():
            error_count += 1
            continue

        try:
            structure = parser.get_structure("pocket", str(pocket_path))

            # 使用集合来存储该口袋中出现过的所有链ID
            chain_ids_in_pocket = set()

            for residue in structure.get_residues():
                # 只考虑标准的氨基酸残基
                if residue.get_resname() in PDBData.protein_letters_3to1:
                    chain_ids_in_pocket.add(residue.get_parent().get_id())

            num_chains = len(chain_ids_in_pocket)
            if num_chains == 1:
                single_chain_count += 1
            elif num_chains > 1:
                multi_chain_count += 1
                # 记录最多10个例子用于展示
                if len(multi_chain_examples) < 10:
                    multi_chain_examples.append(pocket_fn)
            else:
                # 如果一个口袋PDB里一个标准氨基酸都找不到
                error_count += 1
        except Exception as e:
            print(f"\n[!] 解析文件 {pocket_fn} 时出错: {e}")
            error_count += 1

    # --- 打印最终的分析报告 ---
    print("\n" + "="*50)
    print("          口袋链构成分析报告")
    print("="*50)
    print(f"总计口袋数量: {total_pockets}")
    print("-" * 50)

    single_chain_percent = (single_chain_count / total_pockets) * 100 if total_pockets > 0 else 0
    multi_chain_percent = (multi_chain_count / total_pockets) * 100 if total_pockets > 0 else 0
    error_percent = (error_count / total_pockets) * 100 if total_pockets > 0 else 0

    print(f"  - 单链构成的口袋   : {single_chain_count} ({single_chain_percent:.2f}%)")
    print(f"  - 多链构成的口袋   : {multi_chain_count} ({multi_chain_percent:.2f}%) (界面口袋)")
    print(f"  - 解析错误/无氨基酸 : {error_count} ({error_percent:.2f}%)")
    print("="*50)

    if multi_chain_examples:
        print("\n部分多链（界面）口袋示例:")
        for example in multi_chain_examples:
            print(f"  - {example}")

    print("\n分析完成。")


if __name__ == "__main__":
    # --- 路径已硬编码 ---
    POCKET_DATA_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
    analyze_pockets(POCKET_DATA_ROOT)
