import os
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
import argparse
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
    except Exception:
        return None

def clean_and_report(dataset_root: str, full_protein_root: str, output_log_file: str, execute_deletion: bool = False):
    """
    检查所有生成的 embedding，报告并选择性删除行数与PDB残基数不匹配的文件。
    在干运行模式下，会生成一份包含不匹配文件路径的txt日志。
    """
    index_path = Path(dataset_root) / "index.pkl"
    if not index_path.exists():
        print(f"错误: 在指定路径下找不到 index.pkl 文件: {dataset_root}")
        return

    print(f"正在从 {index_path} 加载索引...")
    with open(index_path, "rb") as f:
        master_index = pickle.load(f)

    all_pocket_info = [(item[0], item[2]) for item in master_index if item[0] is not None and item[2] is not None]

    print(f"从索引中找到 {len(all_pocket_info)} 个口袋定义，开始一致性检查...")

    mismatched_files_info = []

    for pocket_fn, protein_fn in tqdm(all_pocket_info, desc="正在检查一致性"):
        pt_path = (Path(dataset_root) / pocket_fn).with_suffix('.pt')

        # 只检查已成功生成的文件
        if not pt_path.exists():
            continue

        pdb_path = pt_path.with_suffix('.pdb')

        try:
            embedding = torch.load(pt_path, map_location='cpu', weights_only=True)
            n_embed = embedding.shape[0]

            n_pdb = get_residue_count_from_pdb(str(pdb_path))

            if n_pdb is None or n_embed != n_pdb:
                mismatched_files_info.append({
                    "pt_path": pt_path,
                    "pocket_fn": pocket_fn,
                    "protein_fn": protein_fn
                })
        except Exception:
            mismatched_files_info.append({
                "pt_path": pt_path,
                "pocket_fn": pocket_fn,
                "protein_fn": protein_fn
            })

    # --- 报告和执行 ---
    print("\n" + "="*50)
    print("          数据一致性检查与清理报告")
    print("="*50)

    if not mismatched_files_info:
        print("✅ 恭喜！所有已生成的 embedding 文件均通过一致性检查！")
    else:
        print(f"[!] 检测到 {len(mismatched_files_info)} 个数据不一致（或损坏）的文件。")

        if execute_deletion:
            print("\n--- 正在执行删除操作 ---")
            for info in tqdm(mismatched_files_info, desc="正在删除文件"):
                try:
                    os.remove(info["pt_path"])
                except OSError as e:
                    print(f"\n删除文件失败: {info['pt_path']}, 错误: {e}")
            print(f"\n✅ 成功删除了 {len(mismatched_files_info)} 个不一致的文件。")
        else:
            print(f"\n--- 这是“干运行”(Dry Run) 模式 ---")
            print(f"正在将这 {len(mismatched_files_info)} 个文件的路径写入到 {output_log_file} ...")
            with open(output_log_file, 'w') as f:
                f.write("pocket_path\tprotein_path\n") # 写入表头
                for info in mismatched_files_info:
                    full_pocket_path = Path(dataset_root) / info["pocket_fn"]
                    full_protein_path = Path(full_protein_root) / info["protein_fn"]
                    f.write(f"{str(full_pocket_path)}\t{str(full_protein_path)}\n")
            print("日志文件写入完成。")
            print("没有文件被删除。要实际删除这些文件，请在命令后添加 --execute-deletion 标志。")

    print("="*50)
    print("脚本运行完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理与PDB残基数不匹配的 embedding 文件。")
    parser.add_argument(
        '--execute-deletion',
        action='store_true',
        help='如果设置此标志，脚本将实际删除不匹配的文件。默认为只报告不删除，并生成txt日志。'
    )
    args = parser.parse_args()

    POCKET_DATA_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
    FULL_PROTEIN_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/crossdocked_v1.1_rmsd1.0"
    OUTPUT_LOG_FILE = "mismatched_embeddings.txt" # 日志将保存在当前目录下

    clean_and_report(POCKET_DATA_ROOT, FULL_PROTEIN_ROOT, OUTPUT_LOG_FILE, args.execute_deletion)
