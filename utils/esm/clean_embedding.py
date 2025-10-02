# safe_delete_embeddings.py
import os
import pickle
from pathlib import Path
from tqdm import tqdm

def safe_delete():
    """
    Safely deletes only the embedding .pt files corresponding to pockets
    listed in the master index.pkl file.
    """
    dataset_root = Path("/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket")
    index_path = dataset_root / 'index.pkl'

    if not index_path.exists():
        print(f"[!] 错误: 未找到主索引文件: {index_path}")
        return

    print(f"[*] 正在从 {index_path} 加载索引...")
    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    # 获取一个唯一的蛋白质口袋PDB文件相对路径列表
    pocket_files_relative = sorted(list(set([item[0] for item in master_index if item[0] is not None])))

    print(f"[*] 索引中包含 {len(pocket_files_relative)} 个独立的口袋文件。")
    print("[*] 开始查找并删除对应的 .pt embedding 文件...")

    deleted_count = 0
    # 使用tqdm显示进度
    for pocket_fn in tqdm(pocket_files_relative, desc="正在清理旧文件"):
        # 构建对应的 .pt 文件路径
        embedding_path = (dataset_root / pocket_fn).with_suffix('.pt')

        # 检查文件是否存在，如果存在则删除
        if embedding_path.exists():
            try:
                os.remove(embedding_path)
                deleted_count += 1
            except OSError as e:
                print(f"\n[!] 删除文件时出错: {embedding_path}, 错误: {e}")

    # 删除可能存在的旧错误日志
    error_log_path = dataset_root / "embedding_errors.log"
    if error_log_path.exists():
        os.remove(error_log_path)
        print("[*] 已删除旧的 embedding_errors.log 文件。")

    print("\n" + "="*50)
    print("  清理完成！")
    print(f"  共删除了 {deleted_count} 个损坏的 embedding .pt 文件。")
    print(f"  像 'split_by_name.pt' 这样的其他文件都安然无恙。")
    print("="*50)

if __name__ == "__main__":
    safe_delete()