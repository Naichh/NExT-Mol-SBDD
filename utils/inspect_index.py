# inspect_index.py
import pickle
from pathlib import Path

def inspect_pickle_file():
    """
    加载并检查 index.pkl 文件的内容和结构。
    """
    # 使用我们确认过的正确路径
    index_path = Path("/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket/index.pkl")

    print(f"[*] 正在读取文件: {index_path}")

    if not index_path.exists():
        print(f"❌ 错误: 文件不存在！")
        return

    with open(index_path, 'rb') as f:
        data = pickle.load(f)

    print("\n" + "="*80)
    print("  索引文件内容分析")
    print("="*80)

    print(f"[*] 文件中数据的顶层类型: {type(data)}")

    if isinstance(data, list):
        print(f"[*] 总条目数量: {len(data)}")
        if len(data) > 0:
            print(f"[*] 每个条目的类型: {type(data[0])}")
            print("\n[*] 显示前5个条目的内容：")
            for i, entry in enumerate(data[:5]):
                print(f"\n--- 条目 {i+1} ---")
                if isinstance(entry, tuple) and len(entry) == 4:
                    pocket_fn, ligand_fn, protein_fn, rmsd = entry
                    print(f"  - 口袋文件 (pocket_fn): '{pocket_fn}'")
                    print(f"  - 配体文件 (ligand_fn): '{ligand_fn}'")
                    print(f"  - 完整蛋白 (protein_fn): '{protein_fn}'")
                    print(f"  - RMSD值: {rmsd}")
                else:
                    print(f"  - 无法解析的条目内容: {entry}")
    else:
        print("[*] 文件内容不是一个列表，无法按条目显示。")

    print("\n" + "="*80)

if __name__ == "__main__":
    inspect_pickle_file()