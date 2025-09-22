# verify_data_consistency.py
import pickle
from pathlib import Path
import random

def verify_consistency():
    """
    使用现有的索引文件，随机抽取一个样本，
    验证 'crossdocked_pocket' 目录中的PDB文件是口袋，
    而 'crossdocked_v1.1_rmsd1.0' 目录中的是完整蛋白。
    """
    print("="*60)
    print("  开始验证数据集一致性...")
    print("="*60)

    try:
        # --- 1. 定义两个数据集的根目录 ---
        base_dir1 = Path("/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd")
        base_dir2=Path("/data/share/liuzhiyuan/nai/NExT-Mol/datasets")
        full_protein_root = base_dir2 / "crossdocked_v1.1_rmsd1.0"

        pocket_data_root = base_dir1 / "crossdocked_pocket"
        
        index_path = pocket_data_root / "index.pkl"

        # --- 2. 检查所有路径是否存在 ---
        if not pocket_data_root.exists():
            print(f"❌ 错误: 口袋数据集目录不存在: {pocket_data_root}")
            return
        if not full_protein_root.exists():
            print(f"❌ 错误: 完整蛋白质数据集目录不存在: {full_protein_root}")
            return
        if not index_path.exists():
            print(f"❌ 错误: 索引文件不存在: {index_path}")
            return
            
        print(f"[*] 成功找到所有必要的目录和文件。")
        
        # --- 3. 加载索引并随机抽取一个样本 ---
        print(f"[*] 正在从 {index_path} 加载索引...")
        with open(index_path, 'rb') as f:
            master_index = pickle.load(f)
        
        # 随机抽取一个有效的样本（确保文件名不为空）
        sample_entry = None
        while sample_entry is None:
            entry = random.choice(master_index)
            if entry[0] is not None and entry[2] is not None:
                sample_entry = entry
        
        pocket_fn, _, protein_fn, _ = sample_entry
        print(f"[*] 已随机抽取样本: 口袋='{pocket_fn}', 完整蛋白='{protein_fn}'")

        # --- 4. 构建两个PDB文件的完整路径 ---
        pocket_pdb_path = pocket_data_root / pocket_fn
        full_pdb_path = full_protein_root / protein_fn
        
        print(f"    -> 口袋文件路径: {pocket_pdb_path}")
        print(f"    -> 完整蛋白文件路径: {full_pdb_path}")
        
        if not pocket_pdb_path.exists() or not full_pdb_path.exists():
            print("\n❌ 错误: 索引中指向的文件至少有一个不存在！请检查您的数据和索引文件是否匹配。")
            return

        # --- 5. 读取文件并比较大小 (以行数为标准) ---
        with open(pocket_pdb_path, 'r') as f:
            pocket_lines = f.readlines()
        
        with open(full_pdb_path, 'r') as f:
            full_protein_lines = f.readlines()
            
        num_pocket_lines = len(pocket_lines)
        num_full_lines = len(full_protein_lines)

        print("\n" + "-"*60)
        print("  比较结果:")
        print(f"  口袋PDB文件行数: {num_pocket_lines}")
        print(f"  完整蛋白PDB文件行数: {num_full_lines}")
        print("-"*60)
        
        # --- 6. 得出结论 ---
        if num_full_lines > num_pocket_lines and num_pocket_lines > 0:
            print("\n✅ 验证成功！")
            print("   这证实了 'crossdocked_v1.1_rmsd1.0' 目录中是完整的蛋白质，")
            print("   而 'crossdocked_pocket' 目录中是对应的、经过裁剪的口袋。")
            print("   并且，您的 'index.pkl' 能够正确地将它们链接起来。")
        else:
            print("\n❌ 验证失败！")
            print("   文件大小关系不符合预期。请检查您的文件内容和目录结构。")
            
        print("="*60)

    except Exception as e:
        print(f"\n❌ 验证过程中发生未知错误: {e}")

if __name__ == "__main__":
    verify_consistency()