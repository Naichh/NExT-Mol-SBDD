import os
import torch
import pickle
from tqdm import tqdm
from rdkit import Chem
import selfies as sf

# ==============================================================================
# 1. 配置路径 (根据你的 bash 脚本自动填好)
# ==============================================================================
DATASET_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
SPLIT_FILE_PATH = os.path.join(DATASET_ROOT, "split_by_name.pt")
INDEX_FILE_PATH = os.path.join(DATASET_ROOT, "index.pkl")

print(f"项目根目录: {DATASET_ROOT}")
print(f"分割文件路径: {SPLIT_FILE_PATH}")
print(f"主索引文件路径: {INDEX_FILE_PATH}")
print("-" * 50)


# ==============================================================================
# 2. 检查函数
# ==============================================================================
def check_data_integrity():
    """
    一个完整的检查流程，模拟 DataModule 的 setup 和 Dataset 的 __getitem__。
    """
    # --- 模拟 DataModule.setup() ---
    print("步骤 1: 正在加载主索引和分割文件...")
    try:
        with open(INDEX_FILE_PATH, 'rb') as f:
            master_index = pickle.load(f)
        
        # 为了快速查找，将主索引转换为一个集合
        master_index_set = set([(p, l) for p, l, _, _ in master_index])
        
        split_data = torch.load(SPLIT_FILE_PATH)
        test_file_pairs = split_data.get('train')
        if not test_file_pairs:
            print("错误: 分割文件中没有找到 'test' 键，或者内容为空。")
            return
            
    except FileNotFoundError as e:
        print(f"错误: 无法加载核心文件: {e}")
        return
    except Exception as e:
        print(f"加载文件时发生未知错误: {e}")
        return

    print(f"加载完成。主索引包含 {len(master_index_set)} 个样本，测试集分割包含 {len(test_file_pairs)} 个样本。")
    print("-" * 50)

    # --- 模拟 __getitem__ 的核心检查逻辑 ---
    print("步骤 2: 开始逐一检查测试集中的每个样本...")
    
    total_samples = len(test_file_pairs)
    passed_samples = 0
    failed_samples = []

    for pocket_fn, ligand_fn in tqdm(test_file_pairs, desc="检查测试集样本"):
        error_reason = None
        
        # 检查 1: 文件名是否能在主索引中匹配上
        key = (pocket_fn, ligand_fn)
        if key not in master_index_set:
            error_reason = "索引不匹配 (文件名在 split 文件中，但不在主 index.pkl 中)"
            failed_samples.append(((pocket_fn, ligand_fn), error_reason))
            continue

        # 构建文件的绝对路径
        pocket_path = os.path.join(DATASET_ROOT, pocket_fn)
        ligand_path = os.path.join(DATASET_ROOT, ligand_fn)
        base_path, _ = os.path.splitext(pocket_path)
        pt_embed_path = base_path + '.pt'

        # 检查 2: 所有相关文件是否存在于磁盘上
        if not os.path.exists(pocket_path):
            error_reason = f"文件丢失: {pocket_path}"
        elif not os.path.exists(ligand_path):
            error_reason = f"文件丢失: {ligand_path}"
        elif not os.path.exists(pt_embed_path):
            error_reason = f"文件丢失: {pt_embed_path}"
        
        if error_reason:
            failed_samples.append(((pocket_fn, ligand_fn), error_reason))
            continue
            
        # 检查 3: 尝试加载所有文件并进行处理
        try:
            # 检查 embedding 文件是否可读
            _ = torch.load(pt_embed_path, map_location='cpu')

            # 检查 ligand 文件是否可被 RDKit 解析
            mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
            if mol is None:
                raise ValueError("RDKit 解析 SDF 文件返回 None")

            # 检查是否能转换为 SELFIES
            smiles = Chem.MolToSmiles(mol)
            _ = sf.encoder(smiles)

        except Exception as e:
            error_reason = f"文件处理失败: {type(e).__name__}: {e}"
            failed_samples.append(((pocket_fn, ligand_fn), error_reason))
            continue

        # 如果所有检查都通过
        passed_samples += 1

    # --- 打印最终报告 ---
    print("-" * 50)
    print("检查完成！最终报告如下：")
    print("=" * 50)
    
    print(f"总计检查测试集样本数: {total_samples}")
    print(f"  - 通过检查的样本数: {passed_samples}")
    print(f"  - 失败的样本数: {len(failed_samples)}")
    
    if failed_samples:
        print("\n--- 失败样本详细列表 ---")
        for i, (files, reason) in enumerate(failed_samples):
            print(f"\n{i+1}. 失败样本:")
            print(f"   - Pocket: {files[0]}")
            print(f"   - Ligand: {files[1]}")
            print(f"   - 失败原因: {reason}")
    
    print("=" * 50)


if __name__ == "__main__":
    # 添加一个关于 torch.load 的警告说明
    print("注意: 脚本中的 torch.load 可能会触发 FutureWarning，这是正常的，不影响检查结果。")
    check_data_integrity()