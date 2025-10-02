import os
import torch
from pathlib import Path

def check_failed_samples_distribution(dataset_root: str, split_file: str, missing_log_file: str):
    """
    检查处理失败的样本在训练集和测试集中的分布情况。
    """
    # --- 1. 检查所有文件是否存在 ---
    if not Path(missing_log_file).exists():
        print(f"错误: 找不到失败样本日志文件: {missing_log_file}")
        return
    if not Path(split_file).exists():
        print(f"错误: 找不到数据集划分文件: {split_file}")
        print("请确保脚本中的 SPLIT_FILE 变量路径正确。")
        return

    # --- 2. 加载数据 ---
    print(f"正在从 {split_file} 加载训练/测试集划分...")
    split = torch.load(split_file)
    # 将划分文件中的口袋相对路径存入 set 中，以提高查找效率
    train_pockets_set = {pocket_fn for pocket_fn, ligand_fn in split['train']}
    test_pockets_set = {pocket_fn for pocket_fn, ligand_fn in split['test']}
    print(f"加载完成。训练集包含 {len(train_pockets_set)} 个口袋，测试集包含 {len(test_pockets_set)} 个口袋。")

    print(f"正在从 {missing_log_file} 加载处理失败的口袋列表...")
    failed_pocket_paths = []
    with open(missing_log_file, 'r') as f:
        next(f) # 跳过表头
        for line in f:
            # 只取第一列（口袋路径）
            abs_path = line.strip().split('\t')[0]
            failed_pocket_paths.append(abs_path)

    total_failed = len(failed_pocket_paths)
    print(f"加载了 {total_failed} 个失败样本的路径。")

    # --- 3. 开始分析 ---
    failed_in_train = 0
    failed_in_test = 0
    failed_in_neither = 0

    train_failure_examples = []
    test_failure_examples = []

    for abs_path in failed_pocket_paths:
        # 将绝对路径转换为相对于 dataset_root 的相对路径，以匹配 split 文件中的格式
        relative_path = os.path.relpath(abs_path, dataset_root)

        if relative_path in train_pockets_set:
            failed_in_train += 1
            if len(train_failure_examples) < 5:
                train_failure_examples.append(relative_path)
        elif relative_path in test_pockets_set:
            failed_in_test += 1
            if len(test_failure_examples) < 5:
                test_failure_examples.append(relative_path)
        else:
            failed_in_neither += 1

    # --- 4. 打印报告 ---
    print("\n" + "="*60)
    print("          失败样本在训练/测试集中的分布报告")
    print("="*60)
    print(f"总计分析失败样本数: {total_failed}")
    print("-" * 60)

    train_percent = (failed_in_train / total_failed) * 100 if total_failed > 0 else 0
    test_percent = (failed_in_test / total_failed) * 100 if total_failed > 0 else 0

    print(f"  - 在【训练集】中找到: {failed_in_train} 个 ({train_percent:.2f}%)")
    print(f"  - 在【测试集】中找到: {failed_in_test} 个 ({test_percent:.2f}%)")
    if failed_in_neither > 0:
        print(f"  - [警告] 在任何集合中都未找到: {failed_in_neither} 个")
    print("="*60)

    if train_failure_examples:
        print("\n部分在【训练集】中失败的样本:")
        for example in train_failure_examples:
            print(f"  - {example}")

    if test_failure_examples:
        print("\n部分在【测试集】中失败的样本:")
        for example in test_failure_examples:
            print(f"  - {example}")

    print("\n分析完成。")

if __name__ == "__main__":
    # --- 请在这里配置您的路径 ---
    # 数据集根目录 (与您主脚本中的 POCKET_DATA_ROOT 保持一致)
    DATASET_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"

    # !!! 关键 !!!
    # 包含训练/测试集划分的文件路径。
    # 根据CrossDocked数据集的惯例，它很可能叫 'split_by_name.pt'。
    # 如果您的文件名不同，请务必修改下面这一行。
    SPLIT_FILE = os.path.join(DATASET_ROOT, "split_by_name.pt")

    # 上一个脚本生成的、记录失败样本的日志文件
    # 请确保此文件与本脚本在同一个目录下，或者提供完整路径
    MISSING_LOG_FILE = "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/missing_embeddings.txt"

    check_failed_samples_distribution(DATASET_ROOT, SPLIT_FILE, MISSING_LOG_FILE)
