import os

def find_empty_files(root_dir, extension=".pkl"):
    """
    遍历指定目录，查找所有指定扩展名的、大小为0的文件。
    """
    empty_files = []
    print(f"--- 正在扫描目录: {root_dir} ---")

    # os.walk会遍历所有子文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                # 检查文件大小是否为0
                if os.path.getsize(file_path) == 0:
                    empty_files.append(file_path)

    if not empty_files:
        print("\n[✓] 恭喜！没有发现空的 .pkl 文件。")
    else:
        print(f"\n[!] 警告！发现了 {len(empty_files)} 个空的 .pkl 文件:")
        for file_path in empty_files:
            print(f"  - {file_path}")

    return empty_files

if __name__ == "__main__":
    # !!! 将此路径修改为你的crossdocked_pocket数据集的根目录 !!!
    dataset_root = "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket"

    empty_pkls = find_empty_files(dataset_root)

    if empty_pkls:
        print("\n--- 建议操作 ---")
        print("你可以手动删除这些文件，或者重新运行你的预处理脚本来修复它们。")
